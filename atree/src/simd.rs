use std::mem::{MaybeUninit, size_of};

use crate::output::QueryOutput;
use crate::scalar::{IdStorage, Scalar};

#[cfg(feature = "simd-compress")]
use wide::CmpLe;

#[derive(Debug, Clone, Copy)]
#[repr(Rust, align(64))]
pub struct PDVec<const D: usize, const W: usize, F: Scalar = f32, I: IdStorage = u32> {
    lanes: [[F; W]; D],
    sqared_half: [F; W],
    ids: [I; W],
}

impl<const D: usize, const W: usize, F: Scalar, I: IdStorage> PDVec<D, W, F, I> {
    // Compile-time guard: f64 with W=16 is not supported (would need 1024-bit registers)
    const _GUARD: () = assert!(
        !(size_of::<F>() == 8 && W == 16),
        "f64 with W=16 is not supported"
    );

    pub fn new(vecs: impl Iterator<Item = ([F; D], usize)>) -> Self {
        let _ = Self::_GUARD;
        let mut inf = Self::inf();
        for (i, (vec, id)) in vecs.enumerate().take(W) {
            inf.sqared_half[i] = vec.iter().copied().map(|x| x * x).sum::<F>() * F::HALF;
            inf.ids[i] = I::from_usize(id);
            for j in 0..D {
                inf.lanes[j][i] = vec[j];
            }
        }
        inf
    }

    pub fn from_slices(vecs: &[[F; D]], ids: &[usize]) -> Self {
        Self::new(vecs.iter().copied().zip(ids.iter().copied()))
    }

    pub fn inf() -> Self {
        let _ = Self::_GUARD;
        Self {
            lanes: [[F::NAN; W]; D],
            sqared_half: [F::INFINITY; W],
            ids: [I::SENTINEL; W],
        }
    }

    #[inline(always)]
    pub fn dist_squared(&self, pos: [F; D]) -> [F; W] {
        let diff = std::array::from_fn(|i| self.lanes[0][i] - pos[0]);
        let mut acc = diff.map(|x| x * x);
        for j in 1..D {
            let diff: [_; W] = std::array::from_fn(|i| self.lanes[j][i] - pos[j]);
            acc = std::array::from_fn(|i| diff[i].mul_add(diff[i], acc[i]));
        }
        acc
    }

    #[inline(always)]
    pub fn dist_half_squared(&self, pos: [F; D], squared_half: F) -> [F; W] {
        let mut acc1: [F; W] = std::array::from_fn(|i| self.sqared_half[i]);
        let mut acc2: [F; W] = std::array::from_fn(|_| squared_half);

        for j in (0..D).step_by(2) {
            acc1 = std::array::from_fn(|i| self.lanes[j][i].mul_add(-pos[j], acc1[i]));
            if j + 1 < D {
                acc2 = std::array::from_fn(|i| self.lanes[j + 1][i].mul_add(-pos[j + 1], acc2[i]));
            }
        }

        std::array::from_fn(|i| acc1[i] + acc2[i])
    }

    // ── Compress: mask + compress IDs & distances → arrays ──

    #[inline(always)]
    pub fn compress(&self, distances: [F; W], threshold: F) -> (usize, [I; W], [F; W]) {
        // f32 SIMD paths
        if size_of::<F>() == 4 && size_of::<I>() == 4 {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx512f") {
                    if W == 8 {
                        return unsafe { self.compress_avx512_f32_u32_8(distances, threshold) };
                    }
                    if W == 16 {
                        return unsafe { self.compress_avx512_f32_u32_16(distances, threshold) };
                    }
                }
            }
            #[cfg(feature = "simd-compress")]
            {
                if W == 8 {
                    return self.compress_wide_f32_u32_8(distances, threshold);
                }
                if W == 16 {
                    return self.compress_wide_f32_u32_16(distances, threshold);
                }
            }
        }

        // f32 with u64 IDs — AVX-512
        if size_of::<F>() == 4 && size_of::<I>() == 8 {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx512f") {
                    if W == 8 {
                        return unsafe { self.compress_avx512_f32_u64_8(distances, threshold) };
                    }
                    if W == 16 {
                        return unsafe { self.compress_avx512_f32_u64_16(distances, threshold) };
                    }
                }
            }
        }

        // f64 SIMD paths
        if size_of::<F>() == 8 {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx512f") {
                    if W == 4 {
                        return unsafe { self.compress_avx512_f64_4(distances, threshold) };
                    }
                    if W == 8 {
                        return unsafe { self.compress_avx512_f64_8(distances, threshold) };
                    }
                }
            }
            #[cfg(feature = "simd-compress")]
            {
                if W == 4 {
                    return self.compress_wide_f64_4(distances, threshold);
                }
                if W == 8 {
                    return self.compress_wide_f64_8(distances, threshold);
                }
            }
        }

        self.compress_scalar(distances, threshold)
    }

    /// Generic compare: compress + type-specific store via QueryOutput.
    #[inline(always)]
    pub fn compare_into<O: QueryOutput<I, F>>(
        &self,
        distances: [F; W],
        threshold: F,
        results: &mut [MaybeUninit<O>; W],
    ) -> usize {
        let (count, ids, dists) = self.compress(distances, threshold);
        O::store_compressed(count, &ids, &dists, results)
    }
    /// Generic compare: compress + type-specific store via QueryOutput.
    #[inline(always)]
    pub fn compare_into_initialized<O: QueryOutput<I, F> + Copy>(
        &self,
        distances: [F; W],
        threshold: F,
        results: &mut [O; W],
    ) -> usize {
        let (count, ids, dists) = self.compress(distances, threshold);
        O::store_compressed(count, &ids, &dists, unsafe {
            std::mem::transmute::<&mut [O; W], &mut [MaybeUninit<O>; W]>(results)
        })
    }

    /// Backward-compatible compare producing `usize` IDs.
    #[inline(always)]
    pub fn compare(
        &self,
        distances: [F; W],
        squared_radius_half: F,
        results: &mut [MaybeUninit<usize>; W],
    ) -> usize
    where
        usize: QueryOutput<I, F>,
    {
        self.compare_into(distances, squared_radius_half, results)
    }

    // ── Scalar compress (fully generic) ─────────────────────────────

    #[inline(never)]
    fn compress_scalar(&self, distances: [F; W], threshold: F) -> (usize, [I; W], [F; W]) {
        let mut ids = [I::default(); W];
        let mut dists = [F::default(); W];
        let mut count = 0;
        for i in 0..W {
            ids[count] = self.ids[i];
            dists[count] = distances[i];
            count += (distances[i] <= threshold) as usize;
        }
        (count, ids, dists)
    }

    // ── AVX-512 f32+u32 compress (original paths, transmuted) ───────

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn compress_avx512_f32_u32_8(
        &self,
        distances: [F; W],
        threshold: F,
    ) -> (usize, [I; W], [F; W]) {
        use std::arch::x86_64::*;
        unsafe {
            let dist = _mm256_loadu_ps(distances.as_ptr() as *const f32);
            let thresh = _mm256_set1_ps(*((&threshold) as *const F as *const f32));
            let mask = _mm256_cmp_ps_mask::<_CMP_LE_OS>(dist, thresh);

            let ids = _mm256_loadu_epi32(self.ids.as_ptr() as *const i32);
            let compressed_ids = _mm256_maskz_compress_epi32(mask, ids);
            let compressed_dists = _mm256_maskz_compress_ps(mask, dist);

            let mut id_arr = [I::default(); W];
            let mut dist_arr = [F::default(); W];
            _mm256_storeu_epi32(id_arr.as_mut_ptr() as *mut i32, compressed_ids);
            _mm256_storeu_ps(dist_arr.as_mut_ptr() as *mut f32, compressed_dists);

            (mask.count_ones() as usize, id_arr, dist_arr)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn compress_avx512_f32_u32_16(
        &self,
        distances: [F; W],
        threshold: F,
    ) -> (usize, [I; W], [F; W]) {
        use std::arch::x86_64::*;
        unsafe {
            let dist = _mm512_loadu_ps(distances.as_ptr() as *const f32);
            let thresh = _mm512_set1_ps(*((&threshold) as *const F as *const f32));
            let mask = _mm512_cmple_ps_mask(dist, thresh);

            let ids = _mm512_loadu_epi32(self.ids.as_ptr() as *const i32);
            let compressed_ids = _mm512_maskz_compress_epi32(mask, ids);
            let compressed_dists = _mm512_maskz_compress_ps(mask, dist);

            let mut id_arr = [I::default(); W];
            let mut dist_arr = [F::default(); W];
            _mm512_storeu_epi32(id_arr.as_mut_ptr() as *mut i32, compressed_ids);
            _mm512_storeu_ps(dist_arr.as_mut_ptr() as *mut f32, compressed_dists);

            (mask.count_ones() as usize, id_arr, dist_arr)
        }
    }

    // ── AVX-512 f32+u64 compress ────────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn compress_avx512_f32_u64_8(
        &self,
        distances: [F; W],
        threshold: F,
    ) -> (usize, [I; W], [F; W]) {
        use std::arch::x86_64::*;
        unsafe {
            let dist = _mm256_loadu_ps(distances.as_ptr() as *const f32);
            let thresh = _mm256_set1_ps(*((&threshold) as *const F as *const f32));
            let mask = _mm256_cmp_ps_mask::<_CMP_LE_OS>(dist, thresh);

            // Compress f32 distances
            let compressed_dists = _mm256_maskz_compress_ps(mask, dist);
            let mut dist_arr = [F::default(); W];
            _mm256_storeu_ps(dist_arr.as_mut_ptr() as *mut f32, compressed_dists);

            // Compress u64 IDs: load as 512-bit, compress as epi64
            let ids = _mm512_loadu_epi64(self.ids.as_ptr() as *const i64);
            let compressed_ids = _mm512_maskz_compress_epi64(mask, ids);
            let mut id_arr = [I::default(); W];
            _mm512_storeu_epi64(id_arr.as_mut_ptr() as *mut i64, compressed_ids);

            (mask.count_ones() as usize, id_arr, dist_arr)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn compress_avx512_f32_u64_16(
        &self,
        distances: [F; W],
        threshold: F,
    ) -> (usize, [I; W], [F; W]) {
        use std::arch::x86_64::*;
        unsafe {
            let dist = _mm512_loadu_ps(distances.as_ptr() as *const f32);
            let thresh = _mm512_set1_ps(*((&threshold) as *const F as *const f32));
            let mask = _mm512_cmple_ps_mask(dist, thresh);

            // Compress f32 distances
            let compressed_dists = _mm512_maskz_compress_ps(mask, dist);
            let mut dist_arr = [F::default(); W];
            _mm512_storeu_ps(dist_arr.as_mut_ptr() as *mut f32, compressed_dists);

            // Compress u64 IDs: two 512-bit halves (8 u64s each)
            let ids_lo = _mm512_loadu_epi64(self.ids.as_ptr() as *const i64);
            let ids_hi = _mm512_loadu_epi64((self.ids.as_ptr() as *const i64).add(8));
            let mask_lo = (mask & 0xFF) as u8;
            let mask_hi = (mask >> 8) as u8;
            let compressed_lo = _mm512_maskz_compress_epi64(mask_lo, ids_lo);
            let compressed_hi = _mm512_maskz_compress_epi64(mask_hi, ids_hi);

            let mut id_arr = [I::default(); W];
            let count_lo = mask_lo.count_ones() as usize;
            _mm512_storeu_epi64(id_arr.as_mut_ptr() as *mut i64, compressed_lo);
            _mm512_storeu_epi64(
                (id_arr.as_mut_ptr() as *mut i64).add(count_lo),
                compressed_hi,
            );

            (mask.count_ones() as usize, id_arr, dist_arr)
        }
    }

    // ── AVX-512 f64 compress ────────────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn compress_avx512_f64_4(
        &self,
        distances: [F; W],
        threshold: F,
    ) -> (usize, [I; W], [F; W]) {
        use std::arch::x86_64::*;
        unsafe {
            let dist = _mm256_loadu_pd(distances.as_ptr() as *const f64);
            let thresh = _mm256_set1_pd(*((&threshold) as *const F as *const f64));
            let mask = _mm256_cmp_pd_mask::<_CMP_LE_OS>(dist, thresh);

            let compressed_dists = _mm256_maskz_compress_pd(mask, dist);
            let mut dist_arr = [F::default(); W];
            _mm256_storeu_pd(dist_arr.as_mut_ptr() as *mut f64, compressed_dists);

            let mut id_arr = [I::default(); W];
            if size_of::<I>() == 4 {
                // u32 IDs: 128-bit (4 × u32)
                let ids = _mm_loadu_epi32(self.ids.as_ptr() as *const i32);
                let compressed_ids = _mm_maskz_compress_epi32(mask, ids);
                _mm_storeu_epi32(id_arr.as_mut_ptr() as *mut i32, compressed_ids);
            } else {
                // u64 IDs: 256-bit (4 × u64)
                let ids = _mm256_loadu_epi64(self.ids.as_ptr() as *const i64);
                let compressed_ids = _mm256_maskz_compress_epi64(mask, ids);
                _mm256_storeu_epi64(id_arr.as_mut_ptr() as *mut i64, compressed_ids);
            }

            (mask.count_ones() as usize, id_arr, dist_arr)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn compress_avx512_f64_8(
        &self,
        distances: [F; W],
        threshold: F,
    ) -> (usize, [I; W], [F; W]) {
        use std::arch::x86_64::*;
        unsafe {
            let dist = _mm512_loadu_pd(distances.as_ptr() as *const f64);
            let thresh = _mm512_set1_pd(*((&threshold) as *const F as *const f64));
            let mask = _mm512_cmple_pd_mask(dist, thresh);

            let compressed_dists = _mm512_maskz_compress_pd(mask, dist);
            let mut dist_arr = [F::default(); W];
            _mm512_storeu_pd(dist_arr.as_mut_ptr() as *mut f64, compressed_dists);

            let mut id_arr = [I::default(); W];
            if size_of::<I>() == 4 {
                // u32 IDs: 256-bit (8 × u32)
                let ids = _mm256_loadu_epi32(self.ids.as_ptr() as *const i32);
                let compressed_ids = _mm256_maskz_compress_epi32(mask, ids);
                _mm256_storeu_epi32(id_arr.as_mut_ptr() as *mut i32, compressed_ids);
            } else {
                // u64 IDs: 512-bit (8 × u64)
                let ids = _mm512_loadu_epi64(self.ids.as_ptr() as *const i64);
                let compressed_ids = _mm512_maskz_compress_epi64(mask, ids);
                _mm512_storeu_epi64(id_arr.as_mut_ptr() as *mut i64, compressed_ids);
            }

            (mask.count_ones() as usize, id_arr, dist_arr)
        }
    }

    // ── wide crate f32+u32 compress ─────────────────────────────────

    #[cfg(feature = "simd-compress")]
    fn compress_wide_f32_u32_8(&self, distances: [F; W], threshold: F) -> (usize, [I; W], [F; W]) {
        use std::array::from_fn;

        use simd_lookup::simd_compress::compress_u32x8;
        use wide::{f32x8, u32x8};

        let dist = f32x8::new(from_fn(|i| unsafe {
            *(&distances[i] as *const F as *const f32)
        }));
        let threshold_f32 = unsafe { *(&threshold as *const F as *const f32) };
        let threshold_v = f32x8::splat(threshold_f32);
        let mask = dist.simd_le(threshold_v).to_bitmask() as u8;

        let ids_v = u32x8::from(from_fn::<u32, 8, _>(|i| unsafe {
            *(&self.ids[i] as *const I as *const u32)
        }));
        let (compressed, count) = compress_u32x8(ids_v, mask);

        let mut id_arr = [I::default(); W];
        let comp = compressed.to_array();
        for i in 0..8 {
            unsafe {
                *(&mut id_arr[i] as *mut I as *mut u32) = comp[i];
            }
        }

        let mut dist_arr = [F::default(); W];
        let mut j = 0;
        for (i, &dist) in distances.iter().enumerate() {
            dist_arr[j] = dist;
            j += ((mask >> i) & 1) as usize;
        }

        (count, id_arr, dist_arr)
    }

    #[cfg(feature = "simd-compress")]
    fn compress_wide_f32_u32_16(&self, distances: [F; W], threshold: F) -> (usize, [I; W], [F; W]) {
        use simd_lookup::simd_compress::compress_u32x8;
        use simd_lookup::wide_utils::SimdSplit;
        use wide::{f32x8, u32x16};

        let dist_lo = f32x8::new(std::array::from_fn(|i| unsafe {
            *(&distances[i] as *const F as *const f32)
        }));
        let dist_hi = f32x8::new(std::array::from_fn(|i| unsafe {
            *(&distances[8 + i] as *const F as *const f32)
        }));
        let threshold_f32 = unsafe { *(&threshold as *const F as *const f32) };
        let threshold_v = f32x8::splat(threshold_f32);
        let mask_lo = dist_lo.simd_le(threshold_v).to_bitmask() as u8;
        let mask_hi = dist_hi.simd_le(threshold_v).to_bitmask() as u8;

        let ids_v = u32x16::from(std::array::from_fn::<u32, 16, _>(|i| unsafe {
            *(&self.ids[i] as *const I as *const u32)
        }));
        let (ids_lo, ids_hi) = ids_v.split_low_high();

        let (comp_lo, count_lo) = compress_u32x8(ids_lo, mask_lo);
        let (comp_hi, count_hi) = compress_u32x8(ids_hi, mask_hi);

        let mut id_arr = [I::default(); W];
        let arr_lo = comp_lo.to_array();
        let arr_hi = comp_hi.to_array();
        for i in 0..8 {
            unsafe {
                *(&mut id_arr[i] as *mut I as *mut u32) = arr_lo[i];
            }
        }
        for i in 0..8 {
            unsafe {
                *(&mut id_arr[count_lo + i] as *mut I as *mut u32) = arr_hi[i];
            }
        }

        let mut dist_arr = [F::default(); W];
        let mut j = 0;
        for (i, &dist) in distances.iter().enumerate() {
            dist_arr[j] = dist;
            j += ((mask_lo >> i) & 1) as usize;
        }
        for i in 0..8 {
            dist_arr[j] = distances[8 + i];
            j += ((mask_hi >> i) & 1) as usize;
        }

        (count_lo + count_hi, id_arr, dist_arr)
    }

    // ── wide crate f64 compress ─────────────────────────────────────

    #[cfg(feature = "simd-compress")]
    fn compress_wide_f64_4(&self, distances: [F; W], threshold: F) -> (usize, [I; W], [F; W]) {
        use wide::f64x4;

        let dist = f64x4::new(std::array::from_fn(|i| unsafe {
            *(&distances[i] as *const F as *const f64)
        }));
        let threshold_f64 = unsafe { *(&threshold as *const F as *const f64) };
        let threshold_v = f64x4::splat(threshold_f64);
        let mask_bits = dist.simd_le(threshold_v).to_bitmask() as u8;

        let mut id_arr = [I::default(); W];
        let mut dist_arr = [F::default(); W];
        let mut j = 0;
        for i in 0..4 {
            id_arr[j] = self.ids[i];
            dist_arr[j] = distances[i];
            j += ((mask_bits >> i) & 1) as usize;
        }

        (j, id_arr, dist_arr)
    }

    #[cfg(feature = "simd-compress")]
    fn compress_wide_f64_8(&self, distances: [F; W], threshold: F) -> (usize, [I; W], [F; W]) {
        use wide::f64x4;

        let dist_lo = f64x4::new(std::array::from_fn(|i| unsafe {
            *(&distances[i] as *const F as *const f64)
        }));
        let dist_hi = f64x4::new(std::array::from_fn(|i| unsafe {
            *(&distances[4 + i] as *const F as *const f64)
        }));
        let threshold_f64 = unsafe { *(&threshold as *const F as *const f64) };
        let threshold_v = f64x4::splat(threshold_f64);
        let mask_lo = dist_lo.simd_le(threshold_v).to_bitmask() as u8;
        let mask_hi = dist_hi.simd_le(threshold_v).to_bitmask() as u8;

        let mut id_arr = [I::default(); W];
        let mut dist_arr = [F::default(); W];
        let mut j = 0;
        for i in 0..4 {
            id_arr[j] = self.ids[i];
            dist_arr[j] = distances[i];
            j += ((mask_lo >> i) & 1) as usize;
        }
        for i in 0..4 {
            id_arr[j] = self.ids[4 + i];
            dist_arr[j] = distances[4 + i];
            j += ((mask_hi >> i) & 1) as usize;
        }

        (j, id_arr, dist_arr)
    }
}

/// Shared compress dispatch for use by DynPDVec.
///
/// Constructs a temporary `PDVec<1, W>` proxy (compress only accesses `ids`,
/// not `lanes`) and delegates to its SIMD-optimized compress.
pub(crate) fn compress_with_ids<const W: usize, F: Scalar, I: IdStorage>(
    ids: [I; W],
    distances: [F; W],
    threshold: F,
) -> (usize, [I; W], [F; W]) {
    let proxy = PDVec::<1, W, F, I> {
        lanes: [[F::NAN; W]],
        sqared_half: [F::INFINITY; W],
        ids,
    };
    proxy.compress(distances, threshold)
}
