use std::mem::MaybeUninit;

use crate::output::QueryOutput;

#[cfg(feature = "simd-compress")]
use wide::CmpLe;

#[derive(Debug, Clone, Copy)]
#[repr(Rust, align(32))]
pub struct PDVec<const D: usize, const W: usize> {
    lanes: [[f32; W]; D],
    sqared_half: [f32; W],
    ids: [u32; W],
}

impl<const D: usize, const W: usize> PDVec<D, W> {
    pub fn new(vecs: impl Iterator<Item = ([f32; D], usize)>) -> Self {
        let mut inf = Self::inf();
        for (i, (vec, id)) in vecs.enumerate().take(W) {
            inf.sqared_half[i] = vec.iter().map(|x| x * x).sum::<f32>() / 2.;
            inf.ids[i] = id as u32;
            for j in 0..D {
                inf.lanes[j][i] = vec[j];
            }
        }
        inf
    }
    pub fn from_slices(vecs: &[[f32; D]], ids: &[usize]) -> Self {
        Self::new(vecs.iter().copied().zip(ids.iter().copied()))
    }
    pub fn inf() -> Self {
        Self {
            lanes: [[f32::NAN; W]; D],
            sqared_half: [f32::INFINITY; W],
            ids: [4242424242; W],
        }
    }

    #[inline(always)]
    pub fn dist_squared(&self, pos: [f32; D]) -> [f32; W] {
        let diff = std::array::from_fn(|i| self.lanes[0][i] - pos[0]);
        let mut acc = diff.map(|x| x * x);
        for j in 1..D {
            let diff: [_; W] = std::array::from_fn(|i| self.lanes[j][i] - pos[j]);
            acc = std::array::from_fn(|i| diff[i].mul_add(diff[i], acc[i]));
        }

        acc
    }
    #[inline(always)]
    pub fn dist_half_squared(&self, pos: [f32; D], squared_half: f32) -> [f32; W] {
        let mut acc1: [f32; W] = std::array::from_fn(|i| self.sqared_half[i]);
        let mut acc2: [f32; W] = std::array::from_fn(|_| squared_half);

        for j in (0..D).step_by(2) {
            acc1 = std::array::from_fn(|i| self.lanes[j][i].mul_add(-pos[j], acc1[i]));
            if j + 1 < D {
                acc2 = std::array::from_fn(|i| self.lanes[j + 1][i].mul_add(-pos[j + 1], acc2[i]));
            }
        }

        std::array::from_fn(|i| acc1[i] + acc2[i])
    }

    // ── Non-generic compress: mask + compress IDs & distances → arrays ──

    /// Compress matching elements into packed arrays.
    /// Returns `(count, compressed_ids, compressed_distances)`.
    #[inline(always)]
    pub fn compress(
        &self,
        distances: [f32; W],
        threshold: f32,
    ) -> (usize, [u32; W], [f32; W]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                if W == 8 {
                    return unsafe { self.compress_avx512_8(distances, threshold) };
                }
                if W == 16 {
                    return unsafe { self.compress_avx512_16(distances, threshold) };
                }
            }
        }
        #[cfg(feature = "simd-compress")]
        {
            if W == 8 {
                return self.compress_wide_8(distances, threshold);
            }
            if W == 16 {
                return self.compress_wide_16(distances, threshold);
            }
        }
        self.compress_scalar(distances, threshold)
    }

    /// Generic compare: compress + type-specific store via QueryOutput.
    #[inline(always)]
    pub fn compare_into<O: QueryOutput>(
        &self,
        distances: [f32; W],
        threshold: f32,
        results: &mut [MaybeUninit<O>; W],
    ) -> usize {
        let (count, ids, dists) = self.compress(distances, threshold);
        O::store_compressed(count, &ids, &dists, results)
    }

    /// Backward-compatible compare producing `usize` IDs.
    #[inline(always)]
    pub fn compare(
        &self,
        distances: [f32; W],
        squared_radius_half: f32,
        results: &mut [MaybeUninit<usize>; W],
    ) -> usize {
        self.compare_into(distances, squared_radius_half, results)
    }

    // ── AVX-512 compress ─────────────────────────────────────────────

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn compress_avx512_8(
        &self,
        distances: [f32; W],
        threshold: f32,
    ) -> (usize, [u32; W], [f32; W]) {
        use std::arch::x86_64::*;
        unsafe {
            let dist = _mm256_loadu_ps(distances.as_ptr());
            let thresh = _mm256_set1_ps(threshold);
            let mask = _mm256_cmp_ps_mask::<_CMP_LE_OS>(dist, thresh);

            let ids = _mm256_loadu_epi32(self.ids.as_ptr() as *const i32);
            let compressed_ids = _mm256_maskz_compress_epi32(mask, ids);
            let compressed_dists = _mm256_maskz_compress_ps(mask, dist);

            let mut id_arr = [0u32; W];
            let mut dist_arr = [0f32; W];
            _mm256_storeu_epi32(id_arr.as_mut_ptr() as *mut i32, compressed_ids);
            _mm256_storeu_ps(dist_arr.as_mut_ptr(), compressed_dists);

            (mask.count_ones() as usize, id_arr, dist_arr)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn compress_avx512_16(
        &self,
        distances: [f32; W],
        threshold: f32,
    ) -> (usize, [u32; W], [f32; W]) {
        use std::arch::x86_64::*;
        unsafe {
            let dist = _mm512_loadu_ps(distances.as_ptr());
            let thresh = _mm512_set1_ps(threshold);
            let mask = _mm512_cmple_ps_mask(dist, thresh);

            let ids = _mm512_loadu_epi32(self.ids.as_ptr() as *const i32);
            let compressed_ids = _mm512_maskz_compress_epi32(mask, ids);
            let compressed_dists = _mm512_maskz_compress_ps(mask, dist);

            let mut id_arr = [0u32; W];
            let mut dist_arr = [0f32; W];
            _mm512_storeu_epi32(id_arr.as_mut_ptr() as *mut i32, compressed_ids);
            _mm512_storeu_ps(dist_arr.as_mut_ptr(), compressed_dists);

            (mask.count_ones() as usize, id_arr, dist_arr)
        }
    }

    // ── wide crate compress (cross-platform fallback) ────────────────

    #[cfg(feature = "simd-compress")]
    fn compress_wide_8(
        &self,
        distances: [f32; W],
        threshold: f32,
    ) -> (usize, [u32; W], [f32; W]) {
        use simd_lookup::simd_compress::compress_u32x8;
        use wide::{f32x8, u32x8};

        let dist = f32x8::new(std::array::from_fn(|i| distances[i]));
        let threshold_v = f32x8::splat(threshold);
        let mask = dist.simd_le(threshold_v).to_bitmask() as u8;

        let ids_v = u32x8::from(std::array::from_fn::<u32, 8, _>(|i| self.ids[i]));
        let (compressed, count) = compress_u32x8(ids_v, mask);

        let mut id_arr = [0u32; W];
        let comp = compressed.to_array();
        for i in 0..8 {
            id_arr[i] = comp[i];
        }

        // Branchless scalar distance compression (no compress_f32x8 available)
        let mut dist_arr = [0f32; W];
        let mut j = 0;
        for i in 0..8 {
            dist_arr[j] = distances[i];
            j += ((mask >> i) & 1) as usize;
        }

        (count, id_arr, dist_arr)
    }

    #[cfg(feature = "simd-compress")]
    fn compress_wide_16(
        &self,
        distances: [f32; W],
        threshold: f32,
    ) -> (usize, [u32; W], [f32; W]) {
        use simd_lookup::simd_compress::compress_u32x8;
        use simd_lookup::wide_utils::SimdSplit;
        use wide::{f32x8, u32x16};

        let dist_lo = f32x8::new(std::array::from_fn(|i| distances[i]));
        let dist_hi = f32x8::new(std::array::from_fn(|i| distances[8 + i]));
        let threshold_v = f32x8::splat(threshold);
        let mask_lo = dist_lo.simd_le(threshold_v).to_bitmask() as u8;
        let mask_hi = dist_hi.simd_le(threshold_v).to_bitmask() as u8;

        let ids_v = u32x16::from(std::array::from_fn::<u32, 16, _>(|i| self.ids[i]));
        let (ids_lo, ids_hi) = ids_v.split_low_high();

        let (comp_lo, count_lo) = compress_u32x8(ids_lo, mask_lo);
        let (comp_hi, count_hi) = compress_u32x8(ids_hi, mask_hi);

        let mut id_arr = [0u32; W];
        let arr_lo = comp_lo.to_array();
        let arr_hi = comp_hi.to_array();
        for i in 0..8 {
            id_arr[i] = arr_lo[i];
        }
        for i in 0..8 {
            id_arr[count_lo + i] = arr_hi[i];
        }

        // Branchless scalar distance compression
        let mut dist_arr = [0f32; W];
        let mut j = 0;
        for i in 0..8 {
            dist_arr[j] = distances[i];
            j += ((mask_lo >> i) & 1) as usize;
        }
        for i in 0..8 {
            dist_arr[j] = distances[8 + i];
            j += ((mask_hi >> i) & 1) as usize;
        }

        (count_lo + count_hi, id_arr, dist_arr)
    }

    // ── Scalar compress ──────────────────────────────────────────────

    #[inline(never)]
    fn compress_scalar(
        &self,
        distances: [f32; W],
        threshold: f32,
    ) -> (usize, [u32; W], [f32; W]) {
        let mut ids = [0u32; W];
        let mut dists = [0f32; W];
        let mut count = 0;
        for i in 0..W {
            ids[count] = self.ids[i];
            dists[count] = distances[i];
            count += (distances[i] <= threshold) as usize;
        }
        (count, ids, dists)
    }
}
