use std::mem::MaybeUninit;

#[cfg(feature = "simd-compress")]
use wide::CmpLe;

use crate::dvec::DVec;

#[derive(Debug, Clone, Copy)]
#[repr(Rust, align(32))]
pub struct PDVec<const D: usize, const W: usize> {
    lanes: [[f32; W]; D],
    sqared_half: [f32; W],
    ids: [u32; W],
}

impl<const D: usize, const W: usize> PDVec<D, W> {
    pub fn new(vecs: impl Iterator<Item = (DVec<D>, usize)>) -> Self {
        let mut inf = Self::inf();
        for (i, (vec, id)) in vecs.enumerate().take(W) {
            inf.sqared_half[i] = vec.magnitude_squared() / 2.;
            inf.ids[i] = id as u32;
            for j in 0..D {
                inf.lanes[j][i] = vec.components[j];
            }
        }
        inf
    }
    pub fn from_slices(vecs: &[DVec<D>], ids: &[usize]) -> Self {
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
    pub fn dist_squared(&self, pos: DVec<D>) -> [f32; W] {
        let diff = std::array::from_fn(|i| self.lanes[0][i] - pos[0]);
        let mut acc = diff.map(|x| x * x);
        for j in 1..D {
            let diff: [_; W] = std::array::from_fn(|i| self.lanes[j][i] - pos[j]);
            acc = std::array::from_fn(|i| diff[i].mul_add(diff[i], acc[i]));
        }

        acc
    }
    #[inline(always)]
    pub fn dist_half_squared(&self, pos: DVec<D>, squared_half: f32) -> [f32; W] {
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

    #[inline(always)]
    pub fn compare(
        &self,
        distances: [f32; W],
        squared_radius_half: f32,
        results: &mut [MaybeUninit<usize>; W],
    ) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                if W == 8 {
                    return unsafe {
                        self.compare_avx512_8(distances, squared_radius_half, results)
                    };
                }
                if W == 16 {
                    return unsafe {
                        self.compare_avx512_16(distances, squared_radius_half, results)
                    };
                }
            }
        }
        #[cfg(feature = "simd-compress")]
        {
            if W == 8 {
                return self.compare_simd_8(distances, squared_radius_half, results);
            }
            if W == 16 {
                return self.compare_simd_16(distances, squared_radius_half, results);
            }
        }
        self.compare_scalar(distances, squared_radius_half, results)
    }

    #[inline(never)]
    fn compare_scalar(
        &self,
        distances: [f32; W],
        squared_radius_half: f32,
        results: &mut [MaybeUninit<usize>; W],
    ) -> usize {
        let mut sum = 0;
        for i in 0..W {
            unsafe { results[sum].as_mut_ptr().write(self.ids[i] as usize) };
            let cond = distances[i] <= squared_radius_half;
            sum += cond as usize;
        }
        sum
    }

    // --- AVX-512 intrinsics (fastest path) ---

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    #[inline(never)]
    unsafe fn compare_avx512_8(
        &self,
        distances: [f32; W],
        squared_radius_half: f32,
        results: &mut [MaybeUninit<usize>; W],
    ) -> usize {
        use std::arch::x86_64::*;
        unsafe {
            let dist = _mm256_loadu_ps(distances.as_ptr());
            let threshold = _mm256_set1_ps(squared_radius_half);
            let mask = _mm256_cmp_ps_mask::<_CMP_LE_OS>(dist, threshold);

            let ids = _mm256_loadu_epi32(self.ids.as_ptr() as *const i32);
            let compressed = _mm256_maskz_compress_epi32(mask, ids);

            let lo128 = _mm256_castsi256_si128(compressed);
            let hi128 = _mm256_extracti128_si256::<1>(compressed);
            let widened_lo = _mm256_cvtepu32_epi64(lo128);
            let widened_hi = _mm256_cvtepu32_epi64(hi128);
            _mm256_storeu_epi64(results.as_mut_ptr() as *mut i64, widened_lo);
            _mm256_storeu_epi64((results.as_mut_ptr() as *mut i64).add(4), widened_hi);

            mask.count_ones() as usize
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    #[inline(never)]
    unsafe fn compare_avx512_16(
        &self,
        distances: [f32; W],
        squared_radius_half: f32,
        results: &mut [MaybeUninit<usize>; W],
    ) -> usize {
        use std::arch::x86_64::*;
        unsafe {
            let dist = _mm512_loadu_ps(distances.as_ptr());
            let threshold = _mm512_set1_ps(squared_radius_half);
            let mask = _mm512_cmple_ps_mask(dist, threshold);

            let ids = _mm512_loadu_epi32(self.ids.as_ptr() as *const i32);
            let compressed = _mm512_maskz_compress_epi32(mask, ids);

            // Widen lower 8 u32s → u64s and store
            let lo256 = _mm512_castsi512_si256(compressed);
            let widened_lo =
                _mm256_cvtepu32_epi64(_mm_loadu_epi32(&lo256 as *const __m256i as *const i32));
            let widened_mid = _mm256_cvtepu32_epi64(_mm_loadu_epi32(
                (&lo256 as *const __m256i as *const i32).add(4),
            ));
            _mm256_storeu_epi64(results.as_mut_ptr() as *mut i64, widened_lo);
            _mm256_storeu_epi64((results.as_mut_ptr() as *mut i64).add(4), widened_mid);

            // Widen upper 8 u32s → u64s and store
            let hi256 = _mm512_extracti64x4_epi64::<1>(compressed);
            let widened_hi_lo =
                _mm256_cvtepu32_epi64(_mm_loadu_epi32(&hi256 as *const __m256i as *const i32));
            let widened_hi_hi = _mm256_cvtepu32_epi64(_mm_loadu_epi32(
                (&hi256 as *const __m256i as *const i32).add(4),
            ));
            _mm256_storeu_epi64((results.as_mut_ptr() as *mut i64).add(8), widened_hi_lo);
            _mm256_storeu_epi64((results.as_mut_ptr() as *mut i64).add(12), widened_hi_hi);

            mask.count_ones() as usize
        }
    }

    // --- simd-lookup fallback (cross-platform, no AVX-512 required) ---

    #[cfg(feature = "simd-compress")]
    #[inline(never)]
    fn compare_simd_8(
        &self,
        distances: [f32; W],
        squared_radius_half: f32,
        results: &mut [MaybeUninit<usize>; W],
    ) -> usize {
        use simd_lookup::simd_compress::compress_u32x8;
        use simd_lookup::wide_utils::WideUtilsExt;
        use wide::{f32x8, u32x4, u32x8};

        let dist = f32x8::new(unsafe { *(distances.as_ptr() as *const [f32; 8]) });
        let threshold = f32x8::splat(squared_radius_half);
        let mask = dist.simd_le(threshold).to_bitmask() as u8;

        let ids = u32x8::from(unsafe { *(self.ids.as_ptr() as *const [u32; 8]) });
        let (compressed, count) = compress_u32x8(ids, mask);

        let arr = compressed.to_array();
        let lo = u32x4::from([arr[0], arr[1], arr[2], arr[3]]);
        let hi = u32x4::from([arr[4], arr[5], arr[6], arr[7]]);
        let widened_lo = lo.widen_to_u64x8();
        let widened_hi = hi.widen_to_u64x8();

        unsafe {
            let ptr = results.as_mut_ptr() as *mut wide::u64x4;
            ptr.write_unaligned(widened_lo);
            ptr.add(1).write_unaligned(widened_hi);
        }
        count
    }

    #[cfg(feature = "simd-compress")]
    #[inline(never)]
    fn compare_simd_16(
        &self,
        distances: [f32; W],
        squared_radius_half: f32,
        results: &mut [MaybeUninit<usize>; W],
    ) -> usize {
        use simd_lookup::simd_compress::compress_u32x8;
        use simd_lookup::wide_utils::{SimdSplit, WideUtilsExt};
        use wide::{f32x8, u32x4, u32x16};

        let dist_lo = f32x8::new(unsafe { *(distances.as_ptr() as *const [f32; 8]) });
        let dist_hi = f32x8::new(unsafe { *(distances.as_ptr().add(8) as *const [f32; 8]) });
        let threshold = f32x8::splat(squared_radius_half);
        let mask_lo = dist_lo.simd_le(threshold).to_bitmask() as u8;
        let mask_hi = dist_hi.simd_le(threshold).to_bitmask() as u8;

        let ids = u32x16::from(unsafe { *(self.ids.as_ptr() as *const [u32; 16]) });
        let (ids_lo, ids_hi) = ids.split_low_high();

        let (comp_lo, count_lo) = compress_u32x8(ids_lo, mask_lo);
        let arr_lo = comp_lo.to_array();
        let lo_lo = u32x4::from([arr_lo[0], arr_lo[1], arr_lo[2], arr_lo[3]]);
        let lo_hi = u32x4::from([arr_lo[4], arr_lo[5], arr_lo[6], arr_lo[7]]);
        unsafe {
            let ptr = results.as_mut_ptr() as *mut wide::u64x4;
            ptr.write_unaligned(lo_lo.widen_to_u64x8());
            ptr.add(1).write_unaligned(lo_hi.widen_to_u64x8());
        }

        let (comp_hi, count_hi) = compress_u32x8(ids_hi, mask_hi);
        let arr_hi = comp_hi.to_array();
        let hi_lo = u32x4::from([arr_hi[0], arr_hi[1], arr_hi[2], arr_hi[3]]);
        let hi_hi = u32x4::from([arr_hi[4], arr_hi[5], arr_hi[6], arr_hi[7]]);
        unsafe {
            let ptr = results.as_mut_ptr().add(count_lo) as *mut wide::u64x4;
            ptr.write_unaligned(hi_lo.widen_to_u64x8());
            ptr.add(1).write_unaligned(hi_hi.widen_to_u64x8());
        }

        count_lo + count_hi
    }
}
