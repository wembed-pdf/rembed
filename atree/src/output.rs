use std::mem::{MaybeUninit, offset_of, size_of};

/// Controls what each SIMD compare call produces.
///
/// `from_match` converts a single `(id, distance)` pair. `store_compressed`
/// batch-stores compressed results, defaulting to a `from_match` loop but
/// overridable for SIMD-optimized widening and interleaving.
pub trait QueryOutput: Copy + Sized {
    fn from_match(id: u32, distance: f32) -> Self;

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u32; W],
        dists: &[f32; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        for i in 0..count {
            dst[i].write(Self::from_match(ids[i], dists[i]));
        }
        count
    }
}

// ── u32: direct SIMD copy ────────────────────────────────────────────

impl QueryOutput for u32 {
    #[inline(always)]
    fn from_match(id: u32, _distance: f32) -> Self {
        id
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u32; W],
        _dists: &[f32; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        // MaybeUninit<u32> has identical layout to u32; memcpy compiles to SIMD moves
        unsafe {
            std::ptr::copy_nonoverlapping(ids.as_ptr(), dst.as_mut_ptr() as *mut u32, W);
        }
        count
    }
}

// ── u64: SIMD-optimized widening ─────────────────────────────────────

impl QueryOutput for u64 {
    #[inline(always)]
    fn from_match(id: u32, _distance: f32) -> Self {
        id as u64
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u32; W],
        _dists: &[f32; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") && W >= 8 {
            unsafe { widen_store_avx512::<W>(ids.as_ptr(), dst.as_mut_ptr() as *mut i64) };
            return count;
        }
        #[cfg(feature = "simd-compress")]
        if W >= 8 {
            unsafe {
                widen_store_wide::<W>(ids.as_ptr(), dst.as_mut_ptr() as *mut MaybeUninit<u64>)
            };
            return count;
        }
        for i in 0..count {
            dst[i].write(ids[i] as u64);
        }
        count
    }
}

// ── usize: delegates to u32/u64 by pointer width ────────────────────

#[cfg(target_pointer_width = "64")]
impl QueryOutput for usize {
    #[inline(always)]
    fn from_match(id: u32, distance: f32) -> Self {
        u64::from_match(id, distance) as usize
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u32; W],
        dists: &[f32; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        // SAFETY: MaybeUninit<usize> and MaybeUninit<u64> have identical layout on 64-bit
        let dst_u64 =
            unsafe { &mut *(dst as *mut [MaybeUninit<usize>; W] as *mut [MaybeUninit<u64>; W]) };
        u64::store_compressed(count, ids, dists, dst_u64)
    }
}

#[cfg(target_pointer_width = "32")]
impl QueryOutput for usize {
    #[inline(always)]
    fn from_match(id: u32, distance: f32) -> Self {
        u32::from_match(id, distance) as usize
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u32; W],
        dists: &[f32; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        // SAFETY: MaybeUninit<usize> and MaybeUninit<u32> have identical layout on 32-bit
        let dst_u32 =
            unsafe { &mut *(dst as *mut [MaybeUninit<usize>; W] as *mut [MaybeUninit<u32>; W]) };
        u32::store_compressed(count, ids, dists, dst_u32)
    }
}

// ── (u32, f32): SIMD interleaved store ───────────────────────────────

// Layout assertions: SIMD interleaving assumes [u32, f32] contiguous at stride 8
const _: () = assert!(size_of::<(u32, f32)>() == 8);
const _: () = assert!(offset_of!((u32, f32), 0) == 0);
const _: () = assert!(offset_of!((u32, f32), 1) == 4);

impl QueryOutput for (u32, f32) {
    #[inline(always)]
    fn from_match(id: u32, distance: f32) -> Self {
        (id, distance)
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u32; W],
        dists: &[f32; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") && W >= 8 {
            unsafe {
                interleave_store_u32_f32_avx512::<W>(
                    ids.as_ptr(),
                    dists.as_ptr(),
                    dst.as_mut_ptr() as *mut f32,
                )
            };
            return count;
        }
        for i in 0..count {
            dst[i].write((ids[i], dists[i]));
        }
        count
    }
}

// ── (u64, f32): SIMD widen + interleave ──────────────────────────────

const _: () = assert!(size_of::<(u64, f32)>() == 16);
const _: () = assert!(offset_of!((u64, f32), 0) == 0);
const _: () = assert!(offset_of!((u64, f32), 1) == 8);

impl QueryOutput for (u64, f32) {
    #[inline(always)]
    fn from_match(id: u32, distance: f32) -> Self {
        (id as u64, distance)
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u32; W],
        dists: &[f32; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") && W >= 8 {
            unsafe {
                interleave_store_u64_f32_avx512::<W>(
                    ids.as_ptr(),
                    dists.as_ptr(),
                    dst.as_mut_ptr() as *mut i64,
                )
            };
            return count;
        }
        for i in 0..count {
            dst[i].write((ids[i] as u64, dists[i]));
        }
        count
    }
}

// ── (usize, f32): delegates to (u64, f32) / (u32, f32) ──────────────

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(size_of::<(usize, f32)>() == 16);
    assert!(offset_of!((usize, f32), 0) == 0);
    assert!(offset_of!((usize, f32), 1) == 8);
};

impl QueryOutput for (usize, f32) {
    #[inline(always)]
    fn from_match(id: u32, distance: f32) -> Self {
        (id as usize, distance)
    }

    #[inline(always)]
    fn store_compressed<const W: usize>(
        count: usize,
        ids: &[u32; W],
        dists: &[f32; W],
        dst: &mut [MaybeUninit<Self>; W],
    ) -> usize {
        // (usize, f32) has same layout as (u64, f32) on 64-bit / (u32, f32) on 32-bit.
        // The SIMD helpers operate on raw pointers and produce the correct byte pattern
        // for the verified layout above.
        #[cfg(target_pointer_width = "64")]
        {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") && W >= 8 {
                unsafe {
                    interleave_store_u64_f32_avx512::<W>(
                        ids.as_ptr(),
                        dists.as_ptr(),
                        dst.as_mut_ptr() as *mut i64,
                    )
                };
                return count;
            }
        }
        #[cfg(target_pointer_width = "32")]
        {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") && W >= 8 {
                unsafe {
                    interleave_store_u32_f32_avx512::<W>(
                        ids.as_ptr(),
                        dists.as_ptr(),
                        dst.as_mut_ptr() as *mut f32,
                    )
                };
                return count;
            }
        }
        for i in 0..count {
            dst[i].write((ids[i] as usize, dists[i]));
        }
        count
    }
}

// ── SIMD helpers ─────────────────────────────────────────────────────

/// Widen u32 IDs → u64 via vpmovzxdq + store.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn widen_store_avx512<const W: usize>(ids: *const u32, dst: *mut i64) {
    use std::arch::x86_64::*;
    unsafe {
        let v = _mm256_loadu_epi32(ids as *const i32);
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        _mm256_storeu_epi64(dst, _mm256_cvtepu32_epi64(lo));
        _mm256_storeu_epi64(dst.add(4), _mm256_cvtepu32_epi64(hi));

        if W >= 16 {
            let v = _mm256_loadu_epi32(ids.add(8) as *const i32);
            let lo = _mm256_castsi256_si128(v);
            let hi = _mm256_extracti128_si256::<1>(v);
            _mm256_storeu_epi64(dst.add(8), _mm256_cvtepu32_epi64(lo));
            _mm256_storeu_epi64(dst.add(12), _mm256_cvtepu32_epi64(hi));
        }
    }
}

/// Widen u32 IDs → u64 via `wide` crate.
#[cfg(feature = "simd-compress")]
unsafe fn widen_store_wide<const W: usize>(ids: *const u32, dst: *mut MaybeUninit<u64>) {
    use simd_lookup::wide_utils::WideUtilsExt;
    use wide::u32x4;
    unsafe {
        let lo = u32x4::from(*(ids as *const [u32; 4]));
        let hi = u32x4::from(*(ids.add(4) as *const [u32; 4]));
        let ptr = dst as *mut wide::u64x4;
        ptr.write_unaligned(lo.widen_to_u64x8());
        ptr.add(1).write_unaligned(hi.widen_to_u64x8());

        if W >= 16 {
            let lo = u32x4::from(*(ids.add(8) as *const [u32; 4]));
            let hi = u32x4::from(*(ids.add(12) as *const [u32; 4]));
            let ptr = (dst as *mut wide::u64x4).add(2);
            ptr.write_unaligned(lo.widen_to_u64x8());
            ptr.add(1).write_unaligned(hi.widen_to_u64x8());
        }
    }
}

/// Interleave u32 IDs + f32 distances → (u32, f32) pairs via AVX unpack + lane permute.
/// Output: [id0, d0, id1, d1, ...] as contiguous 8-byte pairs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn interleave_store_u32_f32_avx512<const W: usize>(
    ids: *const u32,
    dists: *const f32,
    dst: *mut f32,
) {
    use std::arch::x86_64::*;
    unsafe {
        let ids_v = _mm256_castsi256_ps(_mm256_loadu_epi32(ids as *const i32));
        let dists_v = _mm256_loadu_ps(dists);
        // unpack interleaves within 128-bit lanes:
        // lo = [i0,d0,i1,d1 | i4,d4,i5,d5]
        // hi = [i2,d2,i3,d3 | i6,d6,i7,d7]
        let lo = _mm256_unpacklo_ps(ids_v, dists_v);
        let hi = _mm256_unpackhi_ps(ids_v, dists_v);
        // permute crosses lanes to get sequential order
        let r0 = _mm256_permute2f128_ps(lo, hi, 0x20); // [i0,d0,i1,d1,i2,d2,i3,d3]
        let r1 = _mm256_permute2f128_ps(lo, hi, 0x31); // [i4,d4,i5,d5,i6,d6,i7,d7]
        _mm256_storeu_ps(dst, r0);
        _mm256_storeu_ps(dst.add(8), r1);

        if W >= 16 {
            let ids_v = _mm256_castsi256_ps(_mm256_loadu_epi32(ids.add(8) as *const i32));
            let dists_v = _mm256_loadu_ps(dists.add(8));
            let lo = _mm256_unpacklo_ps(ids_v, dists_v);
            let hi = _mm256_unpackhi_ps(ids_v, dists_v);
            let r0 = _mm256_permute2f128_ps(lo, hi, 0x20);
            let r1 = _mm256_permute2f128_ps(lo, hi, 0x31);
            _mm256_storeu_ps(dst.add(16), r0); // offset 16 f32s = 64 bytes = 8 pairs
            _mm256_storeu_ps(dst.add(24), r1);
        }
    }
}

/// Interleave widened u64 IDs + f32 distances → (u64, f32) pairs via AVX-512 permute.
/// Each output element is 16 bytes: [u64_id, f32_dist, u32_padding].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn interleave_store_u64_f32_avx512<const W: usize>(
    ids: *const u32,
    dists: *const f32,
    dst: *mut i64,
) {
    use std::arch::x86_64::*;
    unsafe {
        // Widen 8 u32 IDs → 8 u64 in zmm
        let ids_wide = _mm512_cvtepu32_epi64(_mm256_loadu_epi32(ids as *const i32));
        // Zero-extend f32 bits to 64-bit (preserves f32 value at correct struct offset)
        let dists_wide =
            _mm512_cvtepu32_epi64(_mm256_castps_si256(_mm256_loadu_ps(dists)));

        // Interleave: [I0,D0, I1,D1, ..., I7,D7] via cross-source permute
        let idx_lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
        let idx_hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
        let r0 = _mm512_permutex2var_epi64(ids_wide, idx_lo, dists_wide); // elements 0-3
        let r1 = _mm512_permutex2var_epi64(ids_wide, idx_hi, dists_wide); // elements 4-7
        _mm512_storeu_epi64(dst, r0); // 64 bytes = 4 elements
        _mm512_storeu_epi64(dst.add(8), r1); // next 64 bytes

        if W >= 16 {
            let ids_wide = _mm512_cvtepu32_epi64(_mm256_loadu_epi32(ids.add(8) as *const i32));
            let dists_wide =
                _mm512_cvtepu32_epi64(_mm256_castps_si256(_mm256_loadu_ps(dists.add(8))));
            let r0 = _mm512_permutex2var_epi64(ids_wide, idx_lo, dists_wide);
            let r1 = _mm512_permutex2var_epi64(ids_wide, idx_hi, dists_wide);
            _mm512_storeu_epi64(dst.add(16), r0); // 8 elements * 2 i64s each = offset 16
            _mm512_storeu_epi64(dst.add(24), r1);
        }
    }
}
