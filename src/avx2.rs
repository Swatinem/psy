/// Calculate the Prefix Sum Index using AVX2 intrinsics.
///
/// See [`crate level docs`](crate) for more information.
#[target_feature(enable = "avx2")]
pub unsafe fn prefix_sum_16(offsets: &[u8], lookup: usize) -> Result<(usize, usize), usize> {
    let mut start = 0;
    let mut index = 0;

    //let mut chunks = offsets.array_chunks();
    let mut chunks = offsets.chunks_exact(16);
    for chunk in &mut chunks {
        // SAFETY: `chunks_exact` guarantees this is a `&[u8; 16]`
        // we can avoid this in the future once `array_chunks` is stable.
        let chunk = &*(chunk as *const [u8] as *const [u8; 16]);
        match prefix_sum_16_inner(chunk, lookup - start) {
            Ok((idx, sum)) => return Ok((index + idx, start + sum)),
            Err(sum) => start += sum,
        }
        index += 8;
    }
    let remainder = chunks.remainder();
    let mut buf: [u8; 16] = core::mem::zeroed();
    {
        let (prefix, _) = buf.split_at_mut(remainder.len());
        prefix.copy_from_slice(remainder);
    }
    match prefix_sum_16_inner(&buf, lookup - start) {
        Ok((idx, sum)) => Ok((index + idx, start + sum)),
        Err(sum) => Err(start + sum),
    }
}

#[target_feature(enable = "avx2")]
unsafe fn prefix_sum_16_inner(offsets: &[u8; 16], lookup: usize) -> Result<(usize, usize), usize> {
    // SAFETY:
    // - we have a 32-byte stack allocation that we don’t index out of bounds.
    // - the prefix sum itself is bounded to `u8::MAX * 16`, which is `< i16::MAX`.
    // - we check that lookup is `< i16::MAX` to avoid overflow.
    use core::arch::x86_64::*;
    // copy the 16 bytes
    let mut mm_buf: __m256i = core::mem::zeroed();
    *(&mut mm_buf as *mut __m256i as *mut [u8; 16]) = *offsets;
    // load the 16 bytes into a m128
    let mm = _mm_load_si128(&mm_buf as *const __m256i as *const __m128i);
    // spread the 16xu8 in out to 16xi16
    let mut mm = _mm256_cvtepu8_epi16(mm);

    // do the prefix sum, simplified like this, except we have 8 values:
    //   [a,     b,         c,             d]
    // + [0,     0, a        ,     b        ]
    // = [a, b    , a + c    ,     b     + d]
    // + [0, a    , b        , a     + c    ]
    // = [a, a + b, a + b + c, a + b + c + d]
    mm = _mm256_add_epi16(mm, _mm256_slli_si256::<2>(mm));
    mm = _mm256_add_epi16(mm, _mm256_slli_si256::<4>(mm));
    mm = _mm256_add_epi16(mm, _mm256_slli_si256::<8>(mm));

    // we can’t shift by 16 bytes, as 256-bit simd works with 128-bit lanes :-(
    // but this is a prefix sum after all, so we can just add the last
    // 16-bit element of the first 128-lane to all the 16-bit elements of the second
    let hi = _mm_set1_epi16(_mm256_extract_epi16::<7>(mm) as i16);
    let shifted = _mm256_set_m128i(hi, _mm_setzero_si128());
    mm = _mm256_add_epi16(mm, shifted);

    _mm256_store_si256(&mut mm_buf, mm);
    let u16_buf = &*(&mm_buf as *const __m256i as *const [u16; 16]);

    if lookup > i16::MAX as usize {
        return Err(u16_buf[15] as usize);
    }

    // compare each i16 with our lookup
    let lookup = _mm256_set1_epi16(lookup as i16);

    let cmp = _mm256_cmpgt_epi16(mm, lookup);
    // compress the 16*i16 into one i32
    let mask = _mm256_movemask_epi8(cmp);

    // _mm256_cmpgt_epu16_mask
    // ^ only on `avx512bw,avx512vl`
    // _mm512_cmpgt_epi16_mask
    // ^ only on `avx512bw`

    // get the number of *trailing* zeros
    // trailing, because we are dealing with little-endian bytes here
    let idx = mask.trailing_zeros() as usize / 2;
    if idx > 15 {
        Err(u16_buf[15] as usize)
    } else {
        Ok((idx, u16_buf[idx] as usize))
    }
}

#[cfg(test)]
use crate::prefix_sum_fallback;

#[test]
fn test_simd_16() {
    let offsets = [
        0, //  0
        1, //  1
        0, //  1
        4, //  5
        8, // 13
        1, // 14
        2, // 16
        9, // 25
        8, // 33
        1, // 34
        4, // 38
        1, // 39
        3, // 42
        7, // 49
        1, // 50
        6, // 56
    ];
    assert_eq!(
        unsafe { prefix_sum_16_inner(&offsets, 0) },
        prefix_sum_fallback(&offsets, 0)
    );
    assert_eq!(
        unsafe { prefix_sum_16_inner(&offsets, 1) },
        prefix_sum_fallback(&offsets, 1)
    );
    assert_eq!(
        unsafe { prefix_sum_16_inner(&offsets, 7) },
        prefix_sum_fallback(&offsets, 7)
    );
    assert_eq!(
        unsafe { prefix_sum_16_inner(&offsets, 16) },
        prefix_sum_fallback(&offsets, 16)
    );
    assert_eq!(
        unsafe { prefix_sum_16_inner(&offsets, 25) },
        prefix_sum_fallback(&offsets, 25)
    );
    assert_eq!(
        unsafe { prefix_sum_16_inner(&offsets, 52) },
        prefix_sum_fallback(&offsets, 52)
    );
    assert_eq!(
        unsafe { prefix_sum_16_inner(&offsets, 60) },
        prefix_sum_fallback(&offsets, 60)
    );

    let offsets = [255; 16];
    assert_eq!(
        unsafe { prefix_sum_16_inner(&offsets, 1 << 34) },
        Err(255 * 16)
    );
}
