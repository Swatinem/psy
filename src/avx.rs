/// Calculate the Prefix Sum Index using AVX intrinsics.
///
/// See [`crate level docs`](crate) for more information.
#[target_feature(enable = "avx")]
pub unsafe fn prefix_sum_8(offsets: &[u8], lookup: usize) -> Result<(usize, usize), usize> {
    let mut start = 0;
    let mut index = 0;

    //let mut chunks = offsets.array_chunks();
    let mut chunks = offsets.chunks_exact(8);
    for chunk in &mut chunks {
        // SAFETY: `chunks_exact` guarantees this is a `&[u8; 8]`
        // we can avoid this in the future once `array_chunks` is stable.
        let chunk = &*(chunk as *const [u8] as *const [u8; 8]);
        match prefix_sum_8_inner(chunk, lookup - start) {
            Ok((idx, sum)) => return Ok((index + idx, start + sum)),
            Err(sum) => start += sum,
        }
        index += 8;
    }
    let remainder = chunks.remainder();
    let mut buf = [0u8; 8];
    {
        let (prefix, _) = buf.split_at_mut(remainder.len());
        prefix.copy_from_slice(remainder);
    }
    match prefix_sum_8_inner(&buf, lookup - start) {
        Ok((idx, sum)) => Ok((index + idx, start + sum)),
        Err(sum) => Err(start + sum),
    }
}

#[target_feature(enable = "avx")]
unsafe fn prefix_sum_8_inner(offsets: &[u8; 8], lookup: usize) -> Result<(usize, usize), usize> {
    // SAFETY:
    // - we have a 16-byte stack allocation that we don’t index out of bounds.
    // - the prefix sum itself is bounded to `u8::MAX * 8`, which is `< i16::MAX`.
    // - we check that lookup is `< i16::MAX` to avoid overflow.
    use core::arch::x86_64::*;
    // copy the 8 bytes into the first 8 of 16 bytes
    let mut mm_buf: __m128i = core::mem::zeroed();
    *(&mut mm_buf as *mut __m128i as *mut [u8; 8]) = *offsets;
    // load the 16 bytes into a m128
    let mut mm = _mm_load_si128(&mm_buf as *const __m128i);
    // spread the 8xu8 in the first 64bit out to 8xi16
    mm = _mm_cvtepu8_epi16(mm);

    // do the prefix sum, simplified like this, except we have 8 values:
    //   [a,     b,         c,             d]
    // + [0,     0, a        ,     b        ]
    // = [a, b    , a + c    ,     b     + d]
    // + [0, a    , b        , a     + c    ]
    // = [a, a + b, a + b + c, a + b + c + d]
    mm = _mm_add_epi16(mm, _mm_slli_si128::<8>(mm));
    mm = _mm_add_epi16(mm, _mm_slli_si128::<4>(mm));
    mm = _mm_add_epi16(mm, _mm_slli_si128::<2>(mm));

    _mm_store_si128(&mut mm_buf, mm);
    let u16_buf = &*(&mm_buf as *const __m128i as *const [u16; 8]);

    if lookup > i16::MAX as usize {
        return Err(u16_buf[7] as usize);
    }

    // compare each i16 with our lookup
    let lookup = _mm_set1_epi16(lookup as i16);
    let cmp = _mm_cmpgt_epi16(mm, lookup);

    // compress the 8*i16 into one i32
    let mask = _mm_movemask_epi8(cmp);
    // get the number of *trailing* zeros
    // trailing, because we are dealing with little-endian bytes here
    let idx = mask.trailing_zeros() as usize / 2;
    if idx > 7 {
        Err(u16_buf[7] as usize)
    } else {
        Ok((idx, u16_buf[idx] as usize))
    }
}

#[cfg(test)]
use crate::prefix_sum_fallback;

#[test]
fn test_simd_8() {
    let offsets = [
        0, //  0
        1, //  1
        0, //  1
        4, //  5
        8, // 13
        1, // 14
        2, // 16
        9, // 25
    ];
    assert_eq!(
        unsafe { prefix_sum_8_inner(&offsets, 0) },
        prefix_sum_fallback(&offsets, 0)
    );
    assert_eq!(
        unsafe { prefix_sum_8_inner(&offsets, 1) },
        prefix_sum_fallback(&offsets, 1)
    );
    assert_eq!(
        unsafe { prefix_sum_8_inner(&offsets, 7) },
        prefix_sum_fallback(&offsets, 7)
    );
    assert_eq!(
        unsafe { prefix_sum_8_inner(&offsets, 16) },
        prefix_sum_fallback(&offsets, 16)
    );
    assert_eq!(
        unsafe { prefix_sum_8_inner(&offsets, 25) },
        prefix_sum_fallback(&offsets, 25)
    );

    let offsets = [255; 8];
    assert_eq!(
        unsafe { prefix_sum_8_inner(&offsets, 1 << 34) },
        Err(255 * 8)
    );
}
