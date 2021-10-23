//! Prefix Sum Index
//!
//! Scans through the slice, calculating the prefix sum, returning it along the
//! index of the first element that has a value **greater than** the provided
//! `lookup`.
//!
//! Returns [`Ok((index, prefix_sum))`] on success and [`Err(prefix_sum)`] if
//! `lookup` lies outside of the bounds of the given `offsets`, returning `0`
//! if the given `offsets` are empty.
//!
//! # Examples
//!
//! ```
//! use psy::prefix_sum_index;
//!
//! let offsets = [
//!     0, //  0
//!     1, //  1
//!     0, //  1
//!     4, //  5
//!     8, // 13
//!     1, // 14
//!     2, // 16
//!     9, // 25
//!     8, // 33
//!     1, // 34
//! ];
//!
//! assert_eq!(prefix_sum_index(&[], 0), Err(0));
//! assert_eq!(prefix_sum_index(&offsets, 0), Ok((1, 1)));
//! assert_eq!(prefix_sum_index(&offsets, 1), Ok((3, 5)));
//! assert_eq!(prefix_sum_index(&offsets, 21), Ok((7, 25)));
//! assert_eq!(prefix_sum_index(&offsets, 78), Err(34));
//! ```

/// Calculate the Prefix Sum Index
///
/// See [`crate level docs`](crate) for more information.
pub fn prefix_sum_index(offsets: &[u8], lookup: usize) -> Result<(usize, usize), usize> {
    let mut start = 0;
    let mut index = 0;

    //let mut chunks = offsets.array_chunks();
    let mut chunks = offsets.chunks_exact(8);
    for chunk in &mut chunks {
        // SAFETY: `chunks_exact` guarantees this is a `&[u8; 8]`
        // we can avoid this in the future once `array_chunks` is stable.
        let chunk = unsafe { &*(chunk as *const [u8] as *const [u8; 8]) };
        match unsafe { prefix_sum_8(chunk, lookup - start) } {
            Ok((idx, sum)) => return Ok((index + idx, start + sum)),
            Err(sum) => start += sum,
        }
        index += 8;
    }
    match prefix_sum_fallback(chunks.remainder(), lookup - start) {
        Ok((idx, sum)) => Ok((index + idx, start + sum)),
        Err(sum) => Err(start + sum),
    }
}

fn prefix_sum_fallback(offsets: &[u8], lookup: usize) -> Result<(usize, usize), usize> {
    let mut start = 0;
    for (i, offset) in offsets.iter().enumerate() {
        let current = start + *offset as usize;
        if current > lookup {
            return Ok((i, current));
        }
        start = current;
    }
    Err(start)
}

//#[cfg(all(target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn prefix_sum_8(offsets: &[u8; 8], lookup: usize) -> Result<(usize, usize), usize> {
    // SAFETY:
    // - we have a 16-byte stack allocation that we don’t index out of bounds.
    // - the prefix sum itself is bounded to `u8::MAX * 8`, which is `< i16::MAX`.
    // - we check that lookup is `< i16::MAX` to avoid overflow.
    unsafe {
        use core::arch::x86_64::*;
        // copy the 8 bytes into the first 8 of 16 bytes
        let mut buf = [u64::from_le_bytes(*offsets), 0];
        let buf_ptr = &mut buf as *mut u64 as *mut __m128i;
        // load the 16 bytes into a m128
        let mut mm = _mm_lddqu_si128(buf_ptr);
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

        _mm_storeu_si128(buf_ptr, mm);
        let u16_buf = &mut *(buf_ptr as *mut [u16; 8]);

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
}

#[test]
fn test_fallback() {
    let offsets = [
        0, // 0
        1, // 1
        4, // 5
        8, // 13
    ];
    assert_eq!(prefix_sum_fallback(&offsets, 0,), Ok((1, 1)));
    assert_eq!(prefix_sum_fallback(&offsets, 1,), Ok((2, 5)));
    assert_eq!(prefix_sum_fallback(&offsets, 7,), Ok((3, 13)));
    assert_eq!(prefix_sum_fallback(&offsets, 16), Err(13));
}

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
        unsafe { prefix_sum_8(&offsets, 0) },
        prefix_sum_fallback(&offsets, 0)
    );
    assert_eq!(
        unsafe { prefix_sum_8(&offsets, 1) },
        prefix_sum_fallback(&offsets, 1)
    );
    assert_eq!(
        unsafe { prefix_sum_8(&offsets, 7) },
        prefix_sum_fallback(&offsets, 7)
    );
    assert_eq!(
        unsafe { prefix_sum_8(&offsets, 16) },
        prefix_sum_fallback(&offsets, 16)
    );
    assert_eq!(
        unsafe { prefix_sum_8(&offsets, 25) },
        prefix_sum_fallback(&offsets, 25)
    );

    let offsets = [255; 8];
    assert_eq!(unsafe { prefix_sum_8(&offsets, 1 << 34) }, Err(255 * 8));
}

#[test]
fn test_combined() {
    assert_eq!(prefix_sum_index(&[], 0), Err(0));
    assert_eq!(prefix_sum_index(&[0], 0), Err(0));
    assert_eq!(prefix_sum_index(&[0], 12345), Err(0));
    assert_eq!(prefix_sum_index(&[1], 0), Ok((0, 1)));
    assert_eq!(prefix_sum_index(&[1], 1), Err(1));
    assert_eq!(prefix_sum_index(&[0, 1], 1), Err(1));
    assert_eq!(prefix_sum_index(&[0, 2], 1), Ok((1, 2)));

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
    ];
    assert_eq!(
        prefix_sum_index(&offsets, 0),
        prefix_sum_fallback(&offsets, 0)
    );
    assert_eq!(
        prefix_sum_index(&offsets, 1),
        prefix_sum_fallback(&offsets, 1)
    );
    assert_eq!(
        prefix_sum_index(&offsets, 7),
        prefix_sum_fallback(&offsets, 7)
    );
    assert_eq!(
        prefix_sum_index(&offsets, 16),
        prefix_sum_fallback(&offsets, 16)
    );
    assert_eq!(
        prefix_sum_index(&offsets, 25),
        prefix_sum_fallback(&offsets, 25)
    );
    assert_eq!(
        prefix_sum_index(&offsets, 34),
        prefix_sum_fallback(&offsets, 34)
    );
    assert_eq!(
        prefix_sum_index(&offsets, 35),
        prefix_sum_fallback(&offsets, 35)
    );
}
