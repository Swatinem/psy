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

mod avx;
mod avx2;
#[cfg(test)]
mod fallback;

#[cfg(test)]
use fallback::prefix_sum_fallback;

/// Calculate the Prefix Sum Index
///
/// See [`crate level docs`](crate) for more information.
pub fn prefix_sum_index(offsets: &[u8], lookup: usize) -> Result<(usize, usize), usize> {
    unsafe { avx2::prefix_sum_16(offsets, lookup) }
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
