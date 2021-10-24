pub fn prefix_sum_fallback(offsets: &[u8], lookup: usize) -> Result<(usize, usize), usize> {
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
