use crate::recursion_guard::{get_max_depth, reset_all_guards, with_depth_guard};

#[test]
fn test_normal_recursion() {
    reset_all_guards();

    fn recurse(n: usize) -> usize {
        with_depth_guard("test_recurse", 10, || {
            if n == 0 {
                0
            } else {
                n + recurse(n - 1)
            }
        })
    }

    assert_eq!(recurse(5), 15);
    assert_eq!(get_max_depth("test_recurse"), 6);
}

#[test]
#[should_panic(expected = "RECURSION GUARD")]
fn test_exceeds_limit() {
    reset_all_guards();

    fn deep_recurse() {
        with_depth_guard("deep_recurse", 5, || {
            deep_recurse();
        })
    }

    deep_recurse();
}
