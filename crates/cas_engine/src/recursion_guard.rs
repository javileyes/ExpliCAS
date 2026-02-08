#![allow(dead_code)] // Debugging infrastructure — kept for future diagnostic use
//! Recursion depth guards for debugging stack overflows.
//!
//! This module provides thread-local recursion depth tracking to convert
//! hard-to-debug stack overflows into panics with proper backtraces.
//!
//! # Usage
//!
//! ```ignore
//! use crate::recursion_guard::with_depth_guard;
//!
//! fn recursive_function(expr: ExprId) {
//!     with_depth_guard("recursive_function", 100, || {
//!         // ... recursive body ...
//!     })
//! }
//! ```

use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// Per-function depth counters
    static DEPTHS: RefCell<HashMap<&'static str, usize>> = RefCell::new(HashMap::new());

    /// Maximum depth seen per function (for diagnostics)
    static MAX_DEPTHS: RefCell<HashMap<&'static str, usize>> = RefCell::new(HashMap::new());
}

/// Execute a closure with recursion depth tracking.
/// Panics if depth exceeds the limit, providing a useful backtrace.
///
/// # Intentional Panic
/// This panic is by design - it converts hard-to-debug stack overflows
/// into panics with proper backtraces. The alternative (returning Result)
/// would require threading errors through the entire call stack.
#[inline]
#[allow(clippy::panic)] // Intentional: stack overflow protection
pub fn with_depth_guard<F, R>(label: &'static str, limit: usize, f: F) -> R
where
    F: FnOnce() -> R,
{
    DEPTHS.with(|depths| {
        let mut depths = depths.borrow_mut();
        let depth = depths.entry(label).or_insert(0);
        *depth += 1;
        let current = *depth;

        // Track max depth seen
        MAX_DEPTHS.with(|max_depths| {
            let mut max_depths = max_depths.borrow_mut();
            let max = max_depths.entry(label).or_insert(0);
            if current > *max {
                *max = current;
            }
        });

        if current > limit {
            panic!(
                "RECURSION GUARD: {} exceeded depth limit: {} > {}",
                label, current, limit
            );
        }
        drop(depths);

        let result = f();

        DEPTHS.with(|depths| {
            let mut depths = depths.borrow_mut();
            if let Some(depth) = depths.get_mut(label) {
                *depth -= 1;
            }
        });

        result
    })
}

/// Reset all depth counters. Call at test start.
pub fn reset_all_guards() {
    DEPTHS.with(|d| d.borrow_mut().clear());
    MAX_DEPTHS.with(|d| d.borrow_mut().clear());
}

/// Get the maximum depth seen for a function.
pub fn get_max_depth(label: &'static str) -> usize {
    MAX_DEPTHS.with(|max_depths| *max_depths.borrow().get(label).unwrap_or(&0))
}

/// Get all maximum depths (for diagnostics).
pub fn get_all_max_depths() -> Vec<(&'static str, usize)> {
    MAX_DEPTHS.with(|max_depths| max_depths.borrow().iter().map(|(&k, &v)| (k, v)).collect())
}

/// Execute a closure in a thread with a larger stack size.
/// Useful for running deep simplifications that would overflow the default 8MB stack.
///
/// # Example
/// ```ignore
/// let result = with_stack(16 * 1024 * 1024, || engine.simplify(expr));
/// ```
pub fn with_stack<R: Send + 'static>(
    stack_size: usize,
    f: impl FnOnce() -> R + Send + 'static,
) -> R {
    std::thread::Builder::new()
        .stack_size(stack_size)
        .spawn(f)
        .expect("Failed to spawn thread with custom stack")
        .join()
        .expect("Thread panicked")
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
