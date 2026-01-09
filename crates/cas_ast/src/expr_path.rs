//! Expression path types for occurrence-based highlighting.
//!
//! An `ExprPath` identifies a specific location in an expression tree,
//! allowing highlighting of a single occurrence even when the same
//! value appears multiple times (DAG/hashconsing).

/// A path from the root to a specific node in an expression tree.
///
/// Each element is a child index (0-based):
/// - Binary ops (Add, Sub, Mul, Div, Pow): 0=left/base/num, 1=right/exp/den
/// - Unary ops (Neg): 0=inner
/// - Function: 0..n = arguments
///
/// Example: In `(a + b) * c`, the path to `b` is `[0, 1]`
/// (first child of root is Add, second child of Add is b)
pub type ExprPath = Vec<u8>;

/// Child index constants for clarity
pub mod child {
    /// Left child of binary op, base of Pow, numerator of Div
    pub const LEFT: u8 = 0;
    /// Right child of binary op, exponent of Pow, denominator of Div
    pub const RIGHT: u8 = 1;
    /// Inner expression of Neg
    pub const INNER: u8 = 0;
}

/// Convert a path to a human-readable string (for debugging)
pub fn path_to_string(path: &ExprPath) -> String {
    if path.is_empty() {
        return "root".to_string();
    }
    path.iter()
        .map(|&i| i.to_string())
        .collect::<Vec<_>>()
        .join(".")
}

/// Check if `prefix` is a prefix of `path` (or equal)
pub fn is_prefix_of(prefix: &ExprPath, path: &ExprPath) -> bool {
    if prefix.len() > path.len() {
        return false;
    }
    prefix.iter().zip(path.iter()).all(|(a, b)| a == b)
}

/// Check if two paths are exactly equal
pub fn paths_equal(a: &ExprPath, b: &ExprPath) -> bool {
    a == b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_to_string() {
        let empty: ExprPath = vec![];
        assert_eq!(path_to_string(&empty), "root");

        let path: ExprPath = vec![0, 1, 2];
        assert_eq!(path_to_string(&path), "0.1.2");
    }

    #[test]
    fn test_is_prefix() {
        let path: ExprPath = vec![0, 1, 2];
        let prefix: ExprPath = vec![0, 1];
        let not_prefix: ExprPath = vec![0, 2];

        assert!(is_prefix_of(&prefix, &path));
        assert!(is_prefix_of(&path, &path)); // Equal is prefix
        assert!(!is_prefix_of(&not_prefix, &path));
        assert!(!is_prefix_of(&path, &prefix)); // Longer can't be prefix
    }
}
