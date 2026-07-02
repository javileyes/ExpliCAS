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

#[cfg(test)]
mod tests {}
