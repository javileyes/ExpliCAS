// =============================================================================
// Numerical Helpers (Zero-Clone for Number inspection)
// =============================================================================

use cas_ast::{Context, Expr, ExprId};
use num_traits::ToPrimitive;

/// Get a reference to the Number without cloning.
/// Use for inspection only; the reference is tied to the Context's lifetime.
#[inline]
pub(crate) fn as_number(ctx: &Context, id: ExprId) -> Option<&num_rational::BigRational> {
    match ctx.get(id) {
        Expr::Number(n) => Some(n),
        _ => None,
    }
}

/// Try to extract an i64 value from a Number expression (without cloning).
/// Returns None if not a Number, not an integer, or doesn't fit in i64.
#[inline]
pub(crate) fn as_i64(ctx: &Context, id: ExprId) -> Option<i64> {
    match ctx.get(id) {
        Expr::Number(n) if n.is_integer() => n.to_integer().to_i64(),
        _ => None,
    }
}
