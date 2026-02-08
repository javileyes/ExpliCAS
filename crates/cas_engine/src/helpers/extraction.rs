// ========== Expression Value Extraction ==========

use cas_ast::{Context, Expr, ExprId};
use num_traits::ToPrimitive;

/// Try to extract an integer value from an expression.
///
/// Returns `None` if:
/// - Expression is not a Number
/// - Number is not an integer (has non-1 denominator)
/// - Integer value doesn't fit in `i64`
///
/// For BigInt extraction without i64 limitations, use `get_integer_exact`.
pub(crate) fn get_integer(ctx: &Context, expr: ExprId) -> Option<i64> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            n.to_integer().to_i64()
        } else {
            None
        }
    } else {
        None
    }
}

/// Extract an integer value from an expression as BigInt.
///
/// Unlike `get_integer`, this:
/// - Returns the full BigInt without i64 truncation
/// - Also handles `Neg(e)` by recursively extracting and negating
///
/// Use this for number theory operations where large integers are expected.
pub(crate) fn get_integer_exact(ctx: &Context, expr: ExprId) -> Option<num_bigint::BigInt> {
    match ctx.get(expr) {
        Expr::Number(n) => {
            if n.is_integer() {
                Some(n.to_integer())
            } else {
                None
            }
        }
        Expr::Neg(e) => get_integer_exact(ctx, *e).map(|n| -n),
        _ => None,
    }
}
