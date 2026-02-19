use crate::helpers::is_one;
use cas_ast::{Context, Expr, ExprId};

/// Create a Mul but avoid trivial 1*x or x*1.
///
/// This is the "simplifying" builder - use for rule outputs where 1*x → x is desired.
/// Uses `add_raw` internally to preserve operand order after simplification.
///
/// Alias: `mul2_simpl` (same behavior, clearer intent)
pub(crate) fn smart_mul(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    if is_one(ctx, a) {
        return b;
    }
    if is_one(ctx, b) {
        return a;
    }
    ctx.add_raw(Expr::Mul(a, b))
}
