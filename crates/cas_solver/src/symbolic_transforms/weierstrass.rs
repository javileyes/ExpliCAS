mod recurse;
mod trig;

use cas_ast::{Context, ExprId};

/// Recursively apply Weierstrass substitution:
/// `t = tan(x/2) = sin(x/2)/cos(x/2)`.
pub fn apply_weierstrass_recursive(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();

    if let Some(rewritten) = trig::rewrite_trig_function(ctx, &expr_data) {
        return rewritten;
    }

    recurse::rewrite_children(ctx, expr, expr_data, apply_weierstrass_recursive)
}
