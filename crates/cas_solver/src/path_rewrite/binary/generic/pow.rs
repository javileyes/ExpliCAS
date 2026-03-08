use cas_ast::{Context, Expr, ExprId};

use super::super::super::reconstruct_global_expr;

pub(super) fn reconstruct_pow(
    context: &mut Context,
    root: ExprId,
    base: ExprId,
    exp: ExprId,
    step: &crate::PathStep,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    match step {
        crate::PathStep::Base => {
            let new_base = reconstruct_global_expr(context, base, remaining_path, replacement);
            context.add(Expr::Pow(new_base, exp))
        }
        crate::PathStep::Exponent => {
            let new_exp = reconstruct_global_expr(context, exp, remaining_path, replacement);
            context.add(Expr::Pow(base, new_exp))
        }
        _ => root,
    }
}
