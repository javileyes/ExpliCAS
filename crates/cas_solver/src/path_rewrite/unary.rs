use cas_ast::{Context, Expr, ExprId};

use super::reconstruct_global_expr;

pub(super) fn reconstruct_neg(
    context: &mut Context,
    root: ExprId,
    inner: ExprId,
    step: &crate::PathStep,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    match step {
        crate::PathStep::Inner => {
            let new_inner = reconstruct_global_expr(context, inner, remaining_path, replacement);
            context.add(Expr::Neg(new_inner))
        }
        _ => root,
    }
}

pub(super) fn reconstruct_hold(
    context: &mut Context,
    root: ExprId,
    inner: ExprId,
    step: &crate::PathStep,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    match step {
        crate::PathStep::Inner => {
            let new_inner = reconstruct_global_expr(context, inner, remaining_path, replacement);
            context.add(Expr::Hold(new_inner))
        }
        _ => root,
    }
}
