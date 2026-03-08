use cas_ast::{Context, Expr, ExprId};

use super::super::super::reconstruct_global_expr;

pub(super) fn reconstruct_sub(
    context: &mut Context,
    root: ExprId,
    l: ExprId,
    r: ExprId,
    step: &crate::PathStep,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    reconstruct_left_right_binary(
        context,
        root,
        l,
        r,
        step,
        remaining_path,
        replacement,
        Expr::Sub,
    )
}

pub(super) fn reconstruct_mul(
    context: &mut Context,
    root: ExprId,
    l: ExprId,
    r: ExprId,
    step: &crate::PathStep,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    reconstruct_left_right_binary(
        context,
        root,
        l,
        r,
        step,
        remaining_path,
        replacement,
        Expr::Mul,
    )
}

pub(super) fn reconstruct_div(
    context: &mut Context,
    root: ExprId,
    l: ExprId,
    r: ExprId,
    step: &crate::PathStep,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    reconstruct_left_right_binary(
        context,
        root,
        l,
        r,
        step,
        remaining_path,
        replacement,
        Expr::Div,
    )
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_left_right_binary(
    context: &mut Context,
    root: ExprId,
    l: ExprId,
    r: ExprId,
    step: &crate::PathStep,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
    build: fn(ExprId, ExprId) -> Expr,
) -> ExprId {
    match step {
        crate::PathStep::Left => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(build(new_l, r))
        }
        crate::PathStep::Right => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(build(l, new_r))
        }
        _ => root,
    }
}
