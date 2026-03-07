use cas_ast::{Context, Expr, ExprId};

use super::super::reconstruct_global_expr;

pub(super) fn reconstruct_add(
    context: &mut Context,
    root: ExprId,
    l: ExprId,
    r: ExprId,
    step: &crate::PathStep,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    match step {
        crate::PathStep::Left => reconstruct_add_left(context, l, r, remaining_path, replacement),
        crate::PathStep::Right => reconstruct_add_right(context, l, r, remaining_path, replacement),
        _ => root,
    }
}

fn reconstruct_add_left(
    context: &mut Context,
    l: ExprId,
    r: ExprId,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    if let Expr::Neg(inner) = context.get(l).clone() {
        let new_inner = reconstruct_global_expr(context, inner, remaining_path, replacement);
        let new_neg = context.add(Expr::Neg(new_inner));
        context.add(Expr::Add(new_neg, r))
    } else {
        let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
        context.add(Expr::Add(new_l, r))
    }
}

fn reconstruct_add_right(
    context: &mut Context,
    l: ExprId,
    r: ExprId,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    if let Expr::Neg(inner) = context.get(r).clone() {
        let new_inner = reconstruct_global_expr(context, inner, remaining_path, replacement);
        let new_neg = context.add(Expr::Neg(new_inner));
        context.add(Expr::Add(l, new_neg))
    } else {
        let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
        context.add(Expr::Add(l, new_r))
    }
}
