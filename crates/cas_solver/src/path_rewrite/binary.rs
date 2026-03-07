use cas_ast::{Context, Expr, ExprId};

use super::reconstruct_global_expr;

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
        crate::PathStep::Left => {
            if let Expr::Neg(inner) = context.get(l).clone() {
                let new_inner =
                    reconstruct_global_expr(context, inner, remaining_path, replacement);
                let new_neg = context.add(Expr::Neg(new_inner));
                context.add(Expr::Add(new_neg, r))
            } else {
                let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
                context.add(Expr::Add(new_l, r))
            }
        }
        crate::PathStep::Right => {
            if let Expr::Neg(inner) = context.get(r).clone() {
                let new_inner =
                    reconstruct_global_expr(context, inner, remaining_path, replacement);
                let new_neg = context.add(Expr::Neg(new_inner));
                context.add(Expr::Add(l, new_neg))
            } else {
                let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
                context.add(Expr::Add(l, new_r))
            }
        }
        _ => root,
    }
}

pub(super) fn reconstruct_sub(
    context: &mut Context,
    root: ExprId,
    l: ExprId,
    r: ExprId,
    step: &crate::PathStep,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    match step {
        crate::PathStep::Left => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Sub(new_l, r))
        }
        crate::PathStep::Right => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Sub(l, new_r))
        }
        _ => root,
    }
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
    match step {
        crate::PathStep::Left => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Mul(new_l, r))
        }
        crate::PathStep::Right => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Mul(l, new_r))
        }
        _ => root,
    }
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
    match step {
        crate::PathStep::Left => {
            let new_l = reconstruct_global_expr(context, l, remaining_path, replacement);
            context.add(Expr::Div(new_l, r))
        }
        crate::PathStep::Right => {
            let new_r = reconstruct_global_expr(context, r, remaining_path, replacement);
            context.add(Expr::Div(l, new_r))
        }
        _ => root,
    }
}

pub(super) fn reconstruct_pow(
    context: &mut Context,
    root: ExprId,
    b: ExprId,
    e: ExprId,
    step: &crate::PathStep,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    match step {
        crate::PathStep::Base => {
            let new_b = reconstruct_global_expr(context, b, remaining_path, replacement);
            context.add(Expr::Pow(new_b, e))
        }
        crate::PathStep::Exponent => {
            let new_e = reconstruct_global_expr(context, e, remaining_path, replacement);
            context.add(Expr::Pow(b, new_e))
        }
        _ => root,
    }
}
