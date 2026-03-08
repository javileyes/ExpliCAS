mod left_right;
mod pow;

use cas_ast::{Context, ExprId};

pub(super) fn reconstruct_sub(
    context: &mut Context,
    root: ExprId,
    l: ExprId,
    r: ExprId,
    step: &crate::PathStep,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    left_right::reconstruct_sub(context, root, l, r, step, remaining_path, replacement)
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
    left_right::reconstruct_mul(context, root, l, r, step, remaining_path, replacement)
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
    left_right::reconstruct_div(context, root, l, r, step, remaining_path, replacement)
}

pub(super) fn reconstruct_pow(
    context: &mut Context,
    root: ExprId,
    base: ExprId,
    exp: ExprId,
    step: &crate::PathStep,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    pow::reconstruct_pow(context, root, base, exp, step, remaining_path, replacement)
}
