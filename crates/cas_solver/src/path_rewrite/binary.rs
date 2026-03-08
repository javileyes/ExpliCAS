mod add;
mod generic;

use cas_ast::{Context, ExprId};

pub(super) fn reconstruct_add(
    context: &mut Context,
    root: ExprId,
    l: ExprId,
    r: ExprId,
    step: &crate::PathStep,
    remaining_path: &[crate::PathStep],
    replacement: ExprId,
) -> ExprId {
    add::reconstruct_add(context, root, l, r, step, remaining_path, replacement)
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
    generic::reconstruct_sub(context, root, l, r, step, remaining_path, replacement)
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
    generic::reconstruct_mul(context, root, l, r, step, remaining_path, replacement)
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
    generic::reconstruct_div(context, root, l, r, step, remaining_path, replacement)
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
    generic::reconstruct_pow(context, root, b, e, step, remaining_path, replacement)
}
