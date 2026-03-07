use cas_ast::{Context, ExprId};

pub(crate) fn next_step_root(
    ctx: &mut Context,
    current_root: ExprId,
    step: &crate::Step,
) -> ExprId {
    if let Some(global_after) = step.global_after {
        global_after
    } else {
        crate::reconstruct_global_expr(ctx, current_root, step.path(), step.after)
    }
}
