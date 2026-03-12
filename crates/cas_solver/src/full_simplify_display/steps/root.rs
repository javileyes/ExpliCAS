use cas_ast::{Context, ExprId};
use cas_solver_core::path_rewrite::reconstruct_global_expr;

pub(crate) fn next_step_root(
    ctx: &mut Context,
    current_root: ExprId,
    step: &crate::Step,
) -> ExprId {
    if let Some(global_after) = step.global_after {
        global_after
    } else {
        reconstruct_global_expr(ctx, current_root, step.path(), step.after)
    }
}
