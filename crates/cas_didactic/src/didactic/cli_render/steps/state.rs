use cas_ast::{Context, ExprId};
use cas_solver::Step;

pub(super) fn advance_current_root(ctx: &mut Context, current_root: ExprId, step: &Step) -> ExprId {
    if let Some(global_after) = step.global_after {
        global_after
    } else {
        cas_solver::reconstruct_global_expr(ctx, current_root, step.path(), step.after)
    }
}
