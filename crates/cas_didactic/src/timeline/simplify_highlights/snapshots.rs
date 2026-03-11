use super::TimelineStepSnapshots;
use crate::runtime::Step;
use cas_ast::{Context, ExprId};

pub(super) fn resolve_timeline_step_global_snapshots(
    context: &mut Context,
    steps: &[Step],
    original_expr: ExprId,
    step_idx: usize,
    step: &Step,
) -> TimelineStepSnapshots {
    let global_before_expr = step.global_before.unwrap_or_else(|| {
        if step_idx == 0 {
            original_expr
        } else {
            steps
                .get(step_idx - 1)
                .and_then(|prev| prev.global_after)
                .unwrap_or(original_expr)
        }
    });
    let global_after_expr = step.global_after.unwrap_or_else(|| {
        crate::runtime::reconstruct_global_expr(
            context,
            global_before_expr,
            step.path(),
            step.after,
        )
    });

    TimelineStepSnapshots {
        global_before_expr,
        global_after_expr,
    }
}
