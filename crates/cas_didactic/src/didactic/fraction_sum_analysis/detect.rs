use cas_ast::{Context, ExprId};
use cas_solver::Step;

use super::FractionSumInfo;

pub(super) fn detect_exponent_fraction_change(
    ctx: &Context,
    steps: &[Step],
    step_idx: usize,
    find_fraction_sum_in_expr: fn(&Context, ExprId) -> Option<FractionSumInfo>,
) -> Option<FractionSumInfo> {
    let current_step = &steps[step_idx];
    if !current_step.rule_name.contains("Inverse") && !current_step.rule_name.contains("Power") {
        return None;
    }

    let global_expr = if step_idx > 0 {
        steps[step_idx - 1]
            .global_after
            .unwrap_or(steps[step_idx - 1].after)
    } else {
        current_step.global_after.unwrap_or(current_step.before)
    };

    find_fraction_sum_in_expr(ctx, global_expr)
}
