mod probe;
mod scan;

use cas_ast::{Context, ExprId};
use cas_solver::Step;
use num_rational::BigRational;

use self::probe::find_fraction_sum_in_expr;
use super::IsOne;

/// Information about a fraction sum that was computed.
#[derive(Debug)]
pub(crate) struct FractionSumInfo {
    /// The fractions that were summed.
    pub fractions: Vec<BigRational>,
    /// The result of the sum.
    pub result: BigRational,
}

/// Find all fraction sums in an expression tree.
pub(crate) fn find_all_fraction_sums(ctx: &Context, expr: ExprId) -> Vec<FractionSumInfo> {
    let mut results = Vec::new();
    scan::find_all_fraction_sums_recursive(ctx, expr, &mut results);
    results
}

/// Detect if between this step and the previous one, an exponent changed
/// due to fraction arithmetic.
pub(crate) fn detect_exponent_fraction_change(
    ctx: &Context,
    steps: &[Step],
    step_idx: usize,
) -> Option<FractionSumInfo> {
    let current_step = &steps[step_idx];

    if current_step.rule_name.contains("Inverse") || current_step.rule_name.contains("Power") {
        let global_expr = if step_idx > 0 {
            steps[step_idx - 1]
                .global_after
                .unwrap_or(steps[step_idx - 1].after)
        } else {
            current_step.global_after.unwrap_or(current_step.before)
        };

        if let Some(info) = find_fraction_sum_in_expr(ctx, global_expr) {
            return Some(info);
        }
    }

    None
}
