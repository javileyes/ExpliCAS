mod detect;
mod probe;
mod scan;
mod types;

use cas_ast::{Context, ExprId};
use cas_solver::Step;

use self::probe::find_fraction_sum_in_expr;
pub(crate) use self::types::FractionSumInfo;

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
    detect::detect_exponent_fraction_change(ctx, steps, step_idx, find_fraction_sum_in_expr)
}
