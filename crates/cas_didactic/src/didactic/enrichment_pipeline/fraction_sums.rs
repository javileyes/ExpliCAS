use super::super::fraction_steps::generate_fraction_sum_substeps;
use super::super::fraction_sum_analysis::{
    detect_exponent_fraction_change, find_all_fraction_sums, FractionSumInfo,
};
use super::super::SubStep;
use cas_ast::{Context, ExprId};
use cas_solver::Step;

pub(super) fn extend_primary_fraction_sum_substeps(
    sub_steps: &mut Vec<SubStep>,
    unique_fraction_sums: &[FractionSumInfo],
) {
    for info in unique_fraction_sums {
        sub_steps.extend(generate_fraction_sum_substeps(info));
    }
}

pub(super) fn extend_exponent_fraction_sum_substeps(
    ctx: &Context,
    steps: &[Step],
    step_idx: usize,
    unique_fraction_sums: &[FractionSumInfo],
    sub_steps: &mut Vec<SubStep>,
) {
    if let Some(fraction_info) = detect_exponent_fraction_change(ctx, steps, step_idx) {
        if !unique_fraction_sums
            .iter()
            .any(|o| o.fractions == fraction_info.fractions)
        {
            sub_steps.extend(generate_fraction_sum_substeps(&fraction_info));
        }
    }
}

pub(super) fn standalone_fraction_sum_substeps(
    ctx: &Context,
    original_expr: ExprId,
) -> Vec<SubStep> {
    let unique_fraction_sums = collect_primary_fraction_sums(ctx, original_expr);
    let mut sub_steps = Vec::new();
    extend_primary_fraction_sum_substeps(&mut sub_steps, &unique_fraction_sums);
    sub_steps
}

pub(super) fn collect_primary_fraction_sums(
    ctx: &Context,
    original_expr: ExprId,
) -> Vec<FractionSumInfo> {
    let all_fraction_sums = find_all_fraction_sums(ctx, original_expr);

    if all_fraction_sums.is_empty() {
        return Vec::new();
    }

    let max_fractions = all_fraction_sums
        .iter()
        .map(|s| s.fractions.len())
        .max()
        .unwrap_or(0);
    let mut seen = std::collections::HashSet::new();
    all_fraction_sums
        .into_iter()
        .filter(|info| info.fractions.len() == max_fractions)
        .filter(|info| seen.insert(format!("{}", info.result)))
        .collect()
}
