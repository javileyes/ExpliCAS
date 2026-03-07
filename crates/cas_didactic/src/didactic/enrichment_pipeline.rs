use super::fraction_steps::generate_fraction_sum_substeps;
use super::fraction_sum_analysis::{
    detect_exponent_fraction_change, find_all_fraction_sums, FractionSumInfo,
};
use super::gcd_factorization::generate_gcd_factorization_substeps;
use super::nested_fractions::generate_nested_fraction_substeps;
use super::polynomial_identity::generate_polynomial_identity_substeps;
use super::rationalization::generate_rationalization_substeps;
use super::root_denesting::generate_root_denesting_substeps;
use super::sum_three_cubes::generate_sum_three_cubes_substeps;
use super::{EnrichedStep, SubStep};
use cas_ast::{Context, ExprId};
use cas_solver::Step;

/// Enrich a list of steps with didactic sub-steps
///
/// This is the main entry point for the didactic layer.
/// It analyzes each step and adds explanatory sub-steps where helpful.
pub fn enrich_steps(ctx: &Context, original_expr: ExprId, steps: Vec<Step>) -> Vec<EnrichedStep> {
    let mut enriched = Vec::with_capacity(steps.len());

    // Check original expression for fraction sums (before any simplification)
    let unique_fraction_sums = collect_primary_fraction_sums(ctx, original_expr);

    for (step_idx, step) in steps.iter().enumerate() {
        let mut sub_steps = Vec::new();

        // Attach fraction sum sub-steps to EVERY step.
        // The CLI will track and show them only once on the first visible step.
        if !unique_fraction_sums.is_empty() {
            for info in &unique_fraction_sums {
                sub_steps.extend(generate_fraction_sum_substeps(info));
            }
        }

        // Also check for fraction sums in exponent (between steps).
        if let Some(fraction_info) = detect_exponent_fraction_change(ctx, &steps, step_idx) {
            if !unique_fraction_sums
                .iter()
                .any(|o| o.fractions == fraction_info.fractions)
            {
                sub_steps.extend(generate_fraction_sum_substeps(&fraction_info));
            }
        }

        // Gate by is_chained. If this step came from ChainedRewrite,
        // the Factor→Cancel decomposition already exists as separate steps.
        if step.description.starts_with("Simplified fraction by GCD") && !step.is_chained() {
            sub_steps.extend(generate_gcd_factorization_substeps(ctx, step));
        }

        let is_nested_fraction = step.rule_name.to_lowercase().contains("complex fraction")
            || step.rule_name.to_lowercase().contains("nested fraction")
            || step.description.to_lowercase().contains("nested fraction");
        if is_nested_fraction {
            sub_steps.extend(generate_nested_fraction_substeps(ctx, step));
        }

        if step.description.contains("Rationalize") || step.rule_name.contains("Rationalize") {
            sub_steps.extend(generate_rationalization_substeps(ctx, step));
        }

        if step.poly_proof().is_some() {
            sub_steps.extend(generate_polynomial_identity_substeps(ctx, step));
        }

        if step.rule_name.contains("Sum of Three Cubes") {
            sub_steps.extend(generate_sum_three_cubes_substeps(ctx, step));
        }

        if step.rule_name.contains("Root Denesting") {
            sub_steps.extend(generate_root_denesting_substeps(ctx, step));
        }

        enriched.push(EnrichedStep {
            base_step: step.clone(),
            sub_steps,
        });
    }

    enriched
}

/// Get didactic sub-steps for an expression when there are no simplification steps.
///
/// This is useful when fraction sums are computed during parsing/canonicalization
/// and there are no engine steps to attach the explanation to.
pub fn get_standalone_substeps(ctx: &Context, original_expr: ExprId) -> Vec<SubStep> {
    let unique_fraction_sums = collect_primary_fraction_sums(ctx, original_expr);

    if unique_fraction_sums.is_empty() {
        return Vec::new();
    }

    let mut sub_steps = Vec::new();
    for info in &unique_fraction_sums {
        sub_steps.extend(generate_fraction_sum_substeps(info));
    }
    sub_steps
}

fn collect_primary_fraction_sums(ctx: &Context, original_expr: ExprId) -> Vec<FractionSumInfo> {
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
