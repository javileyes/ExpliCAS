mod fraction_sums;
mod rule_dispatch;

use self::fraction_sums::{
    collect_primary_fraction_sums, extend_exponent_fraction_sum_substeps,
    extend_primary_fraction_sum_substeps, standalone_fraction_sum_substeps,
};
use self::rule_dispatch::extend_step_specific_substeps;
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

        extend_primary_fraction_sum_substeps(&mut sub_steps, &unique_fraction_sums);
        extend_exponent_fraction_sum_substeps(
            ctx,
            &steps,
            step_idx,
            &unique_fraction_sums,
            &mut sub_steps,
        );
        extend_step_specific_substeps(ctx, step, &mut sub_steps);

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
    standalone_fraction_sum_substeps(ctx, original_expr)
}
