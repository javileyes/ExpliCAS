mod fraction_sums;
mod rule_dispatch;
mod setup;
mod standalone;
mod step_loop;

use self::fraction_sums::{
    collect_primary_fraction_sums, extend_exponent_fraction_sum_substeps,
    extend_primary_fraction_sum_substeps, standalone_fraction_sum_substeps,
};
use super::{EnrichedStep, SubStep};
use crate::runtime::Step;
use cas_ast::{Context, ExprId};

/// Enrich a list of steps with didactic sub-steps
///
/// This is the main entry point for the didactic layer.
/// It analyzes each step and adds explanatory sub-steps where helpful.
pub fn enrich_steps(ctx: &Context, original_expr: ExprId, steps: Vec<Step>) -> Vec<EnrichedStep> {
    let unique_fraction_sums =
        setup::prepare_fraction_sum_context(ctx, original_expr, collect_primary_fraction_sums);
    step_loop::enrich_step_loop(
        ctx,
        &steps,
        &unique_fraction_sums,
        extend_primary_fraction_sum_substeps,
        extend_exponent_fraction_sum_substeps,
        rule_dispatch::extend_step_specific_substeps,
    )
}

/// Get didactic sub-steps for an expression when there are no simplification steps.
///
/// This is useful when fraction sums are computed during parsing/canonicalization
/// and there are no engine steps to attach the explanation to.
pub fn get_standalone_substeps(ctx: &Context, original_expr: ExprId) -> Vec<SubStep> {
    standalone::get_standalone_substeps(ctx, original_expr, standalone_fraction_sum_substeps)
}
