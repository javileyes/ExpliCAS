use super::super::fraction_sum_analysis::FractionSumInfo;
use super::super::{EnrichedStep, SubStep};
use crate::cas_solver::Step;
use cas_ast::Context;

pub(super) fn enrich_step_loop(
    ctx: &Context,
    steps: &[Step],
    unique_fraction_sums: &[FractionSumInfo],
    extend_primary_fraction_sum_substeps: fn(&mut Vec<SubStep>, &[FractionSumInfo]),
    extend_exponent_fraction_sum_substeps: fn(
        &Context,
        &[Step],
        usize,
        &[FractionSumInfo],
        &mut Vec<SubStep>,
    ),
    extend_step_specific_substeps: fn(&Context, &Step, &mut Vec<SubStep>),
) -> Vec<EnrichedStep> {
    let mut enriched = Vec::with_capacity(steps.len());

    for (step_idx, step) in steps.iter().enumerate() {
        let mut sub_steps = Vec::new();

        extend_primary_fraction_sum_substeps(&mut sub_steps, unique_fraction_sums);
        extend_exponent_fraction_sum_substeps(
            ctx,
            steps,
            step_idx,
            unique_fraction_sums,
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
