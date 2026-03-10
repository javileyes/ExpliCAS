use crate::eval_output_finalize::finalize_eval_output;
use crate::eval_output_finalize_input::EvalOutputFinalizeInput;

use super::collect::CollectedEvalArtifacts;
use super::{EvalCommandOutput, EvalCommandRunConfig, PreparedEvalRun};

pub(super) fn finalize_eval_collected(
    engine: &mut crate::Engine,
    config: EvalCommandRunConfig<'_>,
    prepared: PreparedEvalRun,
    collected: CollectedEvalArtifacts,
) -> Result<EvalCommandOutput, String> {
    finalize_eval_output(EvalOutputFinalizeInput {
        result: &prepared.output_view.result,
        ctx: &engine.simplifier.context,
        max_chars: config.max_chars,
        input: config.expr,
        input_latex: collected.input_latex,
        steps_mode: config.steps_mode,
        steps: collected.steps,
        solve_steps: collected.solve_steps,
        warnings: collected.warnings,
        required_conditions: collected.required_conditions,
        required_display: collected.required_display,
        raw_steps_count: collected.raw_steps_count,
        raw_solve_steps_count: collected.raw_solve_steps_count,
        budget_preset: config.budget_preset,
        strict: config.strict,
        domain: config.domain,
        timings_us: collected.timings_us,
        context_mode: config.context_mode,
        branch_mode: config.branch_mode,
        expand_policy: config.expand_policy,
        complex_mode: config.complex_mode,
        const_fold: config.const_fold,
        value_domain: config.value_domain,
        complex_branch: config.complex_branch,
        inv_trig: config.inv_trig,
        assume_scope: config.assume_scope,
    })
}
