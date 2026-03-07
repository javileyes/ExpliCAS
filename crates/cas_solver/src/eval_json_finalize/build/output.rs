use cas_api_models::{EvalJsonOutput, EvalJsonOutputBuild};

use super::payload::EvalJsonResultPayload;
use crate::eval_json_finalize_input::EvalJsonFinalizeShared;
use crate::eval_json_finalize_wire::build_eval_wire_value;

pub(crate) fn build_eval_json_output(
    payload: EvalJsonResultPayload,
    steps_count: usize,
    shared: EvalJsonFinalizeShared<'_>,
) -> EvalJsonOutput {
    let EvalJsonFinalizeShared {
        input,
        input_latex,
        steps_mode,
        steps,
        solve_steps,
        warnings,
        required_conditions,
        required_display,
        budget_preset,
        strict,
        domain,
        timings_us,
        context_mode,
        branch_mode,
        expand_policy,
        complex_mode,
        const_fold,
        value_domain,
        complex_branch,
        inv_trig,
        assume_scope,
        ..
    } = shared;

    let wire = build_eval_wire_value(
        &warnings,
        &required_display,
        &payload.result,
        payload.result_latex.as_deref(),
        steps_count,
        steps_mode,
    );

    EvalJsonOutput::from_build(EvalJsonOutputBuild {
        input,
        input_latex,
        result_chars: payload.result_chars,
        result: payload.result,
        result_truncated: payload.result_truncated,
        result_latex: payload.result_latex,
        steps_mode,
        steps_count,
        steps,
        solve_steps,
        warnings,
        required_conditions,
        required_display,
        budget_preset,
        strict,
        domain,
        stats: payload.stats,
        hash: payload.hash,
        timings_us,
        context_mode,
        branch_mode,
        expand_policy,
        complex_mode,
        const_fold,
        value_domain,
        complex_branch,
        inv_trig,
        assume_scope,
        wire,
    })
}
