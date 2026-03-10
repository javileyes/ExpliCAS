use super::super::payload::EvalOutputResultPayload;
use crate::eval_output_finalize::{EvalOutputWire, EvalOutputWireBuild};
use crate::eval_output_finalize_input::EvalOutputFinalizeShared;

pub(super) fn build_eval_output(
    payload: EvalOutputResultPayload,
    steps_count: usize,
    shared: EvalOutputFinalizeShared<'_>,
    wire: Option<serde_json::Value>,
) -> EvalOutputWire {
    let EvalOutputFinalizeShared {
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

    EvalOutputWire::from_build(EvalOutputWireBuild {
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
