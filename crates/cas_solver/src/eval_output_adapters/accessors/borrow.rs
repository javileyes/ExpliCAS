/// Borrow eval result.
pub(crate) fn result_from_eval_output(output: &crate::EvalOutput) -> &crate::EvalResult {
    &output.result
}

/// Borrow solve-step sequence.
pub(crate) fn solve_steps_from_eval_output(output: &crate::EvalOutput) -> &[crate::SolveStep] {
    &output.solve_steps
}

/// Borrow scope tags used for context-aware rendering.
pub(crate) fn output_scopes_from_eval_output(
    output: &crate::EvalOutput,
) -> &[cas_formatter::display_transforms::ScopeTag] {
    &output.output_scopes
}

/// Borrow diagnostics payload.
pub(crate) fn diagnostics_from_eval_output(output: &crate::EvalOutput) -> &crate::Diagnostics {
    &output.diagnostics
}
