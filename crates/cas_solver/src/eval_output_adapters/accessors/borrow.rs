/// Borrow eval result.
pub fn result_from_eval_output(output: &crate::EvalOutput) -> &crate::EvalResult {
    &output.result
}

/// Borrow solve-step sequence.
pub fn solve_steps_from_eval_output(output: &crate::EvalOutput) -> &[crate::SolveStep] {
    &output.solve_steps
}

/// Borrow scope tags used for context-aware rendering.
pub fn output_scopes_from_eval_output(
    output: &crate::EvalOutput,
) -> &[cas_formatter::display_transforms::ScopeTag] {
    &output.output_scopes
}

/// Borrow diagnostics payload.
pub fn diagnostics_from_eval_output(output: &crate::EvalOutput) -> &crate::Diagnostics {
    &output.diagnostics
}
