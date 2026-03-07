/// Convert solver-assumption payload from an eval output into solver-owned records.
pub fn assumption_records_from_eval_output(
    output: &crate::EvalOutput,
) -> Vec<crate::AssumptionRecord> {
    output.solver_assumptions.to_vec()
}

/// Convert blocked-hint payload from an eval output into solver-owned hints.
pub fn blocked_hints_from_eval_output(output: &crate::EvalOutput) -> Vec<crate::BlockedHint> {
    output.blocked_hints.to_vec()
}

/// Clone domain warnings from eval output.
pub fn domain_warnings_from_eval_output(output: &crate::EvalOutput) -> Vec<crate::DomainWarning> {
    output.domain_warnings.clone()
}

/// Clone required conditions from eval output.
pub fn required_conditions_from_eval_output(
    output: &crate::EvalOutput,
) -> Vec<crate::ImplicitCondition> {
    output.required_conditions.clone()
}

/// Get stored history id, if any.
pub fn stored_id_from_eval_output(output: &crate::EvalOutput) -> Option<u64> {
    output.stored_id
}

/// Get parsed expression id.
pub fn parsed_expr_from_eval_output(output: &crate::EvalOutput) -> cas_ast::ExprId {
    output.parsed
}

/// Get resolved expression id.
pub fn resolved_expr_from_eval_output(output: &crate::EvalOutput) -> cas_ast::ExprId {
    output.resolved
}

/// Borrow eval result.
pub fn result_from_eval_output(output: &crate::EvalOutput) -> &crate::EvalResult {
    &output.result
}

/// Clone display-ready simplify steps into solver-owned wrapper.
pub fn steps_from_eval_output(output: &crate::EvalOutput) -> crate::DisplayEvalSteps {
    crate::display_eval_steps::build_display_eval_steps(output.steps.0.clone())
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
