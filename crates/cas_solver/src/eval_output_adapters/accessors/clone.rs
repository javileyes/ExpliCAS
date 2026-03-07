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

/// Clone display-ready simplify steps into solver-owned wrapper.
pub fn steps_from_eval_output(output: &crate::EvalOutput) -> crate::DisplayEvalSteps {
    crate::display_eval_steps::build_display_eval_steps(output.steps.0.clone())
}
