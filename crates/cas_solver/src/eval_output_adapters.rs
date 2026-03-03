//! Helpers to project engine-backed `EvalOutput` into solver-owned view types.
//!
//! These adapters keep conversion logic in one place so frontend crates don't
//! need to access raw `EvalOutput` transport fields directly.

/// Owned, frontend-oriented projection of an eval output payload.
#[derive(Debug, Clone)]
pub struct EvalOutputView {
    pub stored_id: Option<u64>,
    pub parsed: cas_ast::ExprId,
    pub resolved: cas_ast::ExprId,
    pub result: crate::EvalResult,
    pub steps: crate::DisplayEvalSteps,
    pub solve_steps: Vec<crate::SolveStep>,
    pub output_scopes: Vec<cas_formatter::display_transforms::ScopeTag>,
    pub diagnostics: crate::Diagnostics,
    pub required_conditions: Vec<crate::ImplicitCondition>,
    pub domain_warnings: Vec<crate::DomainWarning>,
    pub blocked_hints: Vec<crate::BlockedHint>,
    pub solver_assumptions: Vec<crate::AssumptionRecord>,
}

/// Project eval output into an owned solver view for frontend/application layers.
pub fn eval_output_view(output: &crate::EvalOutput) -> EvalOutputView {
    EvalOutputView {
        stored_id: stored_id_from_eval_output(output),
        parsed: parsed_expr_from_eval_output(output),
        resolved: resolved_expr_from_eval_output(output),
        result: result_from_eval_output(output).clone(),
        steps: steps_from_eval_output(output).clone(),
        solve_steps: solve_steps_from_eval_output(output).to_vec(),
        output_scopes: output_scopes_from_eval_output(output).to_vec(),
        diagnostics: diagnostics_from_eval_output(output).clone(),
        required_conditions: required_conditions_from_eval_output(output),
        domain_warnings: domain_warnings_from_eval_output(output),
        blocked_hints: blocked_hints_from_eval_output(output),
        solver_assumptions: assumption_records_from_eval_output(output),
    }
}

/// Convert solver-assumption payload from an eval output into solver-owned records.
pub fn assumption_records_from_eval_output(
    output: &crate::EvalOutput,
) -> Vec<crate::AssumptionRecord> {
    crate::assumption_model::assumption_records_from_engine(&output.solver_assumptions)
}

/// Convert blocked-hint payload from an eval output into solver-owned hints.
pub fn blocked_hints_from_eval_output(output: &crate::EvalOutput) -> Vec<crate::BlockedHint> {
    crate::blocked_hint::blocked_hints_from_engine(&output.blocked_hints)
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

/// Borrow display-ready simplify steps.
pub fn steps_from_eval_output(output: &crate::EvalOutput) -> &crate::DisplayEvalSteps {
    &output.steps
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
