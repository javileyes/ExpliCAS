use super::{
    assumption_records_from_eval_output, blocked_hints_from_eval_output,
    diagnostics_from_eval_output, domain_warnings_from_eval_output, output_scopes_from_eval_output,
    parsed_expr_from_eval_output, required_conditions_from_eval_output,
    resolved_expr_from_eval_output, result_from_eval_output, solve_steps_from_eval_output,
    steps_from_eval_output, stored_id_from_eval_output,
};

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
        steps: steps_from_eval_output(output),
        solve_steps: solve_steps_from_eval_output(output).to_vec(),
        output_scopes: output_scopes_from_eval_output(output).to_vec(),
        diagnostics: diagnostics_from_eval_output(output).clone(),
        required_conditions: required_conditions_from_eval_output(output),
        domain_warnings: domain_warnings_from_eval_output(output),
        blocked_hints: blocked_hints_from_eval_output(output),
        solver_assumptions: assumption_records_from_eval_output(output),
    }
}
