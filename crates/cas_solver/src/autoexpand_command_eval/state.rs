use cas_solver_core::autoexpand_command_types::AutoexpandBudgetView;

/// Build an autoexpand budget view from eval options.
pub fn autoexpand_budget_view_from_options(
    eval_options: &crate::EvalOptions,
) -> AutoexpandBudgetView {
    let budget = &eval_options.shared.expand_budget;
    AutoexpandBudgetView {
        max_pow_exp: budget.max_pow_exp,
        max_base_terms: budget.max_base_terms,
        max_generated_terms: budget.max_generated_terms,
        max_vars: budget.max_vars,
    }
}
