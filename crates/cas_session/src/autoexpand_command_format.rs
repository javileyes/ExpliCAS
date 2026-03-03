use crate::autoexpand_command_types::AutoexpandBudgetView;

/// Format status output for `autoexpand`.
pub fn format_autoexpand_current_message(
    policy: cas_solver::ExpandPolicy,
    budget: AutoexpandBudgetView,
) -> String {
    let policy_str = match policy {
        cas_solver::ExpandPolicy::Off => "off",
        cas_solver::ExpandPolicy::Auto => "on",
    };
    format!(
        "Auto-expand: {}\n\
           Budget: pow<={}, base_terms<={}, gen_terms<={}, vars<={}\n\
           (use 'autoexpand on|off' to change)",
        policy_str,
        budget.max_pow_exp,
        budget.max_base_terms,
        budget.max_generated_terms,
        budget.max_vars
    )
}

/// Format feedback after applying an auto-expand policy.
pub fn format_autoexpand_set_message(
    policy: cas_solver::ExpandPolicy,
    budget: AutoexpandBudgetView,
) -> String {
    match policy {
        cas_solver::ExpandPolicy::Auto => format!(
            "Auto-expand: on\n\
               Budget: pow<={}, base_terms<={}, gen_terms<={}, vars<={}\n\
               ⚠️ Expands small (sum)^n patterns automatically.",
            budget.max_pow_exp, budget.max_base_terms, budget.max_generated_terms, budget.max_vars
        ),
        cas_solver::ExpandPolicy::Off => {
            "Auto-expand: off\n  Polynomial expansions require explicit expand().".to_string()
        }
    }
}

/// Format unknown-mode error for `autoexpand`.
pub fn format_autoexpand_unknown_mode_message(mode: &str) -> String {
    format!(
        "Unknown autoexpand mode: '{}'\n\
             Usage: autoexpand [on | off]\n\
               on  - Auto-expand cheap polynomial powers\n\
               off - Only expand when explicitly requested (default)",
        mode
    )
}
