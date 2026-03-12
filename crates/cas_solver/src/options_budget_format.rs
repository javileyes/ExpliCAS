use cas_solver_core::session_runtime::SolveBudgetCommandResult;

/// Format a `budget` command result as a user-facing message.
pub fn format_solve_budget_command_message(result: &SolveBudgetCommandResult) -> String {
    match result {
        SolveBudgetCommandResult::Current { max_branches } => format!(
            "Solve budget: max_branches={}\n\
              Controls how many case splits the solver can create.\n\
              0: No splits (fallback to simple solutions)\n\
              1: Conservative (default)\n\
              2+: Allow case splits for symbolic bases (a^x=a, etc)\n\
              (use 'budget N' to change, e.g. 'budget 2')",
            max_branches
        ),
        SolveBudgetCommandResult::Updated { max_branches } => {
            let mode_msg = if *max_branches == 0 {
                "  ⚠️ No case splits allowed (fallback to simple solutions)"
            } else if *max_branches == 1 {
                "  Conservative mode (default)"
            } else {
                "  ✓ Case splits enabled for symbolic bases\n  Try: solve a^x = a"
            };
            format!(
                "Solve budget: max_branches = {}\n{}",
                max_branches, mode_msg
            )
        }
        SolveBudgetCommandResult::Invalid { raw_value } => format!(
            "Invalid budget value: '{}' (expected a number)\n\
                         Usage: budget N\n\
                           budget 0  - No case splits\n\
                           budget 1  - Conservative (default)\n\
                           budget 2  - Allow case splits for a^x=a patterns",
            raw_value
        ),
    }
}
