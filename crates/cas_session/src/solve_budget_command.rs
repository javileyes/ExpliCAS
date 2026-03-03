use crate::SessionState;

/// Evaluate a `budget` command and return a user-facing message.
pub fn evaluate_solve_budget_command_message(state: &mut SessionState, line: &str) -> String {
    let result = crate::apply_solve_budget_command(state, line);
    crate::format_solve_budget_command_message(&result)
}
