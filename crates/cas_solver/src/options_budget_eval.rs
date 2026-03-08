use crate::{format_solve_budget_command_message, SolveBudgetCommandResult};
pub use cas_solver_core::session_runtime::SolveBudgetContext;

/// Apply a `budget` command:
/// - `budget` returns current value
/// - `budget N` updates `max_branches`
pub fn apply_solve_budget_command<C: SolveBudgetContext>(
    context: &mut C,
    input: &str,
) -> SolveBudgetCommandResult {
    let args: Vec<&str> = input.split_whitespace().collect();
    match args.get(1) {
        None => SolveBudgetCommandResult::Current {
            max_branches: context.solve_budget_max_branches(),
        },
        Some(value) => match value.parse::<usize>() {
            Ok(max_branches) => {
                context.set_solve_budget_max_branches(max_branches);
                SolveBudgetCommandResult::Updated { max_branches }
            }
            Err(_) => SolveBudgetCommandResult::Invalid {
                raw_value: (*value).to_string(),
            },
        },
    }
}

/// Evaluate a `budget` command and return a user-facing message.
pub fn evaluate_solve_budget_command_message<C: SolveBudgetContext>(
    context: &mut C,
    line: &str,
) -> String {
    let result = apply_solve_budget_command(context, line);
    format_solve_budget_command_message(&result)
}
