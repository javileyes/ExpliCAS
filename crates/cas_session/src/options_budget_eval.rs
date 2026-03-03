use crate::{options_budget_types::SolveBudgetCommandResult, SessionState};

/// Apply a `budget` command:
/// - `budget` returns current value
/// - `budget N` updates `max_branches`
pub fn apply_solve_budget_command(
    state: &mut SessionState,
    input: &str,
) -> SolveBudgetCommandResult {
    let args: Vec<&str> = input.split_whitespace().collect();
    match args.get(1) {
        None => SolveBudgetCommandResult::Current {
            max_branches: state.options().budget.max_branches,
        },
        Some(value) => match value.parse::<usize>() {
            Ok(max_branches) => {
                state.options_mut().budget.max_branches = max_branches;
                SolveBudgetCommandResult::Updated { max_branches }
            }
            Err(_) => SolveBudgetCommandResult::Invalid {
                raw_value: (*value).to_string(),
            },
        },
    }
}
