use crate::SessionState;

/// Result of applying a `budget` command against session options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveBudgetCommandResult {
    Current { max_branches: usize },
    Updated { max_branches: usize },
    Invalid { raw_value: String },
}

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

#[cfg(test)]
mod tests {
    use super::{
        apply_solve_budget_command, format_solve_budget_command_message, SolveBudgetCommandResult,
    };
    use crate::SessionState;

    #[test]
    fn apply_solve_budget_command_reads_current_value() {
        let mut state = SessionState::new();
        state.options_mut().budget.max_branches = 3;
        let result = apply_solve_budget_command(&mut state, "budget");
        assert_eq!(
            result,
            SolveBudgetCommandResult::Current { max_branches: 3 }
        );
    }

    #[test]
    fn apply_solve_budget_command_updates_value() {
        let mut state = SessionState::new();
        let result = apply_solve_budget_command(&mut state, "budget 5");
        assert_eq!(
            result,
            SolveBudgetCommandResult::Updated { max_branches: 5 }
        );
        assert_eq!(state.options().budget.max_branches, 5);
    }

    #[test]
    fn apply_solve_budget_command_rejects_invalid_value() {
        let mut state = SessionState::new();
        let result = apply_solve_budget_command(&mut state, "budget nope");
        assert_eq!(
            result,
            SolveBudgetCommandResult::Invalid {
                raw_value: "nope".to_string(),
            }
        );
    }

    #[test]
    fn format_solve_budget_command_message_current_mentions_controls() {
        let message = format_solve_budget_command_message(&SolveBudgetCommandResult::Current {
            max_branches: 2,
        });
        assert!(message.contains("Solve budget: max_branches=2"));
        assert!(message.contains("Controls how many case splits"));
    }

    #[test]
    fn format_solve_budget_command_message_updated_mentions_mode() {
        let message = format_solve_budget_command_message(&SolveBudgetCommandResult::Updated {
            max_branches: 0,
        });
        assert!(message.contains("max_branches = 0"));
        assert!(message.contains("No case splits allowed"));
    }

    #[test]
    fn format_solve_budget_command_message_invalid_mentions_usage() {
        let message = format_solve_budget_command_message(&SolveBudgetCommandResult::Invalid {
            raw_value: "x".to_string(),
        });
        assert!(message.contains("Invalid budget value: 'x'"));
        assert!(message.contains("Usage: budget N"));
    }
}
