#[cfg(test)]
mod tests {
    use crate::state_core::SessionState;
    #[allow(unused_imports)]
    use cas_solver::session_api::{assumptions::*, budget::*, runtime::*, simplifier::*};

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
