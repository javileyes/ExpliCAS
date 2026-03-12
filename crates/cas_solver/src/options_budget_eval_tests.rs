#[cfg(test)]
mod tests {
    use crate::options_budget_eval::SolveBudgetContext;
    use crate::session_api::budget::SolveBudgetCommandResult;
    use crate::session_api::budget::{
        apply_solve_budget_command, evaluate_solve_budget_command_message,
    };

    #[derive(Debug, Default)]
    struct TestBudgetContext {
        max_branches: usize,
    }

    impl SolveBudgetContext for TestBudgetContext {
        fn solve_budget_max_branches(&self) -> usize {
            self.max_branches
        }

        fn set_solve_budget_max_branches(&mut self, max_branches: usize) {
            self.max_branches = max_branches;
        }
    }

    #[test]
    fn apply_solve_budget_command_reads_current_value() {
        let mut context = TestBudgetContext { max_branches: 3 };
        let result = apply_solve_budget_command(&mut context, "budget");
        assert_eq!(
            result,
            SolveBudgetCommandResult::Current { max_branches: 3 }
        );
    }

    #[test]
    fn apply_solve_budget_command_updates_value() {
        let mut context = TestBudgetContext::default();
        let result = apply_solve_budget_command(&mut context, "budget 5");
        assert_eq!(
            result,
            SolveBudgetCommandResult::Updated { max_branches: 5 }
        );
        assert_eq!(context.max_branches, 5);
    }

    #[test]
    fn apply_solve_budget_command_rejects_invalid_value() {
        let mut context = TestBudgetContext::default();
        let result = apply_solve_budget_command(&mut context, "budget nope");
        assert_eq!(
            result,
            SolveBudgetCommandResult::Invalid {
                raw_value: "nope".to_string(),
            }
        );
    }

    #[test]
    fn evaluate_budget_message_uses_shared_formatter() {
        let mut context = TestBudgetContext { max_branches: 2 };
        let message = evaluate_solve_budget_command_message(&mut context, "budget");
        assert!(message.contains("Solve budget: max_branches=2"));
        assert!(message.contains("Controls how many case splits"));
    }
}
