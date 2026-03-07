use super::SessionState;

impl cas_solver::SolveBudgetContext for SessionState {
    fn solve_budget_max_branches(&self) -> usize {
        self.options().budget.max_branches
    }

    fn set_solve_budget_max_branches(&mut self, max_branches: usize) {
        self.options_mut().budget.max_branches = max_branches;
    }
}
