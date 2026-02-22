/// Budget for conditional solver branching.
///
/// Controls how many conditional branches the solver can create,
/// preventing combinatorial explosion in complex equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveBudget {
    /// Maximum number of branches that can be created (0 = no branching allowed)
    pub max_branches: usize,
    /// Maximum nesting depth for conditional solutions
    pub max_depth: usize,
}

impl Default for SolveBudget {
    fn default() -> Self {
        Self {
            max_branches: 1,
            max_depth: 2,
        }
    }
}

impl SolveBudget {
    /// No branching allowed - always return residual
    pub fn none() -> Self {
        Self {
            max_branches: 0,
            max_depth: 0,
        }
    }

    /// Check if branching is allowed
    pub fn can_branch(&self) -> bool {
        self.max_branches > 0
    }

    /// Consume one branch, returning remaining budget
    pub fn consume_branch(self) -> Self {
        Self {
            max_branches: self.max_branches.saturating_sub(1),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SolveBudget;

    #[test]
    fn default_budget_allows_branching() {
        let budget = SolveBudget::default();
        assert_eq!(budget.max_branches, 1);
        assert_eq!(budget.max_depth, 2);
        assert!(budget.can_branch());
    }

    #[test]
    fn none_budget_disables_branching() {
        let budget = SolveBudget::none();
        assert_eq!(budget.max_branches, 0);
        assert_eq!(budget.max_depth, 0);
        assert!(!budget.can_branch());
    }

    #[test]
    fn consume_branch_saturates_at_zero() {
        let budget = SolveBudget {
            max_branches: 1,
            max_depth: 3,
        };
        let next = budget.consume_branch();
        let final_budget = next.consume_branch();
        assert_eq!(next.max_branches, 0);
        assert_eq!(next.max_depth, 3);
        assert_eq!(final_budget.max_branches, 0);
    }
}
