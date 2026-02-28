//! Shared solver-option derivations.
//!
//! Centralizes branching/tactic flag derivation so runtime crates only map
//! their semantic options into core inputs.

use crate::isolation_power::PowIsolationKernelInputs;
use crate::log_domain::DomainModeKind;
use crate::solve_budget::SolveBudget;

/// Returns true when exponent-shortcut branching can explore both signs.
pub fn shortcut_can_branch(budget: SolveBudget) -> bool {
    budget.max_branches >= 2
}

/// Returns true when log-isolation may branch under current budget.
pub fn log_can_branch(budget: SolveBudget) -> bool {
    budget.can_branch()
}

/// Returns true when solve-tactic normalization is allowed.
pub fn solve_tactic_enabled(mode: DomainModeKind, value_domain_real_only: bool) -> bool {
    mode == DomainModeKind::Assume && value_domain_real_only
}

/// Build power-isolation kernel inputs from core solver options.
pub fn pow_kernel_inputs(
    mode: DomainModeKind,
    wildcard_scope: bool,
    value_domain_real_only: bool,
    budget: SolveBudget,
) -> PowIsolationKernelInputs {
    PowIsolationKernelInputs {
        shortcut_can_branch: shortcut_can_branch(budget),
        log_can_branch: log_can_branch(budget),
        solve_tactic_enabled: solve_tactic_enabled(mode, value_domain_real_only),
        mode,
        wildcard_scope,
    }
}

#[cfg(test)]
mod tests {
    use super::{log_can_branch, pow_kernel_inputs, shortcut_can_branch, solve_tactic_enabled};
    use crate::log_domain::DomainModeKind;
    use crate::solve_budget::SolveBudget;

    #[test]
    fn branch_flags_follow_budget_contract() {
        let none = SolveBudget::none();
        let one = SolveBudget {
            max_branches: 1,
            ..Default::default()
        };
        let two = SolveBudget {
            max_branches: 2,
            ..Default::default()
        };

        assert!(!shortcut_can_branch(none));
        assert!(!shortcut_can_branch(one));
        assert!(shortcut_can_branch(two));

        assert!(!log_can_branch(none));
        assert!(log_can_branch(one));
    }

    #[test]
    fn tactic_flag_requires_assume_and_real_only() {
        assert!(solve_tactic_enabled(DomainModeKind::Assume, true));
        assert!(!solve_tactic_enabled(DomainModeKind::Assume, false));
        assert!(!solve_tactic_enabled(DomainModeKind::Generic, true));
        assert!(!solve_tactic_enabled(DomainModeKind::Strict, true));
    }

    #[test]
    fn pow_inputs_pack_all_fields() {
        let budget = SolveBudget {
            max_branches: 3,
            max_depth: 2,
        };
        let inputs = pow_kernel_inputs(DomainModeKind::Assume, true, true, budget);
        assert!(inputs.shortcut_can_branch);
        assert!(inputs.log_can_branch);
        assert!(inputs.solve_tactic_enabled);
        assert_eq!(inputs.mode, DomainModeKind::Assume);
        assert!(inputs.wildcard_scope);
    }
}
