//! Shared solver-option derivations.
//!
//! Centralizes branching/tactic flag derivation so runtime crates only map
//! their semantic options into core inputs.

use crate::isolation_power::{
    build_pow_isolation_kernel_config_with, PowIsolationKernelConfig, PowIsolationKernelInputs,
};
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

/// Build [`PowIsolationKernelConfig`] from core solver options and one
/// include-item hook.
pub fn pow_kernel_config_with<FCollectItem>(
    mode: DomainModeKind,
    wildcard_scope: bool,
    value_domain_real_only: bool,
    budget: SolveBudget,
    collect_item: FCollectItem,
) -> PowIsolationKernelConfig
where
    FCollectItem: FnMut() -> bool,
{
    let inputs = pow_kernel_inputs(mode, wildcard_scope, value_domain_real_only, budget);
    build_pow_isolation_kernel_config_with(collect_item, inputs)
}

/// Runtime-facing power-isolation config combining kernel flags and caller
/// tactic options payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowIsolationRuntimeConfig<TTacticOptions> {
    pub kernel: PowIsolationKernelConfig,
    pub tactic_opts: TTacticOptions,
}

/// Build a full runtime power-isolation config with caller-provided tactic
/// options constructor.
pub fn pow_runtime_config_with<TTacticOptions, FCollectItem, FBuildTacticOptions>(
    mode: DomainModeKind,
    wildcard_scope: bool,
    value_domain_real_only: bool,
    budget: SolveBudget,
    collect_item: FCollectItem,
    build_tactic_options: FBuildTacticOptions,
) -> PowIsolationRuntimeConfig<TTacticOptions>
where
    FCollectItem: FnMut() -> bool,
    FBuildTacticOptions: FnOnce() -> TTacticOptions,
{
    PowIsolationRuntimeConfig {
        kernel: pow_kernel_config_with(
            mode,
            wildcard_scope,
            value_domain_real_only,
            budget,
            collect_item,
        ),
        tactic_opts: build_tactic_options(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        log_can_branch, pow_kernel_config_with, pow_kernel_inputs, pow_runtime_config_with,
        shortcut_can_branch, solve_tactic_enabled,
    };
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

    #[test]
    fn pow_kernel_config_with_fans_out_collect_hook() {
        let budget = SolveBudget {
            max_branches: 2,
            max_depth: 4,
        };
        let mut call_count = 0usize;
        let config = pow_kernel_config_with(DomainModeKind::Assume, true, true, budget, || {
            call_count += 1;
            call_count % 2 == 1
        });

        assert_eq!(call_count, 6, "all include-item hooks should be queried");
        assert_eq!(config.mode, DomainModeKind::Assume);
        assert!(config.wildcard_scope);
        assert!(config.shortcut_can_branch);
        assert!(config.log_can_branch);
        assert!(config.solve_tactic_enabled);
    }

    #[test]
    fn pow_runtime_config_with_builds_kernel_and_tactic_payload() {
        let budget = SolveBudget {
            max_branches: 2,
            max_depth: 4,
        };
        let mut calls = 0usize;
        let config = pow_runtime_config_with(
            DomainModeKind::Assume,
            false,
            true,
            budget,
            || {
                calls += 1;
                true
            },
            || 123usize,
        );

        assert_eq!(calls, 6);
        assert_eq!(config.tactic_opts, 123usize);
        assert_eq!(config.kernel.mode, DomainModeKind::Assume);
        assert!(!config.kernel.wildcard_scope);
    }
}
