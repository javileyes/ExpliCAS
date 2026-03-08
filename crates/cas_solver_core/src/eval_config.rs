//! Shared semantic configuration for evaluation/simplification.

use crate::assume_scope::AssumeScope;
use crate::branch_policy::BranchPolicy;
use crate::domain_mode::DomainMode;
use crate::inverse_trig_policy::InverseTrigPolicy;
use crate::value_domain::ValueDomain;

/// Unified semantic configuration for evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EvalConfig {
    /// How to handle variable assumptions (`!=0`, `>0`, etc.).
    pub domain_mode: DomainMode,
    /// Universe of constants (R vs C).
    pub value_domain: ValueDomain,
    /// Multi-valued function branches (active when ComplexEnabled).
    pub branch: BranchPolicy,
    /// Inverse-trig composition policy.
    pub inv_trig: InverseTrigPolicy,
    /// Scope for assumptions (active when DomainMode=Assume).
    pub assume_scope: AssumeScope,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            domain_mode: DomainMode::Generic,
            value_domain: ValueDomain::RealOnly,
            branch: BranchPolicy::Principal,
            inv_trig: InverseTrigPolicy::Strict,
            assume_scope: AssumeScope::Real,
        }
    }
}

impl EvalConfig {
    /// Strict configuration (safest, no assumptions).
    pub fn strict() -> Self {
        Self {
            domain_mode: DomainMode::Strict,
            value_domain: ValueDomain::RealOnly,
            branch: BranchPolicy::Principal,
            inv_trig: InverseTrigPolicy::Strict,
            assume_scope: AssumeScope::Real,
        }
    }

    /// Assume-mode configuration (simplifies with assumptions).
    pub fn assume() -> Self {
        Self {
            domain_mode: DomainMode::Assume,
            value_domain: ValueDomain::RealOnly,
            branch: BranchPolicy::Principal,
            inv_trig: InverseTrigPolicy::Strict,
            assume_scope: AssumeScope::Real,
        }
    }

    /// Complex-enabled configuration.
    pub fn complex() -> Self {
        Self {
            domain_mode: DomainMode::Generic,
            value_domain: ValueDomain::ComplexEnabled,
            branch: BranchPolicy::Principal,
            inv_trig: InverseTrigPolicy::Strict,
            assume_scope: AssumeScope::Real,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EvalConfig;
    use crate::assume_scope::AssumeScope;
    use crate::branch_policy::BranchPolicy;
    use crate::domain_mode::DomainMode;
    use crate::inverse_trig_policy::InverseTrigPolicy;
    use crate::value_domain::ValueDomain;

    #[test]
    fn default_config_matches_contract() {
        let cfg = EvalConfig::default();
        assert_eq!(cfg.domain_mode, DomainMode::Generic);
        assert_eq!(cfg.value_domain, ValueDomain::RealOnly);
        assert_eq!(cfg.branch, BranchPolicy::Principal);
        assert_eq!(cfg.inv_trig, InverseTrigPolicy::Strict);
        assert_eq!(cfg.assume_scope, AssumeScope::Real);
    }

    #[test]
    fn constructor_presets_match_expected_axes() {
        let strict = EvalConfig::strict();
        assert_eq!(strict.domain_mode, DomainMode::Strict);

        let assume = EvalConfig::assume();
        assert_eq!(assume.domain_mode, DomainMode::Assume);

        let complex = EvalConfig::complex();
        assert_eq!(complex.value_domain, ValueDomain::ComplexEnabled);
    }
}
