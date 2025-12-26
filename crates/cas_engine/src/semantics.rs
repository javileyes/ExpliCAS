//! Semantic configuration for evaluation and simplification.
//!
//! This module defines the 4 orthogonal semantic axes that control
//! how ExpliCAS evaluates and simplifies expressions.
//!
//! See `docs/SEMANTICS_POLICY.md` for the full specification.

use crate::domain::DomainMode;

/// Value domain for constant evaluation.
///
/// Controls the universe of values (ℝ vs ℂ).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValueDomain {
    /// Real numbers extended with ±∞ and undefined.
    /// `sqrt(-1)` → `undefined`
    #[default]
    RealOnly,

    /// Complex numbers with principal branch.
    /// `sqrt(-1)` → `i`
    ComplexEnabled,
}

/// Branch policy for multi-valued functions.
///
/// Only applies when `ValueDomain = ComplexEnabled`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BranchPolicy {
    /// Use principal branch (e.g., `log(-1) = iπ`).
    #[default]
    Principal,
}

/// Policy for inverse∘function compositions like `arctan(tan(x))`.
///
/// This is NOT the same as BranchPolicy. InverseTrigPolicy applies
/// to inverse trig functions in ℝ, not complex branch cuts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InverseTrigPolicy {
    /// Do not simplify inverse compositions.
    /// `arctan(tan(x))` → `arctan(tan(x))`
    #[default]
    Strict,

    /// Simplify assuming principal domain with warning.
    /// `arctan(tan(x))` → `x` + warning "assumed x ∈ (-π/2, π/2)"
    PrincipalValue,
}

/// Unified semantic configuration for evaluation.
///
/// This struct combines all 4 semantic axes into a single
/// configuration that can be passed through the engine.
///
/// # Example
///
/// ```ignore
/// let cfg = EvalConfig::default();
/// assert_eq!(cfg.domain_mode, DomainMode::Generic);
/// assert_eq!(cfg.value_domain, ValueDomain::RealOnly);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvalConfig {
    /// How to handle variable assumptions (≠0, >0, etc.)
    pub domain_mode: DomainMode,

    /// Universe of constants (ℝ vs ℂ)
    pub value_domain: ValueDomain,

    /// Multi-valued function branches (only if ComplexEnabled)
    pub branch: BranchPolicy,

    /// Inverse trig composition policy
    pub inv_trig: InverseTrigPolicy,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            domain_mode: DomainMode::Generic,
            value_domain: ValueDomain::RealOnly,
            branch: BranchPolicy::Principal,
            inv_trig: InverseTrigPolicy::Strict,
        }
    }
}

impl EvalConfig {
    /// Create a strict configuration (safest, no assumptions).
    pub fn strict() -> Self {
        Self {
            domain_mode: DomainMode::Strict,
            value_domain: ValueDomain::RealOnly,
            branch: BranchPolicy::Principal,
            inv_trig: InverseTrigPolicy::Strict,
        }
    }

    /// Create an assume configuration (simplifies with warnings).
    pub fn assume() -> Self {
        Self {
            domain_mode: DomainMode::Assume,
            value_domain: ValueDomain::RealOnly,
            branch: BranchPolicy::Principal,
            inv_trig: InverseTrigPolicy::Strict,
        }
    }

    /// Create a complex-enabled configuration.
    pub fn complex() -> Self {
        Self {
            domain_mode: DomainMode::Generic,
            value_domain: ValueDomain::ComplexEnabled,
            branch: BranchPolicy::Principal,
            inv_trig: InverseTrigPolicy::Strict,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = EvalConfig::default();
        assert_eq!(cfg.domain_mode, DomainMode::Generic);
        assert_eq!(cfg.value_domain, ValueDomain::RealOnly);
        assert_eq!(cfg.branch, BranchPolicy::Principal);
        assert_eq!(cfg.inv_trig, InverseTrigPolicy::Strict);
    }

    #[test]
    fn test_strict_config() {
        let cfg = EvalConfig::strict();
        assert_eq!(cfg.domain_mode, DomainMode::Strict);
        assert_eq!(cfg.inv_trig, InverseTrigPolicy::Strict);
    }

    #[test]
    fn test_assume_config() {
        let cfg = EvalConfig::assume();
        assert_eq!(cfg.domain_mode, DomainMode::Assume);
    }

    #[test]
    fn test_complex_config() {
        let cfg = EvalConfig::complex();
        assert_eq!(cfg.value_domain, ValueDomain::ComplexEnabled);
    }
}
