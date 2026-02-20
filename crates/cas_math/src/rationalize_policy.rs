//! Centralized rationalization policy and outcome types.
//!
//! This module provides a single source of truth for rationalization
//! configuration and observable rejection reasons for debugging.

/// Level of automatic rationalization to apply during simplification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AutoRationalizeLevel {
    /// No automatic rationalization
    Off,
    /// Level 0: Single surd (1/√n → √n/n)
    Level0,
    /// Level 1: Binomial denominator (1/(a+b√n) → conjugate method)
    Level1,
    /// Level 1.5: Binomial factor in product (1/(k*(a+b√n)))
    #[default]
    Level15,
}

/// Central configuration for rationalization behavior.
#[derive(Debug, Clone)]
pub struct RationalizePolicy {
    /// Which level of automatic rationalization to enable
    pub auto_level: AutoRationalizeLevel,
    /// Maximum nodes in denominator before rejecting (budget guard)
    pub max_den_nodes: usize,
    /// Maximum node growth allowed after rationalization
    pub max_growth: usize,
    /// Allow rationalizing products with same surd (e.g., (1+√2)²)
    pub allow_same_surd_product: bool,
}

impl Default for RationalizePolicy {
    fn default() -> Self {
        Self {
            auto_level: AutoRationalizeLevel::Level15,
            max_den_nodes: 30,
            max_growth: 20,
            allow_same_surd_product: true,
        }
    }
}

impl RationalizePolicy {
    /// Create a policy that disables all automatic rationalization
    pub fn disabled() -> Self {
        Self {
            auto_level: AutoRationalizeLevel::Off,
            ..Default::default()
        }
    }

    /// Check if the given level is enabled by this policy
    pub fn allows_level(&self, level: AutoRationalizeLevel) -> bool {
        match (self.auto_level, level) {
            (AutoRationalizeLevel::Off, _) => false,
            (AutoRationalizeLevel::Level0, AutoRationalizeLevel::Level0) => true,
            (
                AutoRationalizeLevel::Level1,
                AutoRationalizeLevel::Level0 | AutoRationalizeLevel::Level1,
            ) => true,
            (AutoRationalizeLevel::Level15, _) => level != AutoRationalizeLevel::Off,
            _ => false,
        }
    }
}

/// Reason why rationalization was NOT applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RationalizeReason {
    /// Multiple distinct surds in denominator (reserved for manual `rationalize`)
    MultiSurdBlocked,
    /// Denominator exceeds max_den_nodes budget
    BudgetExceeded,
    /// Result would exceed max_growth limit
    MaxGrowthExceeded,
    /// Non-binomial factor is not surd-free
    KNotSurdFree,
    /// No binomial surd pattern found in denominator
    NoBinomialFound,
    /// Policy has rationalization disabled at this level
    PolicyDisabled,
    /// Expression is not a division
    NotADivision,
    /// Zero distinct surds (nothing to rationalize)
    NoSurdsFound,
}

/// Outcome of attempting automatic rationalization.
#[derive(Debug, Clone)]
pub enum RationalizeOutcome {
    /// Rationalization was successfully applied
    Applied,
    /// Rationalization was not applied, with reason
    NotApplied(RationalizeReason),
}

impl RationalizeOutcome {
    pub fn is_applied(&self) -> bool {
        matches!(self, RationalizeOutcome::Applied)
    }

    pub fn reason(&self) -> Option<RationalizeReason> {
        match self {
            RationalizeOutcome::Applied => None,
            RationalizeOutcome::NotApplied(r) => Some(*r),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_policy_allows_all_levels() {
        let policy = RationalizePolicy::default();
        assert!(policy.allows_level(AutoRationalizeLevel::Level0));
        assert!(policy.allows_level(AutoRationalizeLevel::Level1));
        assert!(policy.allows_level(AutoRationalizeLevel::Level15));
    }

    #[test]
    fn test_disabled_policy() {
        let policy = RationalizePolicy::disabled();
        assert!(!policy.allows_level(AutoRationalizeLevel::Level0));
        assert!(!policy.allows_level(AutoRationalizeLevel::Level1));
        assert!(!policy.allows_level(AutoRationalizeLevel::Level15));
    }

    #[test]
    fn test_level1_only_policy() {
        let policy = RationalizePolicy {
            auto_level: AutoRationalizeLevel::Level1,
            ..Default::default()
        };
        assert!(policy.allows_level(AutoRationalizeLevel::Level0));
        assert!(policy.allows_level(AutoRationalizeLevel::Level1));
        assert!(!policy.allows_level(AutoRationalizeLevel::Level15));
    }
}
