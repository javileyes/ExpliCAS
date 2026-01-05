//! Solve Safety Classification System
//!
//! Classifies rules by their safety for use during equation solving.
//! Prevents solution set corruption by filtering dangerous rules in solver pre-pass.

use crate::assumptions::ConditionClass;

/// Safety classification for rules when used during equation solving.
///
/// # Classification Guide
/// - **Always**: Global equivalence, safe in all contexts (e.g., `x + 0 → x`)
/// - **NeedsCondition**: Valid under conditions, requires assumptions (e.g., `x/x → 1` needs `x ≠ 0`)
/// - **Never**: Never use in solver context (reserved for truly dangerous rewrites)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolveSafety {
    /// Type A: Global equivalence, always safe in solver pre-pass.
    /// These rules preserve solution sets unconditionally.
    #[default]
    Always,

    /// Type B/C: Requires conditions to be valid.
    /// - Definability (B): Requires ≠0 or "is defined" conditions
    /// - Analytic (C): Requires positivity, ranges, or branch conditions
    ///
    /// In SolvePrepass: BLOCKED
    /// In SolveTactic: Allowed only if DomainMode permits unproven conditions
    NeedsCondition(ConditionClass),

    /// Never use in solver context, even as tactic.
    /// Reserved for rewrites that are inherently dangerous even with conditions.
    Never,
}

impl SolveSafety {
    /// Returns true if this rule is safe in solver pre-pass (cleanup phase).
    /// Only `Always` rules are allowed in pre-pass.
    #[inline]
    pub fn safe_for_prepass(&self) -> bool {
        matches!(self, SolveSafety::Always)
    }

    /// Returns true if this rule is safe for solver tactic given domain mode.
    /// - `Always`: always safe
    /// - `NeedsCondition(class)`: safe if mode allows unproven conditions of that class
    /// - `Never`: never safe
    #[inline]
    pub fn safe_for_tactic(&self, domain_mode: crate::domain::DomainMode) -> bool {
        match self {
            SolveSafety::Always => true,
            SolveSafety::NeedsCondition(class) => domain_mode.allows_unproven(*class),
            SolveSafety::Never => false,
        }
    }
}

/// Purpose of simplification, controls which rules are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimplifyPurpose {
    /// Standard evaluation - all rules allowed (default)
    #[default]
    Eval,

    /// Solver pre-pass - only SolveSafety::Always rules.
    /// No assumptions or hints should be generated.
    SolvePrepass,

    /// Solver tactic - conditional rules allowed with assumptions.
    /// Must respect DomainMode.allows_unproven(class).
    SolveTactic,
}

impl SimplifyPurpose {
    /// Returns true if this purpose should block assumption generation.
    #[inline]
    pub fn blocks_assumptions(&self) -> bool {
        matches!(self, SimplifyPurpose::SolvePrepass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assumptions::ConditionClass;
    use crate::domain::DomainMode;

    #[test]
    fn test_always_safe_everywhere() {
        let safety = SolveSafety::Always;
        assert!(safety.safe_for_prepass());
        assert!(safety.safe_for_tactic(DomainMode::Strict));
        assert!(safety.safe_for_tactic(DomainMode::Generic));
        assert!(safety.safe_for_tactic(DomainMode::Assume));
    }

    #[test]
    fn test_definability_blocked_in_prepass() {
        let safety = SolveSafety::NeedsCondition(ConditionClass::Definability);
        assert!(!safety.safe_for_prepass());
        // Definability allowed in Generic and Assume
        assert!(!safety.safe_for_tactic(DomainMode::Strict));
        assert!(safety.safe_for_tactic(DomainMode::Generic));
        assert!(safety.safe_for_tactic(DomainMode::Assume));
    }

    #[test]
    fn test_analytic_blocked_in_prepass() {
        let safety = SolveSafety::NeedsCondition(ConditionClass::Analytic);
        assert!(!safety.safe_for_prepass());
        // Analytic only allowed in Assume
        assert!(!safety.safe_for_tactic(DomainMode::Strict));
        assert!(!safety.safe_for_tactic(DomainMode::Generic));
        assert!(safety.safe_for_tactic(DomainMode::Assume));
    }

    #[test]
    fn test_never_blocked_everywhere() {
        let safety = SolveSafety::Never;
        assert!(!safety.safe_for_prepass());
        assert!(!safety.safe_for_tactic(DomainMode::Strict));
        assert!(!safety.safe_for_tactic(DomainMode::Generic));
        assert!(!safety.safe_for_tactic(DomainMode::Assume));
    }
}
