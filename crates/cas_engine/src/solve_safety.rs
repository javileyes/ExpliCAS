//! Solve Safety Classification System
//!
//! Classifies rules by their safety for use during equation solving.
//! Prevents solution set corruption by filtering dangerous rules in solver pre-pass.
//!
//! # Bridge to Domain Vocabulary
//!
//! [`RequirementDescriptor`] maps the static rule metadata (`SolveSafety`) to
//! the dynamic domain vocabulary (`ConditionClass` + `Provenance`), making
//! the relationship machine-queryable without collapsing two distinct concerns
//! into a single type.

use crate::assumptions::ConditionClass;
use crate::domain_facts::Provenance;

/// Safety classification for rules when used during equation solving.
///
/// # Classification Guide
/// - **Always**: Global equivalence, safe in all contexts (e.g., `x + 0 → x`)
/// - **IntrinsicCondition**: Condition is intrinsic to the input expression (e.g., `exp(ln(x))→x`;
///   `ln(x)` already guarantees `x > 0`). Allowed in SolveTactic(Generic/Assume).
/// - **NeedsCondition**: Valid under conditions that must be introduced (e.g., `x/x → 1` needs `x ≠ 0`)
/// - **Never**: Never use in solver context (reserved for truly dangerous rewrites)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolveSafety {
    /// Type A: Global equivalence, always safe in solver pre-pass.
    /// These rules preserve solution sets unconditionally.
    #[default]
    Always,

    /// Type B-intrinsic: Condition is already guaranteed by the input AST.
    /// The rule's requirement is a precondition of an operator already present
    /// in the expression (e.g., `ln(x)` intrinsically requires `x > 0`).
    ///
    /// In SolvePrepass: BLOCKED (prepass must not change structure)
    /// In SolveTactic(Generic/Assume): ALLOWED (condition is inherited, not introduced)
    /// In SolveTactic(Strict): BLOCKED (Strict requires formal proof)
    IntrinsicCondition(ConditionClass),

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

// =============================================================================
// RequirementDescriptor — Bridge to domain vocabulary
// =============================================================================

/// Bridge between static rule safety metadata and the dynamic domain vocabulary.
///
/// Preserves both the condition class (Definability vs Analytic) and the
/// provenance (Intrinsic vs Introduced), allowing consumers to query
/// "what kind of condition does this rule need?" without pattern-matching
/// on `SolveSafety` variants directly.
///
/// # Mapping
///
/// | `SolveSafety` | `class` | `provenance` |
/// |---|---|---|
/// | `IntrinsicCondition(c)` | `c` | `Intrinsic` |
/// | `NeedsCondition(c)` | `c` | `Introduced` |
/// | `Always` / `Never` | — | — (returns `None`) |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequirementDescriptor {
    /// What kind of condition the rule needs (Definability or Analytic).
    pub class: ConditionClass,
    /// Where the condition comes from: inherited from the AST or introduced by the rule.
    pub provenance: Provenance,
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
            SolveSafety::IntrinsicCondition(_class) => {
                // Intrinsic conditions are inherited from the input AST, not introduced.
                // Allowed in Generic and Assume; blocked in Strict (requires formal proof).
                domain_mode != crate::domain::DomainMode::Strict
            }
            SolveSafety::NeedsCondition(class) => domain_mode.allows_unproven(*class),
            SolveSafety::Never => false,
        }
    }

    /// Bridge to domain vocabulary: maps this safety classification to a
    /// [`RequirementDescriptor`] containing `ConditionClass` + `Provenance`.
    ///
    /// Returns `None` for `Always` (no condition needed) and `Never`
    /// (unconditionally blocked — no meaningful condition to describe).
    #[inline]
    pub fn requirement_descriptor(&self) -> Option<RequirementDescriptor> {
        match self {
            SolveSafety::Always => None,
            SolveSafety::IntrinsicCondition(class) => Some(RequirementDescriptor {
                class: *class,
                provenance: Provenance::Intrinsic,
            }),
            SolveSafety::NeedsCondition(class) => Some(RequirementDescriptor {
                class: *class,
                provenance: Provenance::Introduced,
            }),
            SolveSafety::Never => None,
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
    use crate::domain_facts::Provenance;

    #[test]
    fn test_always_safe_everywhere() {
        let safety = SolveSafety::Always;
        assert!(safety.safe_for_prepass());
        assert!(safety.safe_for_tactic(DomainMode::Strict));
        assert!(safety.safe_for_tactic(DomainMode::Generic));
        assert!(safety.safe_for_tactic(DomainMode::Assume));
    }

    #[test]
    fn test_intrinsic_condition_allowed_in_generic() {
        let safety = SolveSafety::IntrinsicCondition(ConditionClass::Analytic);
        assert!(
            !safety.safe_for_prepass(),
            "Intrinsic still blocked in prepass"
        );
        assert!(
            !safety.safe_for_tactic(DomainMode::Strict),
            "Strict requires proof"
        );
        assert!(
            safety.safe_for_tactic(DomainMode::Generic),
            "Generic inherits intrinsic"
        );
        assert!(
            safety.safe_for_tactic(DomainMode::Assume),
            "Assume allows all"
        );
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
        // Analytic only allowed in Assume (introduced, not intrinsic)
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

    // =====================================================================
    // RequirementDescriptor bridge tests
    // =====================================================================

    #[test]
    fn test_always_has_no_descriptor() {
        assert!(SolveSafety::Always.requirement_descriptor().is_none());
    }

    #[test]
    fn test_never_has_no_descriptor() {
        assert!(SolveSafety::Never.requirement_descriptor().is_none());
    }

    #[test]
    fn test_intrinsic_analytic_descriptor() {
        let desc = SolveSafety::IntrinsicCondition(ConditionClass::Analytic)
            .requirement_descriptor()
            .expect("IntrinsicCondition should produce a descriptor");
        assert_eq!(desc.class, ConditionClass::Analytic);
        assert_eq!(desc.provenance, Provenance::Intrinsic);
    }

    #[test]
    fn test_intrinsic_definability_descriptor() {
        let desc = SolveSafety::IntrinsicCondition(ConditionClass::Definability)
            .requirement_descriptor()
            .expect("IntrinsicCondition should produce a descriptor");
        assert_eq!(desc.class, ConditionClass::Definability);
        assert_eq!(desc.provenance, Provenance::Intrinsic);
    }

    #[test]
    fn test_needs_analytic_descriptor() {
        let desc = SolveSafety::NeedsCondition(ConditionClass::Analytic)
            .requirement_descriptor()
            .expect("NeedsCondition should produce a descriptor");
        assert_eq!(desc.class, ConditionClass::Analytic);
        assert_eq!(desc.provenance, Provenance::Introduced);
    }

    #[test]
    fn test_needs_definability_descriptor() {
        let desc = SolveSafety::NeedsCondition(ConditionClass::Definability)
            .requirement_descriptor()
            .expect("NeedsCondition should produce a descriptor");
        assert_eq!(desc.class, ConditionClass::Definability);
        assert_eq!(desc.provenance, Provenance::Introduced);
    }

    #[test]
    fn test_descriptor_distinguishes_intrinsic_from_introduced() {
        let intrinsic = SolveSafety::IntrinsicCondition(ConditionClass::Analytic)
            .requirement_descriptor()
            .unwrap();
        let introduced = SolveSafety::NeedsCondition(ConditionClass::Analytic)
            .requirement_descriptor()
            .unwrap();
        // Same class, different provenance
        assert_eq!(intrinsic.class, introduced.class);
        assert_ne!(intrinsic.provenance, introduced.provenance);
    }
}
