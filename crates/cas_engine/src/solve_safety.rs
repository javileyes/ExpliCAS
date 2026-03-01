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
use cas_solver_core::solve_safety_policy as core_safety;
pub use cas_solver_core::solve_safety_policy::SimplifyPurpose;

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
        core_safety::safe_for_prepass(to_core_solve_safety(*self))
    }

    /// Returns true if this rule is safe for solver tactic given domain mode.
    /// - `Always`: always safe
    /// - `NeedsCondition(class)`: safe if mode allows unproven conditions of that class
    /// - `Never`: never safe
    #[inline]
    pub fn safe_for_tactic(&self, domain_mode: crate::domain::DomainMode) -> bool {
        core_safety::safe_for_tactic(
            to_core_solve_safety(*self),
            to_core_domain_mode(domain_mode),
        )
    }

    /// Bridge to domain vocabulary: maps this safety classification to a
    /// [`RequirementDescriptor`] containing `ConditionClass` + `Provenance`.
    ///
    /// Returns `None` for `Always` (no condition needed) and `Never`
    /// (unconditionally blocked — no meaningful condition to describe).
    #[inline]
    pub fn requirement_descriptor(&self) -> Option<RequirementDescriptor> {
        core_safety::requirement_descriptor(to_core_solve_safety(*self)).map(|desc| {
            RequirementDescriptor {
                class: from_core_condition_class(desc.class),
                provenance: from_core_provenance(desc.provenance),
            }
        })
    }
}

fn to_core_condition_class(class: ConditionClass) -> core_safety::ConditionClassKind {
    match class {
        ConditionClass::Definability => core_safety::ConditionClassKind::Definability,
        ConditionClass::Analytic => core_safety::ConditionClassKind::Analytic,
    }
}

fn from_core_condition_class(class: core_safety::ConditionClassKind) -> ConditionClass {
    match class {
        core_safety::ConditionClassKind::Definability => ConditionClass::Definability,
        core_safety::ConditionClassKind::Analytic => ConditionClass::Analytic,
    }
}

fn from_core_provenance(provenance: core_safety::ProvenanceKind) -> Provenance {
    match provenance {
        core_safety::ProvenanceKind::Intrinsic => Provenance::Intrinsic,
        core_safety::ProvenanceKind::Introduced => Provenance::Introduced,
    }
}

fn to_core_solve_safety(safety: SolveSafety) -> core_safety::SolveSafetyKind {
    match safety {
        SolveSafety::Always => core_safety::SolveSafetyKind::Always,
        SolveSafety::IntrinsicCondition(class) => {
            core_safety::SolveSafetyKind::IntrinsicCondition(to_core_condition_class(class))
        }
        SolveSafety::NeedsCondition(class) => {
            core_safety::SolveSafetyKind::NeedsCondition(to_core_condition_class(class))
        }
        SolveSafety::Never => core_safety::SolveSafetyKind::Never,
    }
}

fn to_core_domain_mode(
    domain_mode: crate::domain::DomainMode,
) -> cas_solver_core::log_domain::DomainModeKind {
    cas_solver_core::log_domain::domain_mode_kind_from_flags(
        matches!(domain_mode, crate::domain::DomainMode::Assume),
        matches!(domain_mode, crate::domain::DomainMode::Strict),
    )
}
