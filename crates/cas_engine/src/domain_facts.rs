//! Unified Domain Facts — canonical vocabulary for domain predicates.
//!
//! This module introduces a unified model for the six domain-related
//! abstractions that were previously spread across `domain.rs`,
//! `assumptions.rs`, `solve_safety.rs`, and `implicit_domain/`.
//!
//! # Core Types
//!
//! - [`Predicate`]: What is being asserted about an expression (NonZero, Positive, etc.)
//! - [`Provenance`]: Where the fact came from (Intrinsic, Proven, Assumed, Introduced)
//! - [`FactStrength`]: How strong the evidence is (Proven, Unknown, Disproven)
//! - [`DomainFact`]: A complete domain assertion combining all three
//!
//! # Bridge Functions
//!
//! - [`proof_to_strength`]: Convert legacy `Proof` → `FactStrength`
//! - [`strength_to_proof`]: Convert `FactStrength` → legacy `Proof`
//! - [`predicate_condition_class`]: Map `Predicate` → `ConditionClass`
//!
//! # Design
//!
//! This is a **shim layer**: existing types (`DomainMode`, `Proof`, etc.) continue
//! to work. The oracle wraps them with a unified interface. Old code compiles at
//! every step, and consumers can migrate one file at a time.

use cas_ast::ExprId;

// =============================================================================
// Predicate — What is being asserted
// =============================================================================

/// A domain predicate about an expression.
///
/// This unifies `AssumptionKey` variants (NonZero, Positive, etc.) and
/// `ImplicitCondition` variants into a single vocabulary.
///
/// # Mapping from legacy types
///
/// | Predicate | AssumptionKey | ImplicitCondition |
/// |-----------|--------------|-------------------|
/// | `NonZero(e)` | `NonZero { fp }` | `NonZero(e)` |
/// | `Positive(e)` | `Positive { fp }` | `Positive(e)` |
/// | `NonNegative(e)` | `NonNegative { fp }` | `NonNegative(e)` |
/// | `Defined(e)` | `Defined { fp }` | — |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Predicate {
    /// Expression ≠ 0 (Definability class)
    NonZero(ExprId),
    /// Expression > 0 (Analytic class)
    Positive(ExprId),
    /// Expression ≥ 0 (Analytic class)
    NonNegative(ExprId),
    /// Expression is defined at this point (Definability class)
    Defined(ExprId),
}

impl Predicate {
    /// Get the `ConditionClass` for this predicate.
    ///
    /// This replaces the need to call `AssumptionKey::class()` separately.
    #[inline]
    pub fn condition_class(&self) -> crate::assumptions::ConditionClass {
        predicate_condition_class(self)
    }

    /// Get the expression this predicate is about.
    #[inline]
    pub fn expr(&self) -> ExprId {
        match self {
            Predicate::NonZero(e)
            | Predicate::Positive(e)
            | Predicate::NonNegative(e)
            | Predicate::Defined(e) => *e,
        }
    }

    /// Human-readable description of the predicate.
    pub fn describe(&self) -> &'static str {
        match self {
            Predicate::NonZero(_) => "≠ 0",
            Predicate::Positive(_) => "> 0",
            Predicate::NonNegative(_) => "≥ 0",
            Predicate::Defined(_) => "is defined",
        }
    }

    /// Short label for logging/debugging.
    pub fn label(&self) -> &'static str {
        match self {
            Predicate::NonZero(_) => "nonzero",
            Predicate::Positive(_) => "positive",
            Predicate::NonNegative(_) => "nonnegative",
            Predicate::Defined(_) => "defined",
        }
    }
}

// =============================================================================
// Provenance — Where the fact came from
// =============================================================================

/// Origin of a domain fact.
///
/// This unifies `AssumptionKind` (display-oriented) with the implicit/explicit
/// distinction from `ImplicitDomain` and `SolveSafety::IntrinsicCondition`.
///
/// # Mapping from legacy types
///
/// | Provenance | AssumptionKind | SolveSafety |
/// |-----------|---------------|-------------|
/// | `Intrinsic` | `DerivedFromRequires` | `IntrinsicCondition` |
/// | `Proven` | — (not an assumption) | `Always` |
/// | `Assumed` | `HeuristicAssumption` | — |
/// | `Introduced` | `RequiresIntroduced` | `NeedsCondition` |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provenance {
    /// Inherited from input AST structure.
    /// Example: `1/x` intrinsically requires `x ≠ 0`.
    /// Not a new assumption — just documenting what's already implied.
    Intrinsic,

    /// Structurally proven by the engine.
    /// Example: `2 ≠ 0` is provably true for the literal `2`.
    Proven,

    /// Assumed by user or mode policy.
    /// Example: `x ≠ 0` assumed in Generic mode for `x/x → 1`.
    Assumed,

    /// Introduced by a rule as a new side condition.
    /// Example: `a > 0` introduced by `log(a·b) → log(a) + log(b)`.
    Introduced,
}

// =============================================================================
// FactStrength — How strong is the evidence
// =============================================================================

/// Strength of evidence for a domain fact.
///
/// This is a direct mapping of the legacy `Proof` enum, providing a
/// vocabulary that's independent of the prover implementation.
///
/// # Mapping from legacy `Proof`
///
/// | FactStrength | Proof |
/// |-------------|-------|
/// | `Proven` | `Proven` or `ProvenImplicit` |
/// | `Unknown` | `Unknown` |
/// | `Disproven` | `Disproven` |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FactStrength {
    /// Property is provably true.
    Proven,
    /// Property status is unknown (conservative).
    Unknown,
    /// Property is provably false.
    Disproven,
}

impl FactStrength {
    /// Returns true if the property is proven.
    #[inline]
    pub fn is_proven(self) -> bool {
        matches!(self, FactStrength::Proven)
    }

    /// Returns true if the property is unknown.
    #[inline]
    pub fn is_unknown(self) -> bool {
        matches!(self, FactStrength::Unknown)
    }

    /// Returns true if the property is disproven.
    #[inline]
    pub fn is_disproven(self) -> bool {
        matches!(self, FactStrength::Disproven)
    }
}

// =============================================================================
// DomainFact — Complete domain assertion
// =============================================================================

/// A complete domain assertion combining predicate, provenance, and strength.
///
/// # Example
///
/// ```ignore
/// // "x ≠ 0 is structurally proven"
/// DomainFact {
///     predicate: Predicate::NonZero(x_id),
///     provenance: Provenance::Proven,
///     strength: FactStrength::Proven,
/// }
///
/// // "x > 0 is assumed by Generic mode"
/// DomainFact {
///     predicate: Predicate::Positive(x_id),
///     provenance: Provenance::Assumed,
///     strength: FactStrength::Unknown,
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DomainFact {
    /// What is being asserted.
    pub predicate: Predicate,
    /// Where the fact came from.
    pub provenance: Provenance,
    /// How strong the evidence is.
    pub strength: FactStrength,
}

// =============================================================================
// Bridge Functions — Legacy ↔ Unified
// =============================================================================

/// Convert a legacy `Proof` value to `FactStrength`.
///
/// `Proof::Proven` and `Proof::ProvenImplicit` both map to `FactStrength::Proven`
/// because the oracle doesn't distinguish how the proof was obtained.
#[inline]
pub fn proof_to_strength(proof: crate::domain::Proof) -> FactStrength {
    use crate::domain::Proof;
    match proof {
        Proof::Proven | Proof::ProvenImplicit => FactStrength::Proven,
        Proof::Unknown => FactStrength::Unknown,
        Proof::Disproven => FactStrength::Disproven,
    }
}

/// Convert `FactStrength` back to a legacy `Proof`.
///
/// `FactStrength::Proven` maps to `Proof::Proven` (not `ProvenImplicit`).
#[inline]
pub fn strength_to_proof(strength: FactStrength) -> crate::domain::Proof {
    use crate::domain::Proof;
    match strength {
        FactStrength::Proven => Proof::Proven,
        FactStrength::Unknown => Proof::Unknown,
        FactStrength::Disproven => Proof::Disproven,
    }
}

/// Map a `Predicate` to its `ConditionClass`.
///
/// This replaces the per-variant mapping in `AssumptionKey::class()`.
///
/// - `NonZero`, `Defined` → `ConditionClass::Definability` (small holes)
/// - `Positive`, `NonNegative` → `ConditionClass::Analytic` (big restrictions)
#[inline]
pub fn predicate_condition_class(pred: &Predicate) -> crate::assumptions::ConditionClass {
    use crate::assumptions::ConditionClass;
    match pred {
        Predicate::NonZero(_) | Predicate::Defined(_) => ConditionClass::Definability,
        Predicate::Positive(_) | Predicate::NonNegative(_) => ConditionClass::Analytic,
    }
}

/// Check whether a `DomainMode` allows an unproven predicate of this type.
///
/// This wraps `DomainMode::allows_unproven(predicate.condition_class())` in
/// predicate-centric vocabulary.
#[inline]
pub fn mode_allows_predicate(mode: crate::domain::DomainMode, pred: &Predicate) -> bool {
    mode.allows_unproven(pred.condition_class())
}

/// Unified domain decision: given a mode, a predicate, and evidence strength,
/// decide whether to allow the transformation.
///
/// This is the core logic that `can_cancel_factor` and `can_apply_analytic`
/// both implement separately. This function unifies them.
///
/// # Returns
///
/// - `allow: true` with no assumption for proven facts
/// - `allow: false` for disproven facts
/// - For unknown facts: depends on `mode.allows_unproven(predicate.condition_class())`
pub fn decide(
    mode: crate::domain::DomainMode,
    pred: &Predicate,
    strength: FactStrength,
) -> crate::domain::CancelDecision {
    decide_by_class(mode, pred.condition_class(), strength, pred.describe())
}

/// Low-level decision function that takes `ConditionClass` directly.
///
/// This is used by legacy gate functions (`can_cancel_factor`, `can_apply_analytic`)
/// that already know the condition class and have a pre-computed proof, without
/// needing an `ExprId` to construct a full `Predicate`.
///
/// New code should prefer `decide()` or `DomainOracle::allows()` which work
/// with the full `Predicate` type.
pub fn decide_by_class(
    mode: crate::domain::DomainMode,
    class: crate::assumptions::ConditionClass,
    strength: FactStrength,
    assumption_label: &'static str,
) -> crate::domain::CancelDecision {
    match strength {
        FactStrength::Proven => crate::domain::CancelDecision::allow(),
        FactStrength::Disproven => crate::domain::CancelDecision::deny(),
        FactStrength::Unknown => {
            if mode.allows_unproven(class) {
                crate::domain::CancelDecision::allow_with_assumption(assumption_label)
            } else {
                crate::domain::CancelDecision::deny()
            }
        }
    }
}

// =============================================================================
// DomainOracle Trait
// =============================================================================

/// Unified interface for querying domain facts.
///
/// This trait abstracts the current pattern where rules call
/// `prove_nonzero(ctx, expr)` + `can_cancel_factor(mode, proof)` into
/// a single `oracle.allows(&Predicate::NonZero(expr))` call.
///
/// # Migration Path
///
/// Rules can migrate from:
/// ```ignore
/// let mode = parent_ctx.domain_mode();
/// let proof = prove_nonzero(ctx, factor);
/// let decision = can_cancel_factor(mode, proof);
/// ```
/// to:
/// ```ignore
/// let decision = oracle.allows(&Predicate::NonZero(factor));
/// ```
pub trait DomainOracle {
    /// Query the strength of evidence for a predicate.
    fn query(&self, pred: &Predicate) -> FactStrength;

    /// Decide whether a transformation requiring this predicate is allowed.
    ///
    /// Combines `query()` with the current `DomainMode` policy.
    fn allows(&self, pred: &Predicate) -> crate::domain::CancelDecision;
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assumptions::ConditionClass;
    use crate::domain::{DomainMode, Proof};

    #[test]
    fn test_proof_to_strength_mapping() {
        assert_eq!(proof_to_strength(Proof::Proven), FactStrength::Proven);
        assert_eq!(
            proof_to_strength(Proof::ProvenImplicit),
            FactStrength::Proven
        );
        assert_eq!(proof_to_strength(Proof::Unknown), FactStrength::Unknown);
        assert_eq!(proof_to_strength(Proof::Disproven), FactStrength::Disproven);
    }

    #[test]
    fn test_strength_to_proof_roundtrip() {
        // Proven → Proof::Proven (not ProvenImplicit)
        assert_eq!(strength_to_proof(FactStrength::Proven), Proof::Proven);
        assert_eq!(strength_to_proof(FactStrength::Unknown), Proof::Unknown);
        assert_eq!(strength_to_proof(FactStrength::Disproven), Proof::Disproven);
    }

    #[test]
    fn test_predicate_condition_class() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");

        assert_eq!(
            Predicate::NonZero(x).condition_class(),
            ConditionClass::Definability
        );
        assert_eq!(
            Predicate::Defined(x).condition_class(),
            ConditionClass::Definability
        );
        assert_eq!(
            Predicate::Positive(x).condition_class(),
            ConditionClass::Analytic
        );
        assert_eq!(
            Predicate::NonNegative(x).condition_class(),
            ConditionClass::Analytic
        );
    }

    #[test]
    fn test_mode_allows_predicate() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");

        let nonzero = Predicate::NonZero(x);
        let positive = Predicate::Positive(x);

        // Strict: blocks everything
        assert!(!mode_allows_predicate(DomainMode::Strict, &nonzero));
        assert!(!mode_allows_predicate(DomainMode::Strict, &positive));

        // Generic: allows Definability, blocks Analytic
        assert!(mode_allows_predicate(DomainMode::Generic, &nonzero));
        assert!(!mode_allows_predicate(DomainMode::Generic, &positive));

        // Assume: allows everything
        assert!(mode_allows_predicate(DomainMode::Assume, &nonzero));
        assert!(mode_allows_predicate(DomainMode::Assume, &positive));
    }

    #[test]
    fn test_decide_proven() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let pred = Predicate::NonZero(x);

        // Proven facts are always allowed, regardless of mode
        let decision = decide(DomainMode::Strict, &pred, FactStrength::Proven);
        assert!(decision.allow);
    }

    #[test]
    fn test_decide_disproven() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let pred = Predicate::NonZero(x);

        // Disproven facts are never allowed, regardless of mode
        let decision = decide(DomainMode::Assume, &pred, FactStrength::Disproven);
        assert!(!decision.allow);
    }

    #[test]
    fn test_decide_unknown_nonzero() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let pred = Predicate::NonZero(x);

        // NonZero is Definability: blocked in Strict, allowed in Generic/Assume
        assert!(!decide(DomainMode::Strict, &pred, FactStrength::Unknown).allow);
        assert!(decide(DomainMode::Generic, &pred, FactStrength::Unknown).allow);
        assert!(decide(DomainMode::Assume, &pred, FactStrength::Unknown).allow);
    }

    #[test]
    fn test_decide_unknown_positive() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let pred = Predicate::Positive(x);

        // Positive is Analytic: blocked in Strict+Generic, allowed only in Assume
        assert!(!decide(DomainMode::Strict, &pred, FactStrength::Unknown).allow);
        assert!(!decide(DomainMode::Generic, &pred, FactStrength::Unknown).allow);
        assert!(decide(DomainMode::Assume, &pred, FactStrength::Unknown).allow);
    }

    #[test]
    fn test_decide_matches_can_cancel_factor() {
        // Verify that decide() produces the same result as can_cancel_factor()
        // for the NonZero predicate (Definability class)
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let pred = Predicate::NonZero(x);

        for mode in [DomainMode::Strict, DomainMode::Generic, DomainMode::Assume] {
            for proof in [
                Proof::Proven,
                Proof::ProvenImplicit,
                Proof::Unknown,
                Proof::Disproven,
            ] {
                let strength = proof_to_strength(proof);
                let new_decision = decide(mode, &pred, strength);
                let old_decision = crate::domain::can_cancel_factor(mode, proof);
                assert_eq!(
                    new_decision.allow, old_decision.allow,
                    "Mismatch for mode={:?}, proof={:?}: new={}, old={}",
                    mode, proof, new_decision.allow, old_decision.allow,
                );
            }
        }
    }

    #[test]
    fn test_fact_strength_helpers() {
        assert!(FactStrength::Proven.is_proven());
        assert!(!FactStrength::Proven.is_unknown());
        assert!(!FactStrength::Proven.is_disproven());

        assert!(!FactStrength::Unknown.is_proven());
        assert!(FactStrength::Unknown.is_unknown());

        assert!(!FactStrength::Disproven.is_proven());
        assert!(FactStrength::Disproven.is_disproven());
    }

    #[test]
    fn test_predicate_describe() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");

        assert_eq!(Predicate::NonZero(x).describe(), "≠ 0");
        assert_eq!(Predicate::Positive(x).describe(), "> 0");
        assert_eq!(Predicate::NonNegative(x).describe(), "≥ 0");
        assert_eq!(Predicate::Defined(x).describe(), "is defined");
    }
}
