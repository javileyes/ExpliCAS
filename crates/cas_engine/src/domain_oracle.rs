//! Standard Domain Oracle implementation.
//!
//! This module provides [`StandardOracle`], the default implementation of
//! [`DomainOracle`] backed by the existing prover infrastructure
//! (`prove_nonzero`, `prove_positive`, `prove_nonnegative`).
//!
//! # Usage
//!
//! ```ignore
//! use cas_engine::domain_oracle::StandardOracle;
//! use cas_engine::domain_facts::{Predicate, DomainOracle};
//!
//! let oracle = StandardOracle::new(ctx, DomainMode::Generic, ValueDomain::RealOnly);
//! let decision = oracle.allows(&Predicate::NonZero(factor_id));
//! if !decision.allow {
//!     return None; // blocked
//! }
//! ```
//!
//! # Design
//!
//! `StandardOracle` wraps the existing prover functions without changing their
//! behavior. It's a **pure adapter** — the same provers, the same `DomainMode`
//! policy, but accessed through the unified `DomainOracle` trait.

use cas_ast::Context;

use crate::domain::{CancelDecision, DomainMode};
use crate::domain_facts::{decide, proof_to_strength, DomainOracle, FactStrength, Predicate};
use crate::semantics::ValueDomain;

// =============================================================================
// StandardOracle
// =============================================================================

/// Default oracle backed by existing prover infrastructure.
///
/// This oracle delegates to `prove_nonzero`, `prove_positive`, and
/// `prove_nonnegative` from `helpers::predicates`, then applies the
/// `DomainMode` policy via `domain_facts::decide()`.
pub struct StandardOracle<'a> {
    ctx: &'a Context,
    mode: DomainMode,
    value_domain: ValueDomain,
}

impl<'a> StandardOracle<'a> {
    /// Create a new oracle with the given context and semantic configuration.
    pub fn new(ctx: &'a Context, mode: DomainMode, value_domain: ValueDomain) -> Self {
        Self {
            ctx,
            mode,
            value_domain,
        }
    }

    /// Get the underlying `DomainMode`.
    #[inline]
    pub fn mode(&self) -> DomainMode {
        self.mode
    }

    /// Get the underlying `ValueDomain`.
    #[inline]
    pub fn value_domain(&self) -> ValueDomain {
        self.value_domain
    }
}

impl DomainOracle for StandardOracle<'_> {
    /// Query the strength of evidence for a predicate.
    ///
    /// Delegates to the appropriate prover:
    /// - `NonZero` → `prove_nonzero`
    /// - `Positive` → `prove_positive`
    /// - `NonNegative` → `prove_nonnegative`
    /// - `Defined` → always `Unknown` (no prover for general definedness)
    fn query(&self, pred: &Predicate) -> FactStrength {
        use crate::helpers::{prove_nonnegative, prove_nonzero, prove_positive};

        match pred {
            Predicate::NonZero(e) => proof_to_strength(prove_nonzero(self.ctx, *e)),
            Predicate::Positive(e) => {
                proof_to_strength(prove_positive(self.ctx, *e, self.value_domain))
            }
            Predicate::NonNegative(e) => {
                proof_to_strength(prove_nonnegative(self.ctx, *e, self.value_domain))
            }
            Predicate::Defined(_) => {
                // No general prover for definedness yet.
                // Conservative: Unknown.
                FactStrength::Unknown
            }
        }
    }

    /// Decide whether a transformation requiring this predicate is allowed.
    ///
    /// Combines `query()` result with the `DomainMode` policy.
    fn allows(&self, pred: &Predicate) -> CancelDecision {
        let strength = self.query(pred);
        decide(self.mode, pred, strength)
    }
}

// =============================================================================
// Convenience: Rich oracle with hint support
// =============================================================================

/// Rich version of oracle query that emits pedagogical hints for Strict mode.
///
/// This wraps `can_cancel_factor_with_hint` and `can_apply_analytic_with_hint`
/// into a single function using the unified predicate vocabulary.
///
/// # Arguments
///
/// * `ctx` - Expression context
/// * `mode` - Current DomainMode
/// * `pred` - The predicate to check
/// * `rule` - Name of the rule requesting the transformation
///
/// # Returns
///
/// A `CancelDecision` with optional `BlockedHint` for Strict mode.
pub fn oracle_allows_with_hint(
    ctx: &Context,
    mode: DomainMode,
    value_domain: ValueDomain,
    pred: &Predicate,
    rule: &'static str,
) -> CancelDecision {
    use crate::assumptions::AssumptionKey;
    use crate::helpers::{prove_nonnegative, prove_nonzero, prove_positive};

    let expr = pred.expr();

    match pred {
        Predicate::NonZero(_) => {
            let proof = prove_nonzero(ctx, expr);
            let key = AssumptionKey::nonzero_key(ctx, expr);
            crate::domain::can_cancel_factor_with_hint(mode, proof, key, expr, rule)
        }
        Predicate::Positive(_) => {
            let proof = prove_positive(ctx, expr, value_domain);
            let key = AssumptionKey::positive_key(ctx, expr);
            crate::domain::can_apply_analytic_with_hint(mode, proof, key, expr, rule)
        }
        Predicate::NonNegative(_) => {
            let proof = prove_nonnegative(ctx, expr, value_domain);
            let key = AssumptionKey::nonnegative_key(ctx, expr);
            crate::domain::can_apply_analytic_with_hint(mode, proof, key, expr, rule)
        }
        Predicate::Defined(_) => {
            // Defined uses Definability class, same as NonZero
            let key = AssumptionKey::Defined {
                expr_fingerprint: crate::assumptions::expr_fingerprint(ctx, expr),
            };
            // No dedicated prover — treat as Unknown
            crate::domain::can_cancel_factor_with_hint(
                mode,
                crate::domain::Proof::Unknown,
                key,
                expr,
                rule,
            )
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::DomainMode;
    use crate::domain_facts::FactStrength;
    use crate::semantics::ValueDomain;

    #[test]
    fn test_oracle_nonzero_constant() {
        let mut ctx = Context::new();
        let two = ctx.num(2);

        let oracle = StandardOracle::new(&ctx, DomainMode::Strict, ValueDomain::RealOnly);

        // 2 is provably non-zero
        assert_eq!(oracle.query(&Predicate::NonZero(two)), FactStrength::Proven);
        assert!(oracle.allows(&Predicate::NonZero(two)).allow);
    }

    #[test]
    fn test_oracle_nonzero_zero() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);

        let oracle = StandardOracle::new(&ctx, DomainMode::Assume, ValueDomain::RealOnly);

        // 0 is provably zero (disproven nonzero)
        assert_eq!(
            oracle.query(&Predicate::NonZero(zero)),
            FactStrength::Disproven
        );
        // Even in Assume mode, disproven is denied
        assert!(!oracle.allows(&Predicate::NonZero(zero)).allow);
    }

    #[test]
    fn test_oracle_nonzero_variable_strict() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let oracle = StandardOracle::new(&ctx, DomainMode::Strict, ValueDomain::RealOnly);

        // x is unknown nonzero
        assert_eq!(oracle.query(&Predicate::NonZero(x)), FactStrength::Unknown);
        // Strict blocks unknown
        assert!(!oracle.allows(&Predicate::NonZero(x)).allow);
    }

    #[test]
    fn test_oracle_nonzero_variable_generic() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let oracle = StandardOracle::new(&ctx, DomainMode::Generic, ValueDomain::RealOnly);

        // x is unknown nonzero
        assert_eq!(oracle.query(&Predicate::NonZero(x)), FactStrength::Unknown);
        // Generic allows NonZero (Definability)
        assert!(oracle.allows(&Predicate::NonZero(x)).allow);
    }

    #[test]
    fn test_oracle_positive_constant() {
        let mut ctx = Context::new();
        let three = ctx.num(3);

        let oracle = StandardOracle::new(&ctx, DomainMode::Strict, ValueDomain::RealOnly);

        // 3 is provably positive
        assert_eq!(
            oracle.query(&Predicate::Positive(three)),
            FactStrength::Proven
        );
        assert!(oracle.allows(&Predicate::Positive(three)).allow);
    }

    #[test]
    fn test_oracle_positive_variable_generic() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let oracle = StandardOracle::new(&ctx, DomainMode::Generic, ValueDomain::RealOnly);

        // x is unknown positive
        assert_eq!(oracle.query(&Predicate::Positive(x)), FactStrength::Unknown);
        // Generic blocks Positive (Analytic)
        assert!(!oracle.allows(&Predicate::Positive(x)).allow);
    }

    #[test]
    fn test_oracle_positive_variable_assume() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let oracle = StandardOracle::new(&ctx, DomainMode::Assume, ValueDomain::RealOnly);

        // x is unknown positive
        assert_eq!(oracle.query(&Predicate::Positive(x)), FactStrength::Unknown);
        // Assume allows everything
        assert!(oracle.allows(&Predicate::Positive(x)).allow);
    }

    #[test]
    fn test_oracle_nonnegative_zero() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);

        let oracle = StandardOracle::new(&ctx, DomainMode::Strict, ValueDomain::RealOnly);

        // 0 is provably non-negative
        assert_eq!(
            oracle.query(&Predicate::NonNegative(zero)),
            FactStrength::Proven
        );
        assert!(oracle.allows(&Predicate::NonNegative(zero)).allow);
    }

    #[test]
    fn test_oracle_defined_always_unknown() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let oracle = StandardOracle::new(&ctx, DomainMode::Assume, ValueDomain::RealOnly);

        // Defined is always Unknown (no dedicated prover)
        assert_eq!(oracle.query(&Predicate::Defined(x)), FactStrength::Unknown);
    }

    #[test]
    fn test_oracle_parity_with_can_cancel_factor() {
        // Exhaustive parity test: oracle.allows() must produce the same
        // allow/deny as can_cancel_factor() for all (mode, proof) combos
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let zero = ctx.num(0);
        let x = ctx.var("x");

        for mode in [DomainMode::Strict, DomainMode::Generic, DomainMode::Assume] {
            let oracle = StandardOracle::new(&ctx, mode, ValueDomain::RealOnly);

            // Proven nonzero (constant 2)
            let oracle_decision = oracle.allows(&Predicate::NonZero(two));
            let proof = crate::helpers::prove_nonzero(&ctx, two);
            let legacy_decision = crate::domain::can_cancel_factor(mode, proof);
            assert_eq!(
                oracle_decision.allow, legacy_decision.allow,
                "Mismatch for mode={:?}, expr=2",
                mode
            );

            // Disproven nonzero (constant 0)
            let oracle_decision = oracle.allows(&Predicate::NonZero(zero));
            let proof = crate::helpers::prove_nonzero(&ctx, zero);
            let legacy_decision = crate::domain::can_cancel_factor(mode, proof);
            assert_eq!(
                oracle_decision.allow, legacy_decision.allow,
                "Mismatch for mode={:?}, expr=0",
                mode
            );

            // Unknown nonzero (variable x)
            let oracle_decision = oracle.allows(&Predicate::NonZero(x));
            let proof = crate::helpers::prove_nonzero(&ctx, x);
            let legacy_decision = crate::domain::can_cancel_factor(mode, proof);
            assert_eq!(
                oracle_decision.allow, legacy_decision.allow,
                "Mismatch for mode={:?}, expr=x",
                mode
            );
        }
    }
}
