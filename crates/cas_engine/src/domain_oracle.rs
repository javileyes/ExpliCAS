//! Standard Domain Oracle implementation.
//!
//! This module provides [`StandardOracle`], the default implementation of
//! [`DomainOracle`] backed by the existing prover infrastructure
//! (`prove_nonzero`, `prove_positive`, `prove_nonnegative`).
//!
//! # Usage
//!
//! ```ignore
//! use cas_engine::{DomainOracle, Predicate, StandardOracle};
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
use cas_solver_core::domain_facts_model::{FactStrength, Predicate};
use cas_solver_core::domain_oracle_model::DomainOracle;

use crate::semantics::ValueDomain;
use crate::{CancelDecision, DomainMode};

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
    type Decision = CancelDecision;

    /// Query the strength of evidence for a predicate.
    ///
    /// Delegates to the appropriate prover:
    /// - `NonZero` → `prove_nonzero`
    /// - `Positive` → `prove_positive`
    /// - `NonNegative` → `prove_nonnegative`
    /// - `Defined` → always `Unknown` (no prover for general definedness)
    fn query(&self, pred: &Predicate) -> FactStrength {
        use crate::helpers::{prove_nonnegative, prove_nonzero, prove_positive};

        cas_solver_core::domain_oracle_model::query_predicate_strength_with_provers(
            pred,
            |expr| prove_nonzero(self.ctx, expr),
            |expr| prove_positive(self.ctx, expr, self.value_domain),
            |expr| prove_nonnegative(self.ctx, expr, self.value_domain),
        )
    }

    /// Decide whether a transformation requiring this predicate is allowed.
    ///
    /// Combines `query()` result with the `DomainMode` policy.
    fn allows(&self, pred: &Predicate) -> CancelDecision {
        use crate::helpers::{prove_nonnegative, prove_nonzero, prove_positive};
        cas_solver_core::domain_oracle_model::allows_with_provers(
            self.mode,
            pred,
            |expr| prove_nonzero(self.ctx, expr),
            |expr| prove_positive(self.ctx, expr, self.value_domain),
            |expr| prove_nonnegative(self.ctx, expr, self.value_domain),
        )
    }
}

// =============================================================================
// Convenience: Rich oracle with hint support
// =============================================================================

/// Rich version of oracle query that emits pedagogical hints for Strict mode.
///
/// This routes each predicate through the canonical core domain gates
/// using the unified predicate vocabulary.
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
    use crate::helpers::{prove_nonnegative, prove_nonzero, prove_positive};
    cas_solver_core::domain_oracle_model::allows_with_hint_using_provers(
        ctx,
        mode,
        pred,
        rule,
        |expr| prove_nonzero(ctx, expr),
        |expr| prove_positive(ctx, expr, value_domain),
        |expr| prove_nonnegative(ctx, expr, value_domain),
    )
}
