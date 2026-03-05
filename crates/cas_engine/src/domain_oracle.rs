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
use cas_solver_core::standard_oracle as core_standard_oracle;

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
    inner: core_standard_oracle::StandardOracle<'a>,
}

impl<'a> StandardOracle<'a> {
    /// Create a new oracle with the given context and semantic configuration.
    pub fn new(ctx: &'a Context, mode: DomainMode, value_domain: ValueDomain) -> Self {
        Self {
            inner: core_standard_oracle::StandardOracle::new(
                ctx,
                mode,
                value_domain,
                crate::helpers::prove_nonzero,
                crate::helpers::prove_positive,
                crate::helpers::prove_nonnegative,
            ),
        }
    }

    /// Get the underlying `DomainMode`.
    #[inline]
    pub fn mode(&self) -> DomainMode {
        self.inner.mode()
    }

    /// Get the underlying `ValueDomain`.
    #[inline]
    pub fn value_domain(&self) -> ValueDomain {
        self.inner.value_domain()
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
        self.inner.query(pred)
    }

    /// Decide whether a transformation requiring this predicate is allowed.
    ///
    /// Combines `query()` result with the `DomainMode` policy.
    fn allows(&self, pred: &Predicate) -> CancelDecision {
        self.inner.allows(pred)
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
    core_standard_oracle::oracle_allows_with_hint(
        ctx,
        mode,
        value_domain,
        pred,
        rule,
        crate::helpers::prove_nonzero,
        crate::helpers::prove_positive,
        crate::helpers::prove_nonnegative,
    )
}
