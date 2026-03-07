//! Standard Domain Oracle implementation for solver facade.
//!
//! This mirrors engine behavior while keeping oracle surface owned by
//! `cas_solver` during migration.

use cas_ast::Context;
use cas_solver_core::domain_cancel_decision::CancelDecision;
use cas_solver_core::domain_facts_model::{FactStrength, Predicate};
use cas_solver_core::domain_oracle_model::DomainOracle;
use cas_solver_core::standard_oracle as core_standard_oracle;

use crate::{DomainMode, ValueDomain};

/// Default oracle backed by local predicate proof runtime.
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
                cas_solver_core::proof_runtime_bound_runtime::prove_nonzero_with_runtime_proof_simplifier::<crate::Simplifier>,
                cas_solver_core::proof_runtime_bound_runtime::prove_positive_with_runtime_proof_simplifier::<crate::Simplifier>,
                cas_solver_core::proof_runtime_bound_runtime::prove_nonnegative_with_runtime_proof_simplifier::<crate::Simplifier>,
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

    fn query(&self, pred: &Predicate) -> FactStrength {
        self.inner.query(pred)
    }

    fn allows(&self, pred: &Predicate) -> CancelDecision {
        self.inner.allows(pred)
    }
}

/// Rich oracle query that emits pedagogical hints for strict domain mode.
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
        cas_solver_core::proof_runtime_bound_runtime::prove_nonzero_with_runtime_proof_simplifier::<crate::Simplifier>,
        cas_solver_core::proof_runtime_bound_runtime::prove_positive_with_runtime_proof_simplifier::<crate::Simplifier>,
        cas_solver_core::proof_runtime_bound_runtime::prove_nonnegative_with_runtime_proof_simplifier::<crate::Simplifier>,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_solver_core::domain_facts_model::Predicate;

    #[test]
    fn strict_unknown_nonzero_is_blocked() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let oracle = StandardOracle::new(&ctx, DomainMode::Strict, ValueDomain::RealOnly);
        let decision = oracle.allows(&Predicate::NonZero(x));
        assert!(!decision.allow);
    }

    #[test]
    fn assume_unknown_nonzero_is_allowed() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let oracle = StandardOracle::new(&ctx, DomainMode::Assume, ValueDomain::RealOnly);
        let decision = oracle.allows(&Predicate::NonZero(x));
        assert!(decision.allow);
    }

    #[test]
    fn strict_proven_nonzero_is_allowed() {
        let mut ctx = cas_ast::Context::new();
        let two = ctx.num(2);
        let oracle = StandardOracle::new(&ctx, DomainMode::Strict, ValueDomain::RealOnly);
        let decision = oracle.allows(&Predicate::NonZero(two));
        assert!(decision.allow);
        assert!(decision.assumption.is_none());
    }
}
