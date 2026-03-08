use cas_ast::{Context, ExprId};

use crate::domain_cancel_decision::CancelDecision;
use crate::domain_facts_model::{FactStrength, Predicate};
use crate::domain_mode::DomainMode;
use crate::domain_oracle_model::DomainOracle;
use crate::domain_proof::Proof;
use crate::value_domain::ValueDomain;

/// Prover callback for `expr != 0`.
pub type ProveNonZeroFn = fn(&Context, ExprId) -> Proof;
/// Prover callback for `expr > 0`.
pub type ProvePositiveFn = fn(&Context, ExprId, ValueDomain) -> Proof;
/// Prover callback for `expr >= 0`.
pub type ProveNonNegativeFn = fn(&Context, ExprId, ValueDomain) -> Proof;

/// Standard domain oracle powered by injected prover function pointers.
///
/// This keeps oracle mechanics centralized in `cas_solver_core` while runtime
/// crates (`cas_engine`, `cas_solver`) provide their concrete prover callbacks.
pub struct StandardOracle<'a> {
    ctx: &'a Context,
    mode: DomainMode,
    value_domain: ValueDomain,
    prove_nonzero: ProveNonZeroFn,
    prove_positive: ProvePositiveFn,
    prove_nonnegative: ProveNonNegativeFn,
}

impl<'a> StandardOracle<'a> {
    /// Create a new prover-backed standard oracle.
    pub fn new(
        ctx: &'a Context,
        mode: DomainMode,
        value_domain: ValueDomain,
        prove_nonzero: ProveNonZeroFn,
        prove_positive: ProvePositiveFn,
        prove_nonnegative: ProveNonNegativeFn,
    ) -> Self {
        Self {
            ctx,
            mode,
            value_domain,
            prove_nonzero,
            prove_positive,
            prove_nonnegative,
        }
    }

    /// Get the configured domain mode.
    #[inline]
    pub fn mode(&self) -> DomainMode {
        self.mode
    }

    /// Get the configured value domain.
    #[inline]
    pub fn value_domain(&self) -> ValueDomain {
        self.value_domain
    }
}

impl DomainOracle for StandardOracle<'_> {
    type Decision = CancelDecision;

    fn query(&self, pred: &Predicate) -> FactStrength {
        crate::domain_oracle_model::query_predicate_strength_with_provers(
            pred,
            |expr| (self.prove_nonzero)(self.ctx, expr),
            |expr| (self.prove_positive)(self.ctx, expr, self.value_domain),
            |expr| (self.prove_nonnegative)(self.ctx, expr, self.value_domain),
        )
    }

    fn allows(&self, pred: &Predicate) -> CancelDecision {
        crate::domain_oracle_model::allows_with_provers(
            self.mode,
            pred,
            |expr| (self.prove_nonzero)(self.ctx, expr),
            |expr| (self.prove_positive)(self.ctx, expr, self.value_domain),
            |expr| (self.prove_nonnegative)(self.ctx, expr, self.value_domain),
        )
    }
}

/// Hint-aware `allows()` using injected prover callbacks.
#[allow(clippy::too_many_arguments)]
pub fn oracle_allows_with_hint(
    ctx: &Context,
    mode: DomainMode,
    value_domain: ValueDomain,
    pred: &Predicate,
    rule: &'static str,
    prove_nonzero: ProveNonZeroFn,
    prove_positive: ProvePositiveFn,
    prove_nonnegative: ProveNonNegativeFn,
) -> CancelDecision {
    crate::domain_oracle_model::allows_with_hint_using_provers(
        ctx,
        mode,
        pred,
        rule,
        |expr| prove_nonzero(ctx, expr),
        |expr| prove_positive(ctx, expr, value_domain),
        |expr| prove_nonnegative(ctx, expr, value_domain),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prove_nonzero_stub(_ctx: &Context, _expr: ExprId) -> Proof {
        Proof::Unknown
    }

    fn prove_positive_stub(_ctx: &Context, _expr: ExprId, _value_domain: ValueDomain) -> Proof {
        Proof::Unknown
    }

    fn prove_nonnegative_stub(_ctx: &Context, _expr: ExprId, _value_domain: ValueDomain) -> Proof {
        Proof::Unknown
    }

    #[test]
    fn oracle_accessors_return_configured_axes() {
        let ctx = Context::new();
        let oracle = StandardOracle::new(
            &ctx,
            DomainMode::Assume,
            ValueDomain::ComplexEnabled,
            prove_nonzero_stub,
            prove_positive_stub,
            prove_nonnegative_stub,
        );
        assert_eq!(oracle.mode(), DomainMode::Assume);
        assert_eq!(oracle.value_domain(), ValueDomain::ComplexEnabled);
    }

    #[test]
    fn strict_unknown_nonzero_is_blocked() {
        let mut ctx = Context::new();
        let x = ctx.var("x");
        let oracle = StandardOracle::new(
            &ctx,
            DomainMode::Strict,
            ValueDomain::RealOnly,
            prove_nonzero_stub,
            prove_positive_stub,
            prove_nonnegative_stub,
        );
        let decision = oracle.allows(&Predicate::NonZero(x));
        assert!(!decision.allow);
    }
}
