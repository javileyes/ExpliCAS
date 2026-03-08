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

    #[inline]
    pub fn mode(&self) -> DomainMode {
        self.inner.mode()
    }

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
