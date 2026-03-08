use cas_ast::Context;
use cas_solver_core::domain_cancel_decision::CancelDecision;
use cas_solver_core::domain_facts_model::Predicate;
use cas_solver_core::standard_oracle as core_standard_oracle;

use crate::{DomainMode, ValueDomain};

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
