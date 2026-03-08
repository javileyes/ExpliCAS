// ========== Common Expression Predicates ==========
// Runtime adapters over generic cas_math predicate cores.

use cas_ast::{Context, ExprId};

/// Attempt to prove whether an expression is non-zero.
pub fn prove_nonzero(ctx: &Context, expr: ExprId) -> crate::Proof {
    cas_solver_core::proof_runtime_bound_runtime::prove_nonzero_with_runtime_proof_simplifier::<
        crate::Simplifier,
    >(ctx, expr)
}

/// Attempt to prove whether an expression is strictly positive (> 0).
pub fn prove_positive(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> crate::Proof {
    cas_solver_core::proof_runtime_bound_runtime::prove_positive_with_runtime_proof_simplifier::<
        crate::Simplifier,
    >(ctx, expr, value_domain)
}

/// Attempt to prove whether an expression is non-negative (≥ 0).
pub(crate) fn prove_nonnegative(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> crate::Proof {
    cas_solver_core::proof_runtime_bound_runtime::prove_nonnegative_with_runtime_proof_simplifier::<
        crate::Simplifier,
    >(ctx, expr, value_domain)
}
