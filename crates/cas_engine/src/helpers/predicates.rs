// ========== Common Expression Predicates ==========
// Runtime adapters over generic cas_math predicate cores.

use cas_ast::{Context, ExprId};

/// Attempt to prove whether an expression is non-zero.
pub fn prove_nonzero(ctx: &Context, expr: ExprId) -> crate::Proof {
    crate::proof_runtime::prove_nonzero(ctx, expr)
}

/// Attempt to prove whether an expression is strictly positive (> 0).
pub fn prove_positive(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> crate::Proof {
    crate::proof_runtime::prove_positive(ctx, expr, value_domain)
}

/// Attempt to prove whether an expression is non-negative (≥ 0).
pub(crate) fn prove_nonnegative(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> crate::Proof {
    crate::proof_runtime::prove_nonnegative(ctx, expr, value_domain)
}
