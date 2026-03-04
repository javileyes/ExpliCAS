// ========== Common Expression Predicates ==========
// Runtime adapters over generic cas_math predicate cores.

use cas_ast::{Context, ExprId};

/// Attempt to prove whether an expression is non-zero.
pub fn prove_nonzero(ctx: &Context, expr: ExprId) -> crate::Proof {
    // Use depth-limited version with max 50 levels to prevent stack overflow
    prove_nonzero_depth(ctx, expr, 50)
}

/// Bridge `prove_nonzero` to a core [`TriProof`].
pub(crate) fn prove_nonzero_core(ctx: &Context, expr: ExprId) -> cas_math::tri_proof::TriProof {
    cas_solver_core::predicate_proofs::proof_to_core(prove_nonzero(ctx, expr))
}

/// Bridge `prove_positive` to a core [`TriProof`].
pub(crate) fn prove_positive_core(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> cas_math::tri_proof::TriProof {
    cas_solver_core::predicate_proofs::proof_to_core(prove_positive(ctx, expr, value_domain))
}

/// Bridge `prove_nonnegative` to a core [`TriProof`].
pub(crate) fn prove_nonnegative_core(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> cas_math::tri_proof::TriProof {
    cas_solver_core::predicate_proofs::proof_to_core(prove_nonnegative(ctx, expr, value_domain))
}

/// Internal prove_nonzero with explicit depth limit.
pub(crate) fn prove_nonzero_depth(ctx: &Context, expr: ExprId, depth: usize) -> crate::Proof {
    cas_solver_core::predicate_proofs::prove_nonzero_depth_with(
        ctx,
        expr,
        depth,
        |core_ctx, inner| prove_positive(core_ctx, inner, crate::semantics::ValueDomain::RealOnly),
        super::ground_eval::try_ground_nonzero,
    )
}

/// Attempt to prove whether an expression is strictly positive (> 0).
pub fn prove_positive(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> crate::Proof {
    // Use depth-limited version with max 50 levels to prevent stack overflow
    prove_positive_depth(ctx, expr, value_domain, 50)
}

/// Internal prove_positive with explicit depth limit.
fn prove_positive_depth(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
    depth: usize,
) -> crate::Proof {
    cas_solver_core::predicate_proofs::prove_positive_depth_with(
        ctx,
        expr,
        value_domain,
        depth,
        prove_nonzero_depth,
    )
}

/// Attempt to prove whether an expression is non-negative (≥ 0).
pub(crate) fn prove_nonnegative(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> crate::Proof {
    // Use depth-limited version with max 50 levels to prevent stack overflow
    prove_nonnegative_depth(ctx, expr, value_domain, 50)
}

/// Internal prove_nonnegative with explicit depth limit.
fn prove_nonnegative_depth(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
    depth: usize,
) -> crate::Proof {
    cas_solver_core::predicate_proofs::prove_nonnegative_depth_with(
        ctx,
        expr,
        value_domain,
        depth,
        prove_nonzero_depth,
    )
}
