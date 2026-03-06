// ========== Common Expression Predicates ==========
// Runtime adapters over generic cas_math predicate cores.

use cas_ast::{Context, ExprId};

/// Attempt to prove whether an expression is non-zero.
pub fn prove_nonzero(ctx: &Context, expr: ExprId) -> crate::Proof {
    cas_solver_core::predicate_proofs::prove_nonzero_with_default_depth_with_runtime_evaluator(
        ctx,
        expr,
        crate::runtime_ground_eval::ground_eval_candidate,
    )
}

/// Attempt to prove whether an expression is strictly positive (> 0).
pub fn prove_positive(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> crate::Proof {
    cas_solver_core::predicate_proofs::prove_positive_with_default_depth_with_runtime_evaluator(
        ctx,
        expr,
        value_domain,
        crate::runtime_ground_eval::ground_eval_candidate,
    )
}

/// Attempt to prove whether an expression is non-negative (≥ 0).
pub(crate) fn prove_nonnegative(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> crate::Proof {
    cas_solver_core::predicate_proofs::prove_nonnegative_with_default_depth_with_runtime_evaluator(
        ctx,
        expr,
        value_domain,
        crate::runtime_ground_eval::ground_eval_candidate,
    )
}
