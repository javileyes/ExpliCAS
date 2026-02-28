// ========== Common Expression Predicates ==========
// Runtime adapters over generic cas_math predicate cores.

use cas_ast::{Context, ExprId};

/// Attempt to prove whether an expression is non-zero.
pub fn prove_nonzero(ctx: &Context, expr: ExprId) -> crate::domain::Proof {
    // Use depth-limited version with max 50 levels to prevent stack overflow
    prove_nonzero_depth(ctx, expr, 50)
}

fn core_to_engine_proof(proof: cas_math::tri_proof::TriProof) -> crate::domain::Proof {
    match proof {
        cas_math::tri_proof::TriProof::Proven => crate::domain::Proof::Proven,
        cas_math::tri_proof::TriProof::Disproven => crate::domain::Proof::Disproven,
        cas_math::tri_proof::TriProof::Unknown => crate::domain::Proof::Unknown,
    }
}

fn engine_to_core_proof(proof: crate::domain::Proof) -> cas_math::tri_proof::TriProof {
    match proof {
        crate::domain::Proof::Proven | crate::domain::Proof::ProvenImplicit => {
            cas_math::tri_proof::TriProof::Proven
        }
        crate::domain::Proof::Disproven => cas_math::tri_proof::TriProof::Disproven,
        crate::domain::Proof::Unknown => cas_math::tri_proof::TriProof::Unknown,
    }
}

/// Internal prove_nonzero with explicit depth limit.
pub(crate) fn prove_nonzero_depth(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
) -> crate::domain::Proof {
    let core = cas_math::prove_nonzero::prove_nonzero_depth_with(
        ctx,
        expr,
        depth,
        |core_ctx, inner| {
            engine_to_core_proof(prove_positive(
                core_ctx,
                inner,
                crate::semantics::ValueDomain::RealOnly,
            ))
        },
        |core_ctx, inner| {
            super::ground_eval::try_ground_nonzero(core_ctx, inner).map(engine_to_core_proof)
        },
    );
    core_to_engine_proof(core)
}

/// Attempt to prove whether an expression is strictly positive (> 0).
pub fn prove_positive(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> crate::domain::Proof {
    // Use depth-limited version with max 50 levels to prevent stack overflow
    prove_positive_depth(ctx, expr, value_domain, 50)
}

/// Internal prove_positive with explicit depth limit.
fn prove_positive_depth(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
    depth: usize,
) -> crate::domain::Proof {
    let real_only = value_domain == crate::semantics::ValueDomain::RealOnly;
    let core = cas_math::prove_sign::prove_positive_depth_with(
        ctx,
        expr,
        depth,
        real_only,
        |core_ctx, inner, inner_depth| {
            engine_to_core_proof(prove_nonzero_depth(core_ctx, inner, inner_depth))
        },
    );
    core_to_engine_proof(core)
}

/// Attempt to prove whether an expression is non-negative (≥ 0).
pub(crate) fn prove_nonnegative(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
) -> crate::domain::Proof {
    // Use depth-limited version with max 50 levels to prevent stack overflow
    prove_nonnegative_depth(ctx, expr, value_domain, 50)
}

/// Internal prove_nonnegative with explicit depth limit.
fn prove_nonnegative_depth(
    ctx: &Context,
    expr: ExprId,
    value_domain: crate::semantics::ValueDomain,
    depth: usize,
) -> crate::domain::Proof {
    let real_only = value_domain == crate::semantics::ValueDomain::RealOnly;
    let core = cas_math::prove_sign::prove_nonnegative_depth_with(
        ctx,
        expr,
        depth,
        real_only,
        |core_ctx, inner, inner_depth| {
            engine_to_core_proof(prove_nonzero_depth(core_ctx, inner, inner_depth))
        },
    );
    core_to_engine_proof(core)
}
