//! Helpers to map external proof enums into solver-core proof statuses.

use crate::linear_solution::NonZeroStatus;
use crate::log_domain::ProofStatus;
use crate::proof_status::proof_status_to_nonzero_status;
use cas_ast::{Context, ExprId};
use cas_math::tri_proof::TriProof;

/// Map an external proof value to [`ProofStatus`].
pub fn map_external_proof_status_with<P, FIsProven, FIsDisproven>(
    proof: P,
    is_proven: FIsProven,
    is_disproven: FIsDisproven,
) -> ProofStatus
where
    FIsProven: Fn(&P) -> bool,
    FIsDisproven: Fn(&P) -> bool,
{
    if is_proven(&proof) {
        ProofStatus::Proven
    } else if is_disproven(&proof) {
        ProofStatus::Disproven
    } else {
        ProofStatus::Unknown
    }
}

/// Map an external proof value to [`NonZeroStatus`].
pub fn map_external_nonzero_status_with<P, FIsProven, FIsDisproven>(
    proof: P,
    is_proven: FIsProven,
    is_disproven: FIsDisproven,
) -> NonZeroStatus
where
    FIsProven: Fn(&P) -> bool,
    FIsDisproven: Fn(&P) -> bool,
{
    let status = map_external_proof_status_with(proof, is_proven, is_disproven);
    proof_status_to_nonzero_status(status)
}

/// Map [`TriProof`] to [`ProofStatus`].
pub fn map_tri_proof_status(proof: TriProof) -> ProofStatus {
    match proof {
        TriProof::Proven => ProofStatus::Proven,
        TriProof::Disproven => ProofStatus::Disproven,
        TriProof::Unknown => ProofStatus::Unknown,
    }
}

/// Map [`TriProof`] to [`NonZeroStatus`].
pub fn map_tri_nonzero_status(proof: TriProof) -> NonZeroStatus {
    proof_status_to_nonzero_status(map_tri_proof_status(proof))
}

/// Evaluate one expression with a tri-valued non-zero prover and map to
/// [`NonZeroStatus`].
pub fn classify_nonzero_status_with_tri_prover<FProve>(
    ctx: &Context,
    expr: ExprId,
    mut prove_nonzero: FProve,
) -> NonZeroStatus
where
    FProve: FnMut(&Context, ExprId) -> TriProof,
{
    map_tri_nonzero_status(prove_nonzero(ctx, expr))
}

#[cfg(test)]
mod tests {
    use super::{
        classify_nonzero_status_with_tri_prover, map_external_nonzero_status_with,
        map_external_proof_status_with, map_tri_nonzero_status, map_tri_proof_status,
    };
    use crate::linear_solution::NonZeroStatus;
    use crate::log_domain::ProofStatus;
    use cas_math::tri_proof::TriProof;

    #[derive(Debug, Clone, Copy)]
    enum ExternalProof {
        Yes,
        No,
        Maybe,
    }

    #[test]
    fn map_external_proof_status_maps_three_valued_domain() {
        let proven = map_external_proof_status_with(
            ExternalProof::Yes,
            |p| matches!(p, ExternalProof::Yes),
            |p| matches!(p, ExternalProof::No),
        );
        let disproven = map_external_proof_status_with(
            ExternalProof::No,
            |p| matches!(p, ExternalProof::Yes),
            |p| matches!(p, ExternalProof::No),
        );
        let unknown = map_external_proof_status_with(
            ExternalProof::Maybe,
            |p| matches!(p, ExternalProof::Yes),
            |p| matches!(p, ExternalProof::No),
        );

        assert_eq!(proven, ProofStatus::Proven);
        assert_eq!(disproven, ProofStatus::Disproven);
        assert_eq!(unknown, ProofStatus::Unknown);
    }

    #[test]
    fn map_external_nonzero_status_composes_with_proof_mapping() {
        let out = map_external_nonzero_status_with(
            ExternalProof::Yes,
            |p| matches!(p, ExternalProof::Yes),
            |p| matches!(p, ExternalProof::No),
        );
        assert_eq!(out, NonZeroStatus::NonZero);
    }

    #[test]
    fn map_tri_status_helpers_cover_all_variants() {
        assert_eq!(map_tri_proof_status(TriProof::Proven), ProofStatus::Proven);
        assert_eq!(
            map_tri_proof_status(TriProof::Disproven),
            ProofStatus::Disproven
        );
        assert_eq!(
            map_tri_proof_status(TriProof::Unknown),
            ProofStatus::Unknown
        );

        assert_eq!(
            map_tri_nonzero_status(TriProof::Proven),
            NonZeroStatus::NonZero
        );
        assert_eq!(
            map_tri_nonzero_status(TriProof::Disproven),
            NonZeroStatus::Zero
        );
        assert_eq!(
            map_tri_nonzero_status(TriProof::Unknown),
            NonZeroStatus::Unknown
        );
    }

    #[test]
    fn classify_nonzero_status_with_tri_prover_maps_callback_result() {
        let mut ctx = cas_ast::Context::new();
        let x = ctx.var("x");
        let zero = ctx.num(0);

        let status_x = classify_nonzero_status_with_tri_prover(&ctx, x, |_core_ctx, expr| {
            if expr == x {
                TriProof::Proven
            } else {
                TriProof::Unknown
            }
        });
        assert_eq!(status_x, NonZeroStatus::NonZero);

        let status_zero = classify_nonzero_status_with_tri_prover(&ctx, zero, |_core_ctx, expr| {
            if expr == zero {
                TriProof::Disproven
            } else {
                TriProof::Unknown
            }
        });
        assert_eq!(status_zero, NonZeroStatus::Zero);
    }
}
