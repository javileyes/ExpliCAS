use crate::domain_proof::Proof;
use crate::value_domain::ValueDomain;
use cas_ast::{Context, ExprId};
use cas_math::tri_proof::TriProof;

/// Convert a core `TriProof` into domain `Proof`.
pub fn core_to_proof(proof: TriProof) -> Proof {
    match proof {
        TriProof::Proven => Proof::Proven,
        TriProof::Disproven => Proof::Disproven,
        TriProof::Unknown => Proof::Unknown,
    }
}

/// Convert domain `Proof` into core `TriProof`.
///
/// `ProvenImplicit` is treated as `Proven` for the core sign/nonzero kernels.
pub fn proof_to_core(proof: Proof) -> TriProof {
    match proof {
        Proof::Proven | Proof::ProvenImplicit => TriProof::Proven,
        Proof::Disproven => TriProof::Disproven,
        Proof::Unknown => TriProof::Unknown,
    }
}

/// Run depth-limited non-zero proof with adapter closures.
pub fn prove_nonzero_depth_with<FPositive, FGround>(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
    mut prove_positive: FPositive,
    mut ground_nonzero: FGround,
) -> Proof
where
    FPositive: FnMut(&Context, ExprId) -> Proof,
    FGround: FnMut(&Context, ExprId) -> Option<Proof>,
{
    let core = cas_math::prove_nonzero::prove_nonzero_depth_with(
        ctx,
        expr,
        depth,
        |core_ctx, inner| proof_to_core(prove_positive(core_ctx, inner)),
        |core_ctx, inner| ground_nonzero(core_ctx, inner).map(proof_to_core),
    );
    core_to_proof(core)
}

/// Run depth-limited positive proof with adapter closure.
pub fn prove_positive_depth_with<FNonZero>(
    ctx: &Context,
    expr: ExprId,
    value_domain: ValueDomain,
    depth: usize,
    mut prove_nonzero: FNonZero,
) -> Proof
where
    FNonZero: FnMut(&Context, ExprId, usize) -> Proof,
{
    let real_only = value_domain == ValueDomain::RealOnly;
    let core = cas_math::prove_sign::prove_positive_depth_with(
        ctx,
        expr,
        depth,
        real_only,
        |core_ctx, inner, inner_depth| proof_to_core(prove_nonzero(core_ctx, inner, inner_depth)),
    );
    core_to_proof(core)
}

/// Run depth-limited non-negative proof with adapter closure.
pub fn prove_nonnegative_depth_with<FNonZero>(
    ctx: &Context,
    expr: ExprId,
    value_domain: ValueDomain,
    depth: usize,
    mut prove_nonzero: FNonZero,
) -> Proof
where
    FNonZero: FnMut(&Context, ExprId, usize) -> Proof,
{
    let real_only = value_domain == ValueDomain::RealOnly;
    let core = cas_math::prove_sign::prove_nonnegative_depth_with(
        ctx,
        expr,
        depth,
        real_only,
        |core_ctx, inner, inner_depth| proof_to_core(prove_nonzero(core_ctx, inner, inner_depth)),
    );
    core_to_proof(core)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_ast::Context;

    #[test]
    fn proof_mappings_roundtrip_expected() {
        assert_eq!(core_to_proof(TriProof::Proven), Proof::Proven);
        assert_eq!(core_to_proof(TriProof::Disproven), Proof::Disproven);
        assert_eq!(core_to_proof(TriProof::Unknown), Proof::Unknown);
        assert_eq!(proof_to_core(Proof::Proven), TriProof::Proven);
        assert_eq!(proof_to_core(Proof::ProvenImplicit), TriProof::Proven);
        assert_eq!(proof_to_core(Proof::Disproven), TriProof::Disproven);
        assert_eq!(proof_to_core(Proof::Unknown), TriProof::Unknown);
    }

    #[test]
    fn prove_nonzero_depth_with_uses_ground_shortcut() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let proof = prove_nonzero_depth_with(
            &ctx,
            one,
            5,
            |_c, _e| Proof::Unknown,
            |_c, _e| Some(Proof::Proven),
        );
        assert_eq!(proof, Proof::Proven);
    }

    #[test]
    fn prove_positive_depth_with_works_for_positive_literal() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let proof = prove_positive_depth_with(&ctx, two, ValueDomain::RealOnly, 5, |_c, _e, _d| {
            Proof::Unknown
        });
        assert_eq!(proof, Proof::Proven);
    }

    #[test]
    fn prove_nonnegative_depth_with_works_for_zero() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let proof =
            prove_nonnegative_depth_with(&ctx, zero, ValueDomain::RealOnly, 5, |_c, _e, _d| {
                Proof::Unknown
            });
        assert_eq!(proof, Proof::Proven);
    }
}
