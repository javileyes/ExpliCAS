use crate::domain_proof::Proof;
use crate::value_domain::ValueDomain;
use cas_ast::{Context, Expr, ExprId};
use cas_math::tri_proof::TriProof;
use num_traits::Zero;

/// Default recursion depth budget for public predicate proof entrypoints.
pub const DEFAULT_PROOF_DEPTH: usize = 50;

/// Shallow recursion depth used by ground-eval fallback recursion.
pub const SHALLOW_PROOF_DEPTH: usize = 8;

/// Runtime callback used by predicate-proof adapters when a ground-expression
/// non-zero fallback is available in the host runtime crate.
pub type TryGroundNonZeroFn = fn(&Context, ExprId) -> Option<Proof>;

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

/// Adapter: run a runtime non-zero prover and convert to core `TriProof`.
pub fn prove_nonzero_core_with<F>(ctx: &Context, expr: ExprId, mut prove_nonzero: F) -> TriProof
where
    F: FnMut(&Context, ExprId) -> Proof,
{
    proof_to_core(prove_nonzero(ctx, expr))
}

/// Adapter: run a runtime positive prover and convert to core `TriProof`.
pub fn prove_positive_core_with<F>(
    ctx: &Context,
    expr: ExprId,
    value_domain: ValueDomain,
    mut prove_positive: F,
) -> TriProof
where
    F: FnMut(&Context, ExprId, ValueDomain) -> Proof,
{
    proof_to_core(prove_positive(ctx, expr, value_domain))
}

/// Adapter: run a runtime non-negative prover and convert to core `TriProof`.
pub fn prove_nonnegative_core_with<F>(
    ctx: &Context,
    expr: ExprId,
    value_domain: ValueDomain,
    mut prove_nonnegative: F,
) -> TriProof
where
    F: FnMut(&Context, ExprId, ValueDomain) -> Proof,
{
    proof_to_core(prove_nonnegative(ctx, expr, value_domain))
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

/// Run non-zero proof using [`DEFAULT_PROOF_DEPTH`].
pub fn prove_nonzero_with_default_depth<FPositive, FGround>(
    ctx: &Context,
    expr: ExprId,
    prove_positive: FPositive,
    ground_nonzero: FGround,
) -> Proof
where
    FPositive: FnMut(&Context, ExprId) -> Proof,
    FGround: FnMut(&Context, ExprId) -> Option<Proof>,
{
    prove_nonzero_depth_with(
        ctx,
        expr,
        DEFAULT_PROOF_DEPTH,
        prove_positive,
        ground_nonzero,
    )
}

/// Depth-limited non-zero proof wired to the standard runtime pattern:
/// `prove_positive(RealOnly)` + runtime `try_ground_nonzero`.
pub fn prove_nonzero_depth_with_runtime_ground(
    ctx: &Context,
    expr: ExprId,
    depth: usize,
    try_ground_nonzero: TryGroundNonZeroFn,
) -> Proof {
    prove_nonzero_depth_with(
        ctx,
        expr,
        depth,
        |core_ctx, inner| {
            prove_positive_with_default_depth_with_runtime_ground(
                core_ctx,
                inner,
                ValueDomain::RealOnly,
                try_ground_nonzero,
            )
        },
        try_ground_nonzero,
    )
}

/// Non-zero proof with [`DEFAULT_PROOF_DEPTH`] wired to the standard runtime
/// pattern: `prove_positive(RealOnly)` + runtime `try_ground_nonzero`.
pub fn prove_nonzero_with_default_depth_with_runtime_ground(
    ctx: &Context,
    expr: ExprId,
    try_ground_nonzero: TryGroundNonZeroFn,
) -> Proof {
    prove_nonzero_depth_with_runtime_ground(ctx, expr, DEFAULT_PROOF_DEPTH, try_ground_nonzero)
}

/// Shallow non-zero proof helper used by runtime ground-eval recursion.
#[inline]
pub fn prove_nonzero_shallow_with_runtime_ground(
    ctx: &Context,
    expr: ExprId,
    try_ground_nonzero: TryGroundNonZeroFn,
) -> Proof {
    prove_nonzero_depth_with_runtime_ground(ctx, expr, SHALLOW_PROOF_DEPTH, try_ground_nonzero)
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

/// Run positive proof using [`DEFAULT_PROOF_DEPTH`].
pub fn prove_positive_with_default_depth<FNonZero>(
    ctx: &Context,
    expr: ExprId,
    value_domain: ValueDomain,
    prove_nonzero: FNonZero,
) -> Proof
where
    FNonZero: FnMut(&Context, ExprId, usize) -> Proof,
{
    prove_positive_depth_with(ctx, expr, value_domain, DEFAULT_PROOF_DEPTH, prove_nonzero)
}

/// Positive proof with [`DEFAULT_PROOF_DEPTH`] wired to runtime
/// `try_ground_nonzero` via the standard recursive non-zero adapter.
pub fn prove_positive_with_default_depth_with_runtime_ground(
    ctx: &Context,
    expr: ExprId,
    value_domain: ValueDomain,
    try_ground_nonzero: TryGroundNonZeroFn,
) -> Proof {
    prove_positive_with_default_depth(ctx, expr, value_domain, |core_ctx, inner, inner_depth| {
        prove_nonzero_depth_with_runtime_ground(core_ctx, inner, inner_depth, try_ground_nonzero)
    })
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

/// Run non-negative proof using [`DEFAULT_PROOF_DEPTH`].
pub fn prove_nonnegative_with_default_depth<FNonZero>(
    ctx: &Context,
    expr: ExprId,
    value_domain: ValueDomain,
    prove_nonzero: FNonZero,
) -> Proof
where
    FNonZero: FnMut(&Context, ExprId, usize) -> Proof,
{
    prove_nonnegative_depth_with(ctx, expr, value_domain, DEFAULT_PROOF_DEPTH, prove_nonzero)
}

/// Non-negative proof with [`DEFAULT_PROOF_DEPTH`] wired to runtime
/// `try_ground_nonzero` via the standard recursive non-zero adapter.
pub fn prove_nonnegative_with_default_depth_with_runtime_ground(
    ctx: &Context,
    expr: ExprId,
    value_domain: ValueDomain,
    try_ground_nonzero: TryGroundNonZeroFn,
) -> Proof {
    prove_nonnegative_with_default_depth(ctx, expr, value_domain, |core_ctx, inner, inner_depth| {
        prove_nonzero_depth_with_runtime_ground(core_ctx, inner, inner_depth, try_ground_nonzero)
    })
}

/// Run the shared "ground expression non-zero" fallback.
///
/// The caller provides a simplification/evaluation closure and a recursive
/// non-zero prover used as a bounded fallback when simplification does not
/// collapse to a numeric literal directly.
pub fn try_ground_nonzero_with<FGroundEval, FRecursiveNonZero>(
    ctx: &Context,
    expr: ExprId,
    mut ground_eval: FGroundEval,
    mut recursive_nonzero: FRecursiveNonZero,
) -> Option<Proof>
where
    FGroundEval: FnMut(&Context, ExprId) -> Option<(Context, ExprId)>,
    FRecursiveNonZero: FnMut(&Context, ExprId) -> Proof,
{
    cas_math::ground_nonzero::try_ground_nonzero_with(
        ctx,
        expr,
        |source_ctx, source_expr| ground_eval(source_ctx, source_expr),
        |evaluated_ctx, evaluated_expr| match evaluated_ctx.get(evaluated_expr) {
            Expr::Number(n) => {
                if n.is_zero() {
                    Some(Proof::Disproven)
                } else {
                    Some(Proof::Proven)
                }
            }
            _ => None,
        },
        |evaluated_ctx, evaluated_expr| {
            let proof = recursive_nonzero(evaluated_ctx, evaluated_expr);
            if proof == Proof::Proven || proof == Proof::Disproven {
                Some(proof)
            } else {
                None
            }
        },
    )
}

/// Ground non-zero fallback wired to the standard shallow recursive
/// non-zero adapter based on runtime `try_ground_nonzero`.
pub fn try_ground_nonzero_with_shallow_recursive<FGroundEval>(
    ctx: &Context,
    expr: ExprId,
    ground_eval: FGroundEval,
    try_ground_nonzero: TryGroundNonZeroFn,
) -> Option<Proof>
where
    FGroundEval: FnMut(&Context, ExprId) -> Option<(Context, ExprId)>,
{
    try_ground_nonzero_with(ctx, expr, ground_eval, |evaluated_ctx, evaluated_expr| {
        prove_nonzero_shallow_with_runtime_ground(evaluated_ctx, evaluated_expr, try_ground_nonzero)
    })
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
    fn core_adapter_helpers_map_runtime_proofs() {
        let mut ctx = Context::new();
        let x = ctx.var("x");

        let nonzero = prove_nonzero_core_with(&ctx, x, |_core_ctx, _expr| Proof::ProvenImplicit);
        let positive =
            prove_positive_core_with(&ctx, x, ValueDomain::RealOnly, |_core_ctx, _expr, _vd| {
                Proof::Disproven
            });
        let nonnegative =
            prove_nonnegative_core_with(&ctx, x, ValueDomain::RealOnly, |_core_ctx, _expr, _vd| {
                Proof::Unknown
            });

        assert_eq!(nonzero, TriProof::Proven);
        assert_eq!(positive, TriProof::Disproven);
        assert_eq!(nonnegative, TriProof::Unknown);
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

    #[test]
    fn try_ground_nonzero_with_returns_proven_for_nonzero_number() {
        let mut src = Context::new();
        let expr = src.num(7);

        let proof = try_ground_nonzero_with(
            &src,
            expr,
            |ctx, id| Some((ctx.clone(), id)),
            |_ctx, _id| Proof::Unknown,
        );

        assert_eq!(proof, Some(Proof::Proven));
    }

    #[test]
    fn try_ground_nonzero_with_uses_recursive_fallback() {
        let mut src = Context::new();
        let expr = src.num(1);

        let proof = try_ground_nonzero_with(
            &src,
            expr,
            |ctx, _id| {
                let mut out = ctx.clone();
                let x = out.var("x");
                Some((out, x))
            },
            |_ctx, _id| Proof::Disproven,
        );

        assert_eq!(proof, Some(Proof::Disproven));
    }

    fn runtime_ground_proven(_ctx: &Context, _expr: ExprId) -> Option<Proof> {
        Some(Proof::Proven)
    }

    #[test]
    fn runtime_ground_nonzero_adapter_uses_callback() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let f = ctx.intern_symbol("f");
        let fx = ctx.add(cas_ast::Expr::Function(f, vec![one]));
        let proof = prove_nonzero_depth_with_runtime_ground(&ctx, fx, 3, runtime_ground_proven);
        assert_eq!(proof, Proof::Proven);
    }

    #[test]
    fn runtime_ground_positive_adapter_keeps_literal_behavior() {
        let mut ctx = Context::new();
        let two = ctx.num(2);
        let proof = prove_positive_with_default_depth_with_runtime_ground(
            &ctx,
            two,
            ValueDomain::RealOnly,
            runtime_ground_proven,
        );
        assert_eq!(proof, Proof::Proven);
    }

    #[test]
    fn runtime_ground_nonnegative_adapter_keeps_literal_behavior() {
        let mut ctx = Context::new();
        let zero = ctx.num(0);
        let proof = prove_nonnegative_with_default_depth_with_runtime_ground(
            &ctx,
            zero,
            ValueDomain::RealOnly,
            runtime_ground_proven,
        );
        assert_eq!(proof, Proof::Proven);
    }

    #[test]
    fn try_ground_nonzero_with_shallow_recursive_uses_runtime_callback() {
        let mut src = Context::new();
        let one = src.num(1);
        let f = src.intern_symbol("f");
        let fx = src.add(cas_ast::Expr::Function(f, vec![one]));

        fn runtime_ground(_ctx: &Context, _expr: ExprId) -> Option<Proof> {
            Some(Proof::Disproven)
        }

        let proof = try_ground_nonzero_with_shallow_recursive(
            &src,
            fx,
            |ctx, id| Some((ctx.clone(), id)),
            runtime_ground,
        );

        assert_eq!(proof, Some(Proof::Disproven));
    }
}
