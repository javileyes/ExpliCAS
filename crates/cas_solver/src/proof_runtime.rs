//! Local predicate-proof runtime for solver facade.

use cas_ast::{Context, Expr, ExprId};

use crate::{
    DomainMode, EvalConfig, PhaseBudgets, Proof, SharedSemanticConfig, Simplifier, SimplifyOptions,
    ValueDomain,
};

fn prove_nonzero_depth(ctx: &Context, expr: ExprId, depth: usize) -> Proof {
    cas_solver_core::predicate_proofs::prove_nonzero_depth_with(
        ctx,
        expr,
        depth,
        |core_ctx, inner| prove_positive(core_ctx, inner, ValueDomain::RealOnly),
        try_ground_nonzero,
    )
}

fn prove_positive_depth(
    ctx: &Context,
    expr: ExprId,
    value_domain: ValueDomain,
    depth: usize,
) -> Proof {
    cas_solver_core::predicate_proofs::prove_positive_depth_with(
        ctx,
        expr,
        value_domain,
        depth,
        prove_nonzero_depth,
    )
}

fn try_ground_nonzero(ctx: &Context, expr: ExprId) -> Option<Proof> {
    cas_math::ground_nonzero::try_ground_nonzero_with(
        ctx,
        expr,
        |source_ctx, source_expr| {
            let mut simplifier = Simplifier::with_context(source_ctx.clone());
            simplifier.set_collect_steps(false);

            let opts = SimplifyOptions {
                collect_steps: false,
                expand_mode: false,
                shared: SharedSemanticConfig {
                    semantics: EvalConfig {
                        domain_mode: DomainMode::Generic,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                budgets: PhaseBudgets {
                    core_iters: 4,
                    transform_iters: 2,
                    rationalize_iters: 0,
                    post_iters: 2,
                    max_total_rewrites: 50,
                },
                ..Default::default()
            };

            let (result, _, _) = simplifier.simplify_with_stats(source_expr, opts);
            Some((simplifier.context, result))
        },
        |evaluated_ctx, evaluated_expr| match evaluated_ctx.get(evaluated_expr) {
            Expr::Number(n) => {
                if num_traits::Zero::is_zero(n) {
                    Some(Proof::Disproven)
                } else {
                    Some(Proof::Proven)
                }
            }
            _ => None,
        },
        |evaluated_ctx, evaluated_expr| {
            let proof = prove_nonzero_depth(evaluated_ctx, evaluated_expr, 8);
            if proof == Proof::Proven || proof == Proof::Disproven {
                Some(proof)
            } else {
                None
            }
        },
    )
}

pub(crate) fn prove_nonzero(ctx: &Context, expr: ExprId) -> Proof {
    prove_nonzero_depth(ctx, expr, 50)
}

pub(crate) fn prove_positive(ctx: &Context, expr: ExprId, value_domain: ValueDomain) -> Proof {
    prove_positive_depth(ctx, expr, value_domain, 50)
}
