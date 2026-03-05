//! Local predicate-proof runtime for solver facade.

use cas_ast::{Context, ExprId};

use crate::{Proof, Simplifier, ValueDomain};

fn try_ground_nonzero(ctx: &Context, expr: ExprId) -> Option<Proof> {
    cas_solver_core::predicate_proofs::try_ground_nonzero_with_shallow_recursive(
        ctx,
        expr,
        |source_ctx, source_expr| {
            let mut simplifier = Simplifier::with_context(source_ctx.clone());
            simplifier.set_collect_steps(false);

            let opts = crate::conservative_simplify::conservative_numeric_fold_options();
            let (result, _, _) = simplifier.simplify_with_stats(source_expr, opts);
            Some((simplifier.context, result))
        },
        try_ground_nonzero,
    )
}

pub(crate) fn prove_nonzero(ctx: &Context, expr: ExprId) -> Proof {
    cas_solver_core::predicate_proofs::prove_nonzero_with_default_depth_with_runtime_ground(
        ctx,
        expr,
        try_ground_nonzero,
    )
}

pub(crate) fn prove_positive(ctx: &Context, expr: ExprId, value_domain: ValueDomain) -> Proof {
    cas_solver_core::predicate_proofs::prove_positive_with_default_depth_with_runtime_ground(
        ctx,
        expr,
        value_domain,
        try_ground_nonzero,
    )
}

pub(crate) fn prove_nonnegative(ctx: &Context, expr: ExprId, value_domain: ValueDomain) -> Proof {
    cas_solver_core::predicate_proofs::prove_nonnegative_with_default_depth_with_runtime_ground(
        ctx,
        expr,
        value_domain,
        try_ground_nonzero,
    )
}
