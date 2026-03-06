//! Local predicate-proof runtime for engine facade.

use cas_ast::{Context, ExprId};

use crate::ValueDomain;

pub(crate) fn ground_eval_candidate(
    source_ctx: &Context,
    source_expr: ExprId,
    opts: &cas_solver_core::simplify_options::SimplifyOptions,
) -> Option<(Context, ExprId)> {
    cas_solver_core::ground_eval_runtime::ground_eval_candidate_with_runtime_simplifier_with_state(
        source_ctx,
        source_expr,
        opts,
        crate::Simplifier::with_context,
        |state, collect| state.set_collect_steps(collect),
        |state, expr, simplify_opts| state.simplify_with_stats(expr, simplify_opts).0,
        |state| state.context,
    )
}

pub(crate) fn prove_nonzero(ctx: &Context, expr: ExprId) -> crate::Proof {
    cas_solver_core::predicate_proofs::prove_nonzero_with_default_depth_with_runtime_evaluator(
        ctx,
        expr,
        ground_eval_candidate,
    )
}

pub(crate) fn prove_positive(
    ctx: &Context,
    expr: ExprId,
    value_domain: ValueDomain,
) -> crate::Proof {
    cas_solver_core::predicate_proofs::prove_positive_with_default_depth_with_runtime_evaluator(
        ctx,
        expr,
        value_domain,
        ground_eval_candidate,
    )
}

pub(crate) fn prove_nonnegative(
    ctx: &Context,
    expr: ExprId,
    value_domain: ValueDomain,
) -> crate::Proof {
    cas_solver_core::predicate_proofs::prove_nonnegative_with_default_depth_with_runtime_evaluator(
        ctx,
        expr,
        value_domain,
        ground_eval_candidate,
    )
}
