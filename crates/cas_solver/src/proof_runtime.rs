//! Local predicate-proof runtime for solver facade.

use cas_ast::{Context, ExprId};

use crate::{Proof, ValueDomain};

pub(crate) fn ground_eval_candidate(
    source_ctx: &Context,
    source_expr: ExprId,
    opts: &cas_solver_core::simplify_options::SimplifyOptions,
) -> Option<(Context, ExprId)> {
    cas_solver_core::proof_runtime_bound_runtime::ground_eval_candidate_with_runtime_simplifier_contract(
        source_ctx,
        source_expr,
        opts,
        crate::Simplifier::with_context,
        |state, collect| state.set_collect_steps(collect),
        |state, expr, simplify_opts| state.simplify_with_stats(expr, simplify_opts).0,
        |state| state.context,
    )
}

pub(crate) fn prove_nonzero(ctx: &Context, expr: ExprId) -> Proof {
    cas_solver_core::proof_runtime_bound_runtime::prove_nonzero_with_runtime_simplifier_contract(
        ctx,
        expr,
        crate::Simplifier::with_context,
        |state, collect| state.set_collect_steps(collect),
        |state, expr, simplify_opts| state.simplify_with_stats(expr, simplify_opts).0,
        |state| state.context,
    )
}

pub(crate) fn prove_positive(ctx: &Context, expr: ExprId, value_domain: ValueDomain) -> Proof {
    cas_solver_core::proof_runtime_bound_runtime::prove_positive_with_runtime_simplifier_contract(
        ctx,
        expr,
        value_domain,
        crate::Simplifier::with_context,
        |state, collect| state.set_collect_steps(collect),
        |state, expr, simplify_opts| state.simplify_with_stats(expr, simplify_opts).0,
        |state| state.context,
    )
}

pub(crate) fn prove_nonnegative(ctx: &Context, expr: ExprId, value_domain: ValueDomain) -> Proof {
    cas_solver_core::proof_runtime_bound_runtime::prove_nonnegative_with_runtime_simplifier_contract(
        ctx,
        expr,
        value_domain,
        crate::Simplifier::with_context,
        |state, collect| state.set_collect_steps(collect),
        |state, expr, simplify_opts| state.simplify_with_stats(expr, simplify_opts).0,
        |state| state.context,
    )
}
