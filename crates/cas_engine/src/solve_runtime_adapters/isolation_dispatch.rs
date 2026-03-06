use crate::solve_runtime_adapters::{
    context_render_expr, map_isolation_error, map_unsupported_in_real_domain_error,
    simplifier_collect_steps, simplifier_context, simplifier_context_mut,
    simplifier_is_known_negative, simplifier_prove_nonzero_status, simplifier_simplify_expr,
    simplify_rhs_with_step_pairs, sym_name_as_string, SolveCtx, SolveStep, SolverOptions,
};
use crate::{CasError, Simplifier};
use cas_ast::{ExprId, RelOp, SolutionSet};

#[allow(clippy::too_many_arguments)]
pub(crate) fn isolate_with_default_depth(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    cas_solver_core::solve_runtime_isolation_entry_runtime::isolate_with_default_depth_guard_and_error_with_state(
        simplifier,
        ctx.depth(),
        |state| dispatch_isolation_with_default_routes(lhs, rhs, op.clone(), var, state, opts, ctx),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_isolation_with_default_routes(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    cas_solver_core::solve_runtime_isolation_dispatch_runtime::dispatch_isolation_with_default_routes_and_mappers_with_state(
        simplifier,
        lhs,
        rhs,
        op.clone(),
        var,
        simplifier_context,
        simplifier_context_mut,
        simplifier_simplify_expr,
        simplifier_collect_steps,
        simplifier_simplify_expr,
        simplifier_prove_nonzero_status,
        context_render_expr,
        |state| state.collect_steps(),
        |simplifier, equation, solve_var| {
            crate::solve_core_runtime::solve_inner(equation, solve_var, simplifier, opts, ctx)
        },
        simplifier_is_known_negative,
        |simplifier, equation, solve_var| {
            isolate_with_default_depth(
                equation.lhs,
                equation.rhs,
                equation.op.clone(),
                solve_var,
                simplifier,
                opts,
                ctx,
            )
        },
        opts.core_domain_mode(),
        opts.wildcard_scope(),
        opts.value_domain == crate::ValueDomain::RealOnly,
        opts.budget,
        simplifier_collect_steps,
        crate::SimplifyOptions::for_solve_tactic(opts.domain_mode),
        simplifier_simplify_expr,
        |_simplifier| crate::clear_blocked_hints(),
        |simplifier, expr, tactic_opts| simplifier.simplify_with_options(expr, tactic_opts.clone()).0,
        |simplifier, tactic_base, tactic_rhs| {
            cas_solver_core::solve_runtime_flow::classify_log_solve_with_domain_env_and_runtime_positive_prover(
                &simplifier.context,
                tactic_base,
                tactic_rhs,
                opts.value_domain,
                opts.core_domain_mode(),
                &ctx.domain_env,
                crate::proof_runtime::prove_positive,
            )
        },
        map_isolation_error,
        |core_ctx, base, eq_rhs, assumption| {
            cas_solver_core::solve_runtime_flow::note_log_assumption_with_runtime_sink(
                core_ctx,
                base,
                eq_rhs,
                assumption,
                |event| ctx.note_assumption(event),
            );
        },
        |core_ctx, hint| {
            cas_solver_core::solve_runtime_flow::note_log_blocked_hint_with_runtime_sink(
                core_ctx,
                hint,
                crate::register_blocked_hint,
            );
        },
        map_unsupported_in_real_domain_error,
        simplifier_collect_steps,
        simplify_rhs_with_step_pairs,
        sym_name_as_string,
        |simplifier, equation, solve_var| {
            isolate_with_default_depth(
                equation.lhs,
                equation.rhs,
                equation.op.clone(),
                solve_var,
                simplifier,
                opts,
                ctx,
            )
        },
        simplifier_collect_steps,
    )
}
