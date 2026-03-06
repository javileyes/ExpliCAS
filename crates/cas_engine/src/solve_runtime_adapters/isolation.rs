use super::*;
use crate::engine::Simplifier;
use crate::error::CasError;
use cas_ast::{ExprId, RelOp, SolutionSet};

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
    cas_solver_core::solve_runtime_flow::dispatch_isolation_with_default_kernels_and_default_arithmetic_pow_function_routes_with_state(
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
            solve_equation_with_solver_ctx(simplifier, equation, solve_var, opts, ctx)
        },
        simplifier_is_known_negative,
        |simplifier, equation, solve_var| {
            isolate_equation_with_solver_ctx(simplifier, &equation, solve_var, opts, ctx)
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
            classify_log_solve_with_solver_ctx(
                &simplifier.context,
                tactic_base,
                tactic_rhs,
                &opts,
                ctx,
            )
        },
        map_isolation_error,
        |core_ctx, base, eq_rhs, assumption| {
            note_log_assumption_with_solver_ctx(core_ctx, base, eq_rhs, assumption, ctx)
        },
        note_log_blocked_hint_with_default_sink,
        map_unsupported_in_real_domain_error,
        simplifier_collect_steps,
        simplify_rhs_with_step_pairs,
        sym_name_as_string,
        |_simplifier, missing_var| map_variable_not_found_solver_error(missing_var),
        |_simplifier, unsupported_var, message| map_isolation_error(unsupported_var, message),
        |_simplifier, fn_name| map_unknown_function_error(fn_name),
        |simplifier, equation, solve_var| {
            isolate_equation_with_solver_ctx(simplifier, &equation, solve_var, opts, ctx)
        },
        simplifier_collect_steps,
        medium_step,
        |_simplifier, lhs_expr| map_isolation_cannot_isolate_error(var, lhs_expr),
    )
}
