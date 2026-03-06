use crate::solve_runtime_adapters::{
    context_render_expr, emit_quadratic_formula_scope, isolate_with_default_depth,
    simplifier_context, simplifier_context_mut, simplifier_expand_expr, simplifier_render_expr,
    simplifier_simplify_expr, SolveCtx, SolveStep, SolverOptions,
};
use crate::{CasError, Simplifier};
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::strategy_order::SolveStrategyKind;

pub(crate) fn apply_strategy(
    kind: SolveStrategyKind,
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let is_real_only = matches!(opts.value_domain, crate::ValueDomain::RealOnly);
    cas_solver_core::solve_runtime_pipeline_strategy_apply_runtime::apply_strategy_with_default_mappers_and_state(
        simplifier,
        kind,
        equation,
        var,
        is_real_only,
        opts.core_domain_mode(),
        opts.wildcard_scope(),
        |state| state.collect_steps(),
        simplifier_context,
        simplifier_context_mut,
        |simplifier, collecting| simplifier.set_collect_steps(collecting),
        simplifier_simplify_expr,
        simplifier_expand_expr,
        simplifier_render_expr,
        context_render_expr,
        |simplifier, equation, solve_var| {
            crate::solve_core_runtime::solve_inner(equation, solve_var, simplifier, *opts, ctx)
        },
        |simplifier, next_eq, solve_var| {
            isolate_with_default_depth(
                next_eq.lhs,
                next_eq.rhs,
                next_eq.op.clone(),
                solve_var,
                simplifier,
                *opts,
                ctx,
            )
        },
        |core_ctx, base, other_side| {
            cas_solver_core::solve_runtime_flow::classify_log_solve_with_domain_env_and_runtime_positive_prover(
                core_ctx,
                base,
                other_side,
                opts.value_domain,
                opts.core_domain_mode(),
                &ctx.domain_env,
                crate::proof_runtime::prove_positive,
            )
        },
        |simplifier, record| {
            cas_solver_core::solve_runtime_flow::note_log_assumption_with_runtime_sink(
                &simplifier.context,
                record.base,
                record.other_side,
                record.assumption,
                |event| ctx.note_assumption(event),
            );
        },
        |_| emit_quadratic_formula_scope(ctx),
    )
}
