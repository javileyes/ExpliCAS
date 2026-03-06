use crate::solve_runtime_adapters::{
    attach_substeps, classify_log_solve_with_solver_ctx, context_render_expr,
    emit_quadratic_formula_scope, isolate_equation_with_solver_ctx, low_substep,
    map_symbolic_inequalities_not_supported_error, map_variable_not_found_solver_error,
    medium_step, note_log_assumption_with_solver_ctx, simplifier_context, simplifier_context_mut,
    simplifier_expand_expr, simplifier_render_expr, simplifier_simplify_expr,
    solve_equation_with_solver_ctx, SolveCtx, SolveStep, SolverOptions,
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
    cas_solver_core::solve_runtime_flow::apply_strategy_kind_with_default_kernels_and_default_step_and_error_mappers_with_state(
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
            solve_equation_with_solver_ctx(simplifier, equation, solve_var, *opts, ctx)
        },
        |simplifier, next_eq, solve_var| {
            isolate_equation_with_solver_ctx(simplifier, next_eq, solve_var, *opts, ctx)
        },
        |core_ctx, base, other_side| {
            classify_log_solve_with_solver_ctx(core_ctx, base, other_side, opts, ctx)
        },
        |simplifier, record| {
            note_log_assumption_with_solver_ctx(
                &simplifier.context,
                record.base,
                record.other_side,
                record.assumption,
                ctx,
            );
        },
        medium_step,
        attach_substeps,
        low_substep,
        map_symbolic_inequalities_not_supported_error,
        map_variable_not_found_solver_error,
        |_| emit_quadratic_formula_scope(ctx),
    )
}
