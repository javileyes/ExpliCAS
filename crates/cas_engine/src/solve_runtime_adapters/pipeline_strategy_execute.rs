use super::pipeline_strategy_apply::apply_strategy;
use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solve_runtime::{SolveCtx, SolveStep, SolverOptions};
use crate::solve_runtime_adapters::{
    context_render_expr, map_no_strategy_solved_error, medium_step, simplifier_contains_var,
    simplifier_context, simplifier_context_mut, solver_cycle_detected_error,
};
use cas_ast::{Equation, SolutionSet};

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_strategy_pipeline(
    simplifier: &mut Simplifier,
    original_eq: &Equation,
    simplified_eq: &Equation,
    diff_simplified: cas_ast::ExprId,
    var: &str,
    opts: SolverOptions,
    ctx: &SolveCtx,
    domain_exclusions: &[cas_ast::ExprId],
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    cas_solver_core::solve_runtime_flow::execute_default_strategy_order_pipeline_with_default_cycle_guard_and_default_var_elimination_and_discrete_resolution_with_state(
        simplifier,
        original_eq,
        simplified_eq,
        diff_simplified,
        var,
        domain_exclusions,
        simplifier_contains_var,
        |state| state.collect_steps(),
        simplifier_context,
        simplifier_context,
        simplifier_context_mut,
        context_render_expr,
        medium_step,
        solver_cycle_detected_error,
        |simplifier, strategy_kind| {
            apply_strategy(strategy_kind, simplified_eq, var, simplifier, &opts, ctx)
        },
        cas_solver_core::solve_analysis::is_soft_strategy_error_by_parts::<CasError>,
        |state, equation, solve_var, solution| {
            cas_solver_core::verify_substitution::substitute_equation_sides(
                &mut state.context,
                equation,
                solve_var,
                solution,
            )
        },
        |state, expr| state.simplify(expr).0,
        |state, lhs, rhs| state.are_equivalent(lhs, rhs),
        map_no_strategy_solved_error(),
    )
}
