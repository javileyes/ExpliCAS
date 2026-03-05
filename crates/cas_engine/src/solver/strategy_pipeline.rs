use super::strategy_apply::apply_strategy;
use super::{
    context_render_expr, medium_step, simplifier_contains_var, SolveCtx, SolveStep, SolverOptions,
};
use crate::engine::Simplifier;
use crate::error::CasError;
use cas_ast::{Equation, ExprId, SolutionSet};
use cas_solver_core::solve_analysis::{
    execute_prepared_equation_strategy_pipeline_with_state, is_symbolic_expr,
    resolve_discrete_strategy_result_against_equation_with_state,
    resolve_var_eliminated_residual_with_exclusions, try_enter_equation_cycle_guard_with_error,
};
use cas_solver_core::strategy_order::{default_solve_strategy_order, strategy_should_verify};

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_strategy_pipeline(
    simplifier: &mut Simplifier,
    original_eq: &Equation,
    simplified_eq: &Equation,
    diff_simplified: ExprId,
    var: &str,
    opts: SolverOptions,
    ctx: &SolveCtx,
    domain_exclusions: &[ExprId],
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let strategies = default_solve_strategy_order();
    execute_prepared_equation_strategy_pipeline_with_state(
        simplifier,
        simplified_eq,
        diff_simplified,
        var,
        strategies,
        simplifier_contains_var,
        |simplifier, residual, var_name| {
            let include_item = simplifier.collect_steps();
            Ok(resolve_var_eliminated_residual_with_exclusions(
                &mut simplifier.context,
                residual,
                var_name,
                include_item,
                domain_exclusions,
                context_render_expr,
                medium_step,
            ))
        },
        |simplifier, equation, var_name| {
            try_enter_equation_cycle_guard_with_error(
                &simplifier.context,
                equation,
                var_name,
                || {
                    CasError::SolverError(
                        "Cycle detected: equation revisited after rewriting (equivalent form loop)"
                            .to_string(),
                    )
                },
            )
        },
        |simplifier, strategy_kind| {
            let should_verify = strategy_should_verify(*strategy_kind);
            let attempt =
                apply_strategy(*strategy_kind, simplified_eq, var, simplifier, &opts, ctx);
            (attempt, should_verify)
        },
        cas_solver_core::solve_analysis::is_soft_strategy_error_by_parts::<CasError>,
        |simplifier, solutions, steps| {
            resolve_discrete_strategy_result_against_equation_with_state(
                simplifier,
                original_eq,
                var,
                solutions,
                steps,
                |state, solution| is_symbolic_expr(&state.context, solution),
                |state, equation, var_name, solution| {
                    // Verify against ORIGINAL equation, not simplified form, so
                    // domain-invalid roots (e.g. division by zero) are rejected.
                    cas_solver_core::verify_substitution::verify_solution_with_state(
                        state,
                        equation,
                        var_name,
                        solution,
                        |state, equation, var_name, candidate| {
                            cas_solver_core::verify_substitution::substitute_equation_sides(
                                &mut state.context,
                                equation,
                                var_name,
                                candidate,
                            )
                        },
                        |state, expr| state.simplify(expr).0,
                        |state, lhs, rhs| state.are_equivalent(lhs, rhs),
                    )
                },
            )
        },
        CasError::SolverError("No strategy could solve this equation.".to_string()),
    )
}
