use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{
    context_render_expr, medium_step, simplifier_context, simplifier_context_mut,
    simplifier_expand_expr, simplifier_render_expr, simplifier_simplify_expr, SolveCtx, SolveStep,
    SolveSubStep, SolverOptions,
};
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::quadratic_strategy::execute_quadratic_strategy_with_default_factorized_and_candidate_pipelines_and_unified_step_mappers_with_state;
use cas_solver_core::substitution::execute_exponential_substitution_strategy_result_pipeline_with_default_substitution_var_and_plan_with_state;

use super::solve_entrypoints::solve_with_ctx_and_options;

pub(super) fn apply_substitution_strategy(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let include_didactic_items = simplifier.collect_steps();
    execute_exponential_substitution_strategy_result_pipeline_with_default_substitution_var_and_plan_with_state(
        simplifier,
        equation,
        var,
        include_didactic_items,
        simplifier_context_mut,
        simplifier_render_expr,
        |simplifier, next_eq, solve_var| {
            solve_with_ctx_and_options(next_eq, solve_var, simplifier, *opts, ctx)
        },
        medium_step,
    )
}

pub(super) fn apply_quadratic_strategy(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let include_items = simplifier.collect_steps();
    let is_real_only = matches!(opts.value_domain, crate::ValueDomain::RealOnly);
    execute_quadratic_strategy_with_default_factorized_and_candidate_pipelines_and_unified_step_mappers_with_state(
        simplifier,
        equation,
        var,
        include_items,
        is_real_only,
        simplifier_context,
        simplifier_context_mut,
        |simplifier, collecting| simplifier.set_collect_steps(collecting),
        simplifier_simplify_expr,
        simplifier_expand_expr,
        context_render_expr,
        |simplifier, next_eq| solve_with_ctx_and_options(next_eq, var, simplifier, *opts, ctx),
        |description, next_eq, substeps: Option<Vec<SolveSubStep>>| {
            let step = medium_step(description, next_eq);
            if let Some(substeps) = substeps {
                step.with_substeps(substeps)
            } else {
                step
            }
        },
        |description, next_eq| SolveSubStep {
            description,
            equation_after: next_eq,
            importance: crate::ImportanceLevel::Low,
        },
        |_| {
            CasError::SolverError(
                "Inequalities with symbolic coefficients not yet supported".to_string(),
            )
        },
        |_| {
            ctx.emit_scope(cas_formatter::display_transforms::ScopeTag::Rule(
                "QuadraticFormula",
            ));
        },
    )
}
