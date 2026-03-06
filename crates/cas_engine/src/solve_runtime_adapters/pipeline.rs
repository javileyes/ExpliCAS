use super::*;
use crate::engine::Simplifier;
use crate::error::CasError;
use cas_ast::{Equation, Expr, SolutionSet};
use cas_solver_core::solve_analysis::{PreflightContext, PreparedEquationResidual};
use cas_solver_core::strategy_order::SolveStrategyKind;

pub(crate) fn build_solve_preflight_state(
    simplifier: &Simplifier,
    eq: &Equation,
    var: &str,
    value_domain: crate::ValueDomain,
    parent_ctx: &SolveCtx,
) -> PreflightContext<SolveCtx> {
    cas_solver_core::solve_runtime_flow::analyze_preflight_and_fork_context_with_existing_condition_derivation(
        &simplifier.context,
        eq,
        var,
        value_domain,
        parent_ctx,
        |expr, eval_domain| {
            crate::infer_implicit_domain(&simplifier.context, expr, eval_domain)
                .conditions()
                .iter()
                .cloned()
                .collect::<Vec<_>>()
        },
        crate::ImplicitDomain::empty,
        |domain, cond| {
            domain.conditions_mut().insert(cond);
        },
        |lhs, rhs, domain, eval_domain| {
            crate::derive_requires_from_equation(
                &simplifier.context,
                lhs,
                rhs,
                domain,
                eval_domain,
            )
        },
        SolveDomainEnv::new(),
        |domain_env, cond| {
            domain_env.required.conditions_mut().insert(cond.clone());
        },
    )
}

pub(crate) fn prepare_equation_for_strategy(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
) -> PreparedEquationResidual {
    cas_solver_core::solve_runtime_flow::prepare_equation_for_strategy_with_default_structural_recompose_and_cancel_and_default_residual_acceptance_with_state(
        simplifier,
        equation,
        var,
        simplifier_contains_var,
        |state, expr| state.simplify_for_solve(expr),
        |state, lhs, rhs| {
            crate::cancel_common_terms_runtime::cancel_additive_terms_semantic(state, lhs, rhs)
                .map(|rewrite| (rewrite.new_lhs, rewrite.new_rhs))
        },
        |state, lhs, rhs| state.context.add(Expr::Sub(lhs, rhs)),
        simplifier_expand_expr,
        |state, expr| state.expand(expr).0,
        |state| &state.context,
        simplifier_context_mut,
        simplifier_zero_expr,
    )
}

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
