//! Strategy-pipeline runtime wrappers extracted from `solve_runtime_flow`.

use cas_ast::{Equation, ExprId, SolutionSet};

/// Resolve a variable-eliminated residual with domain exclusions using the
/// default solve-analysis resolver and runtime-provided rendering/step mapping.
#[allow(clippy::too_many_arguments)]
pub fn resolve_var_eliminated_residual_with_default_exclusion_policy_and_unified_step_mapper_with_state<
    T,
    S,
    FCollectSteps,
    FContextMut,
    FRenderExpr,
    FMapStep,
>(
    state: &mut T,
    residual: ExprId,
    var: &str,
    domain_exclusions: &[ExprId],
    mut collect_steps: FCollectSteps,
    mut context_mut: FContextMut,
    render_expr: FRenderExpr,
    map_step: FMapStep,
) -> (SolutionSet, Vec<S>)
where
    FCollectSteps: FnMut(&mut T) -> bool,
    FContextMut: FnMut(&mut T) -> &mut cas_ast::Context,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FMapStep: FnMut(String, Equation) -> S,
{
    let include_item = collect_steps(state);
    crate::solve_analysis::resolve_var_eliminated_residual_with_exclusions(
        context_mut(state),
        residual,
        var,
        include_item,
        domain_exclusions,
        |ctx, expr| render_expr(ctx, expr),
        map_step,
    )
}

/// Resolve discrete strategy candidates against an equation using:
/// - symbolic-root passthrough policy, and
/// - substitution-based verification for numeric roots.
#[allow(clippy::too_many_arguments)]
pub fn resolve_discrete_strategy_result_against_equation_with_default_symbolic_and_substitution_verification_with_state<
    T,
    S,
    FContextRef,
    FSubstituteSides,
    FSimplifyExpr,
    FAreEquivalent,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    solutions: Vec<ExprId>,
    steps: Vec<S>,
    mut context_ref: FContextRef,
    mut substitute_sides: FSubstituteSides,
    mut simplify_expr: FSimplifyExpr,
    mut are_equivalent: FAreEquivalent,
) -> (SolutionSet, Vec<S>)
where
    FContextRef: FnMut(&mut T) -> &cas_ast::Context,
    FSubstituteSides: FnMut(&mut T, &Equation, &str, ExprId) -> (ExprId, ExprId),
    FSimplifyExpr: FnMut(&mut T, ExprId) -> ExprId,
    FAreEquivalent: FnMut(&mut T, ExprId, ExprId) -> bool,
{
    crate::solve_analysis::resolve_discrete_strategy_result_against_equation_with_state(
        state,
        equation,
        var,
        solutions,
        steps,
        |state, solution| crate::solve_analysis::is_symbolic_expr(context_ref(state), solution),
        |state, equation, solve_var, solution| {
            crate::verify_substitution::verify_solution_with_state(
                state,
                equation,
                solve_var,
                solution,
                |state, equation, solve_var, candidate| {
                    substitute_sides(state, equation, solve_var, candidate)
                },
                |state, expr| simplify_expr(state, expr),
                |state, lhs, rhs| are_equivalent(state, lhs, rhs),
            )
        },
    )
}

/// Execute solve strategy dispatch with the default strategy order and
/// per-strategy verification policy.
///
/// Callers provide:
/// - variable-presence check and residual fast-path resolver,
/// - cycle-guard entry,
/// - strategy application for each [`crate::strategy_order::SolveStrategyKind`],
/// - soft-error classifier,
/// - discrete-candidate resolution against `(equation, var)`.
#[allow(clippy::too_many_arguments)]
pub fn execute_default_strategy_order_pipeline_with_state<
    SState,
    S,
    E,
    Guard,
    FContainsVar,
    FResolveVarEliminated,
    FEnterCycle,
    FApplyStrategy,
    FSoftError,
    FResolveDiscreteAgainstEquation,
>(
    state: &mut SState,
    original_equation: &Equation,
    normalized_equation: &Equation,
    residual: ExprId,
    var: &str,
    contains_var: FContainsVar,
    resolve_var_eliminated: FResolveVarEliminated,
    enter_cycle: FEnterCycle,
    mut apply_strategy: FApplyStrategy,
    is_soft_error: FSoftError,
    mut resolve_discrete_against_equation: FResolveDiscreteAgainstEquation,
    no_solution_error: E,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FContainsVar: FnMut(&mut SState, ExprId, &str) -> bool,
    FResolveVarEliminated: FnMut(&mut SState, ExprId, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FEnterCycle: FnMut(&mut SState, &Equation, &str) -> Result<Guard, E>,
    FApplyStrategy: FnMut(
        &mut SState,
        crate::strategy_order::SolveStrategyKind,
    ) -> Option<Result<(SolutionSet, Vec<S>), E>>,
    FSoftError: FnMut(&E) -> bool,
    FResolveDiscreteAgainstEquation:
        FnMut(&mut SState, &Equation, &str, Vec<ExprId>, Vec<S>) -> (SolutionSet, Vec<S>),
{
    let strategy_order = crate::strategy_order::default_solve_strategy_order();
    crate::solve_analysis::execute_prepared_equation_strategy_pipeline_with_state(
        state,
        normalized_equation,
        residual,
        var,
        strategy_order,
        contains_var,
        resolve_var_eliminated,
        enter_cycle,
        |state, strategy_kind| {
            let should_verify = crate::strategy_order::strategy_should_verify(*strategy_kind);
            let attempt = apply_strategy(state, *strategy_kind);
            (attempt, should_verify)
        },
        is_soft_error,
        |state, solutions, steps| {
            resolve_discrete_against_equation(state, original_equation, var, solutions, steps)
        },
        no_solution_error,
    )
}

/// Execute strategy pipeline with default var-elimination and discrete-result
/// resolvers, while keeping strategy execution and cycle guards runtime-defined.
#[allow(clippy::too_many_arguments)]
pub fn execute_default_strategy_order_pipeline_with_default_var_elimination_and_discrete_resolution_with_state<
    SState,
    S,
    E,
    Guard,
    FContainsVar,
    FCollectSteps,
    FContextRef,
    FContextMut,
    FRenderExpr,
    FMapStep,
    FEnterCycle,
    FApplyStrategy,
    FSoftError,
    FSubstituteSides,
    FSimplifyExpr,
    FAreEquivalent,
>(
    state: &mut SState,
    original_equation: &Equation,
    normalized_equation: &Equation,
    residual: ExprId,
    var: &str,
    domain_exclusions: &[ExprId],
    mut contains_var: FContainsVar,
    mut collect_steps: FCollectSteps,
    mut context_ref: FContextRef,
    mut context_mut: FContextMut,
    render_expr: FRenderExpr,
    mut map_step: FMapStep,
    enter_cycle: FEnterCycle,
    apply_strategy: FApplyStrategy,
    is_soft_error: FSoftError,
    mut substitute_sides: FSubstituteSides,
    mut simplify_expr: FSimplifyExpr,
    mut are_equivalent: FAreEquivalent,
    no_solution_error: E,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FContainsVar: FnMut(&mut SState, ExprId, &str) -> bool,
    FCollectSteps: FnMut(&mut SState) -> bool,
    FContextRef: FnMut(&mut SState) -> &cas_ast::Context,
    FContextMut: FnMut(&mut SState) -> &mut cas_ast::Context,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FMapStep: FnMut(String, Equation) -> S,
    FEnterCycle: FnMut(&mut SState, &Equation, &str) -> Result<Guard, E>,
    FApplyStrategy: FnMut(
        &mut SState,
        crate::strategy_order::SolveStrategyKind,
    ) -> Option<Result<(SolutionSet, Vec<S>), E>>,
    FSoftError: FnMut(&E) -> bool,
    FSubstituteSides: FnMut(&mut SState, &Equation, &str, ExprId) -> (ExprId, ExprId),
    FSimplifyExpr: FnMut(&mut SState, ExprId) -> ExprId,
    FAreEquivalent: FnMut(&mut SState, ExprId, ExprId) -> bool,
{
    let render_expr = &render_expr;

    execute_default_strategy_order_pipeline_with_state(
        state,
        original_equation,
        normalized_equation,
        residual,
        var,
        |state, expr, var_name| contains_var(state, expr, var_name),
        |state, residual, var_name| {
            Ok(resolve_var_eliminated_residual_with_default_exclusion_policy_and_unified_step_mapper_with_state(
                state,
                residual,
                var_name,
                domain_exclusions,
                |state| collect_steps(state),
                |state| context_mut(state),
                |ctx, expr| render_expr(ctx, expr),
                &mut map_step,
            ))
        },
        enter_cycle,
        apply_strategy,
        is_soft_error,
        |state, equation, var_name, solutions, steps| {
            resolve_discrete_strategy_result_against_equation_with_default_symbolic_and_substitution_verification_with_state(
                state,
                equation,
                var_name,
                solutions,
                steps,
                |state| context_ref(state),
                |state, equation, solve_var, solution| {
                    substitute_sides(state, equation, solve_var, solution)
                },
                |state, expr| simplify_expr(state, expr),
                |state, lhs, rhs| are_equivalent(state, lhs, rhs),
            )
        },
        no_solution_error,
    )
}

/// Execute strategy pipeline with:
/// - default equation-fingerprint cycle guard,
/// - default var-elimination and discrete-result resolvers,
///
/// Strategy execution and verification kernels remain runtime-defined.
#[allow(clippy::too_many_arguments)]
pub fn execute_default_strategy_order_pipeline_with_default_cycle_guard_and_default_var_elimination_and_discrete_resolution_with_state<
    SState,
    S,
    E,
    FContainsVar,
    FCollectSteps,
    FContextRef,
    FContextRefForCycle,
    FContextMut,
    FRenderExpr,
    FMapStep,
    FMapCycleError,
    FApplyStrategy,
    FSoftError,
    FSubstituteSides,
    FSimplifyExpr,
    FAreEquivalent,
>(
    state: &mut SState,
    original_equation: &Equation,
    normalized_equation: &Equation,
    residual: ExprId,
    var: &str,
    domain_exclusions: &[ExprId],
    contains_var: FContainsVar,
    collect_steps: FCollectSteps,
    context_ref: FContextRef,
    context_ref_for_cycle: FContextRefForCycle,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    map_step: FMapStep,
    mut map_cycle_error: FMapCycleError,
    apply_strategy: FApplyStrategy,
    is_soft_error: FSoftError,
    substitute_sides: FSubstituteSides,
    simplify_expr: FSimplifyExpr,
    are_equivalent: FAreEquivalent,
    no_solution_error: E,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FContainsVar: FnMut(&mut SState, ExprId, &str) -> bool,
    FCollectSteps: FnMut(&mut SState) -> bool,
    FContextRef: FnMut(&mut SState) -> &cas_ast::Context,
    FContextRefForCycle: FnMut(&mut SState) -> &cas_ast::Context,
    FContextMut: FnMut(&mut SState) -> &mut cas_ast::Context,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FMapStep: FnMut(String, Equation) -> S,
    FMapCycleError: FnMut() -> E,
    FApplyStrategy: FnMut(
        &mut SState,
        crate::strategy_order::SolveStrategyKind,
    ) -> Option<Result<(SolutionSet, Vec<S>), E>>,
    FSoftError: FnMut(&E) -> bool,
    FSubstituteSides: FnMut(&mut SState, &Equation, &str, ExprId) -> (ExprId, ExprId),
    FSimplifyExpr: FnMut(&mut SState, ExprId) -> ExprId,
    FAreEquivalent: FnMut(&mut SState, ExprId, ExprId) -> bool,
{
    let context_ref_for_cycle = std::cell::RefCell::new(context_ref_for_cycle);
    execute_default_strategy_order_pipeline_with_default_var_elimination_and_discrete_resolution_with_state(
        state,
        original_equation,
        normalized_equation,
        residual,
        var,
        domain_exclusions,
        contains_var,
        collect_steps,
        context_ref,
        context_mut,
        render_expr,
        map_step,
        |state, equation, var_name| {
            crate::solve_analysis::try_enter_equation_cycle_guard_with_error(
                (context_ref_for_cycle.borrow_mut())(state),
                equation,
                var_name,
                &mut map_cycle_error,
            )
        },
        apply_strategy,
        is_soft_error,
        substitute_sides,
        simplify_expr,
        are_equivalent,
        no_solution_error,
    )
}
