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
