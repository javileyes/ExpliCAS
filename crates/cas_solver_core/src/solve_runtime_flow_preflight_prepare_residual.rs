use cas_ast::{Equation, ExprId};

/// Prepare one equation for strategy dispatch using default residual-candidate
/// acceptance (`accept when smaller or when variable is eliminated`).
#[allow(clippy::too_many_arguments)]
pub fn prepare_equation_for_strategy_with_default_residual_acceptance_and_state<
    SState,
    FContainsVar,
    FSimplifyForSolve,
    FRecomposePowQuotient,
    FStructuralRewrite,
    FSemanticRewrite,
    FBuildDifference,
    FExpandAlgebraic,
    FExpandTrig,
    FContext,
    FZeroExpr,
>(
    state: &mut SState,
    equation: &Equation,
    var: &str,
    contains_var: FContainsVar,
    simplify_for_solve: FSimplifyForSolve,
    try_recompose_pow_quotient: FRecomposePowQuotient,
    structural_rewrite: FStructuralRewrite,
    semantic_rewrite: FSemanticRewrite,
    build_difference: FBuildDifference,
    expand_algebraic: FExpandAlgebraic,
    expand_trig: FExpandTrig,
    mut context: FContext,
    zero_expr: FZeroExpr,
) -> crate::solve_analysis::PreparedEquationResidual
where
    FContainsVar: FnMut(&mut SState, ExprId, &str) -> bool,
    FSimplifyForSolve: FnMut(&mut SState, ExprId) -> ExprId,
    FRecomposePowQuotient: FnMut(&mut SState, ExprId) -> Option<ExprId>,
    FStructuralRewrite: FnMut(&mut SState, ExprId, ExprId) -> Option<(ExprId, ExprId)>,
    FSemanticRewrite: FnMut(&mut SState, ExprId, ExprId) -> Option<(ExprId, ExprId)>,
    FBuildDifference: FnMut(&mut SState, ExprId, ExprId) -> ExprId,
    FExpandAlgebraic: FnMut(&mut SState, ExprId) -> ExprId,
    FExpandTrig: FnMut(&mut SState, ExprId) -> ExprId,
    FContext: FnMut(&mut SState) -> &cas_ast::Context,
    FZeroExpr: FnMut(&mut SState) -> ExprId,
{
    crate::solve_analysis::prepare_equation_for_strategy_with_state(
        state,
        equation,
        var,
        contains_var,
        simplify_for_solve,
        try_recompose_pow_quotient,
        structural_rewrite,
        semantic_rewrite,
        build_difference,
        expand_algebraic,
        expand_trig,
        |state, current, candidate, var_name| {
            crate::solve_analysis::accept_residual_rewrite_candidate(
                context(state),
                current,
                candidate,
                var_name,
            )
        },
        zero_expr,
    )
}
