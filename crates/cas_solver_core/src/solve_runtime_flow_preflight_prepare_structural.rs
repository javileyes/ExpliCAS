use super::solve_runtime_flow_preflight_prepare_residual::prepare_equation_for_strategy_with_default_residual_acceptance_and_state;
use cas_ast::{Equation, ExprId};

/// Prepare one equation for strategy dispatch using:
/// - default residual acceptance policy,
/// - default `pow`-quotient recomposition,
/// - default structural additive cancellation.
///
/// Runtime crates only provide semantic fallback rewrite and runtime kernels.
#[allow(clippy::too_many_arguments)]
pub fn prepare_equation_for_strategy_with_default_structural_recompose_and_cancel_and_default_residual_acceptance_with_state<
    SState,
    FContainsVar,
    FSimplifyForSolve,
    FSemanticRewrite,
    FBuildDifference,
    FExpandAlgebraic,
    FExpandTrig,
    FContext,
    FContextMut,
    FZeroExpr,
>(
    state: &mut SState,
    equation: &Equation,
    var: &str,
    contains_var: FContainsVar,
    simplify_for_solve: FSimplifyForSolve,
    semantic_rewrite: FSemanticRewrite,
    build_difference: FBuildDifference,
    expand_algebraic: FExpandAlgebraic,
    expand_trig: FExpandTrig,
    context: FContext,
    context_mut: FContextMut,
    zero_expr: FZeroExpr,
) -> crate::solve_analysis::PreparedEquationResidual
where
    FContainsVar: FnMut(&mut SState, ExprId, &str) -> bool,
    FSimplifyForSolve: FnMut(&mut SState, ExprId) -> ExprId,
    FSemanticRewrite: FnMut(&mut SState, ExprId, ExprId) -> Option<(ExprId, ExprId)>,
    FBuildDifference: FnMut(&mut SState, ExprId, ExprId) -> ExprId,
    FExpandAlgebraic: FnMut(&mut SState, ExprId) -> ExprId,
    FExpandTrig: FnMut(&mut SState, ExprId) -> ExprId,
    FContext: FnMut(&mut SState) -> &cas_ast::Context,
    FContextMut: FnMut(&mut SState) -> &mut cas_ast::Context,
    FZeroExpr: FnMut(&mut SState) -> ExprId,
{
    let context_mut = std::cell::RefCell::new(context_mut);

    prepare_equation_for_strategy_with_default_residual_acceptance_and_state(
        state,
        equation,
        var,
        contains_var,
        simplify_for_solve,
        |state, expr| {
            crate::isolation_utils::try_recompose_pow_quotient(
                (context_mut.borrow_mut())(state),
                expr,
            )
        },
        |state, lhs, rhs| {
            crate::cancel_common_terms::cancel_common_additive_terms(
                (context_mut.borrow_mut())(state),
                lhs,
                rhs,
            )
            .map(|rewrite| (rewrite.new_lhs, rewrite.new_rhs))
        },
        semantic_rewrite,
        build_difference,
        expand_algebraic,
        expand_trig,
        context,
        zero_expr,
    )
}
