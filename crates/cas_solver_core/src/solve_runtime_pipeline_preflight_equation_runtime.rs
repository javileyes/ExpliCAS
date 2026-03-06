//! Shared runtime adapter for pre-strategy equation preparation.

use cas_ast::{Equation, Expr, ExprId};

/// Prepare equation for strategy execution using default structural
/// recomposition (`lhs - rhs`) and residual acceptance policy.
#[allow(clippy::too_many_arguments)]
pub fn prepare_equation_for_strategy_with_default_structural_recompose_and_cancel_with_state<
    SState,
    FContainsVar,
    FSimplifyForSolve,
    FCancelTerms,
    FExpandStructural,
    FExpandFull,
    FContextRef,
    FContextMut,
    FZeroExpr,
>(
    state: &mut SState,
    equation: &Equation,
    var: &str,
    contains_var: FContainsVar,
    simplify_for_solve: FSimplifyForSolve,
    cancel_terms: FCancelTerms,
    expand_structural: FExpandStructural,
    expand_full: FExpandFull,
    context_ref: FContextRef,
    context_mut: FContextMut,
    zero_expr: FZeroExpr,
) -> crate::solve_analysis::PreparedEquationResidual
where
    FContainsVar: FnMut(&mut SState, ExprId, &str) -> bool,
    FSimplifyForSolve: FnMut(&mut SState, ExprId) -> ExprId,
    FCancelTerms: FnMut(&mut SState, ExprId, ExprId) -> Option<(ExprId, ExprId)>,
    FExpandStructural: FnMut(&mut SState, ExprId) -> ExprId,
    FExpandFull: FnMut(&mut SState, ExprId) -> ExprId,
    FContextRef: FnMut(&mut SState) -> &cas_ast::Context,
    FContextMut: FnMut(&mut SState) -> &mut cas_ast::Context,
    FZeroExpr: FnMut(&mut SState) -> ExprId,
{
    let context_mut = std::cell::RefCell::new(context_mut);

    crate::solve_runtime_flow::prepare_equation_for_strategy_with_default_structural_recompose_and_cancel_and_default_residual_acceptance_with_state(
        state,
        equation,
        var,
        contains_var,
        simplify_for_solve,
        cancel_terms,
        |state, lhs, rhs| (context_mut.borrow_mut())(state).add(Expr::Sub(lhs, rhs)),
        expand_structural,
        expand_full,
        context_ref,
        |state| (context_mut.borrow_mut())(state),
        zero_expr,
    )
}
