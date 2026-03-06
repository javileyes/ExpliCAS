//! Shared pre-strategy equation preparation bound to the current runtime
//! context/difference construction model.

use cas_ast::{Equation, ExprId};

/// Prepare one equation for strategy execution using the runtime context model,
/// the default structural `lhs - rhs` recomposition, and a caller-provided
/// semantic cancellation fallback.
#[allow(clippy::too_many_arguments)]
pub fn prepare_equation_for_strategy_with_runtime_context_and_default_structural_recompose_and_cancel_with_state<
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
    crate::solve_runtime_pipeline_preflight_equation_runtime::prepare_equation_for_strategy_with_default_structural_recompose_and_cancel_with_state(
        state,
        equation,
        var,
        contains_var,
        simplify_for_solve,
        cancel_terms,
        expand_structural,
        expand_full,
        context_ref,
        context_mut,
        zero_expr,
    )
}
