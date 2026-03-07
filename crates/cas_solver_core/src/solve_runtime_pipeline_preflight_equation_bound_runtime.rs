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

/// Prepare one equation for strategy execution using only
/// [`RuntimeSolveAdapterState`] plus the default structural recomposition.
pub fn prepare_equation_for_strategy_with_adapter_state_and_default_structural_recompose<SState>(
    state: &mut SState,
    equation: &Equation,
    var: &str,
) -> crate::solve_analysis::PreparedEquationResidual
where
    SState: crate::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState,
{
    prepare_equation_for_strategy_with_runtime_context_and_default_structural_recompose_and_cancel_with_state(
        state,
        equation,
        var,
        crate::solve_runtime_adapter_state_runtime::simplifier_contains_var,
        crate::solve_runtime_adapter_state_runtime::simplifier_simplify_for_solve,
        crate::solve_runtime_adapter_state_runtime::simplifier_cancel_additive_terms_semantic,
        crate::solve_runtime_adapter_state_runtime::simplifier_expand_expr,
        crate::solve_runtime_adapter_state_runtime::simplifier_expand_full_expr,
        crate::solve_runtime_adapter_state_runtime::simplifier_context,
        crate::solve_runtime_adapter_state_runtime::simplifier_context_mut,
        crate::solve_runtime_adapter_state_runtime::simplifier_zero_expr,
    )
}
