//! Shared strategy-pipeline wrapper bound to the runtime solve context and the
//! concrete `apply_strategy` integration callback.

use cas_ast::{Equation, ExprId, SolutionSet};

/// Execute the default strategy pipeline while binding runtime-specific solve
/// options/context into the `apply_strategy` callback and reusing the default
/// soft-error and substitution-verification kernels.
#[allow(clippy::too_many_arguments)]
pub fn execute_strategy_pipeline_with_apply_strategy_runtime_ctx_and_default_verification_with_state<
    SState,
    TOptions,
    TCtx,
    FContainsVar,
    FCollectSteps,
    FContextRef,
    FContextMut,
    FRenderExpr,
    FApplyStrategyWithEquation,
    FSimplifyExpr,
    FAreEquivalent,
>(
    state: &mut SState,
    original_eq: &Equation,
    simplified_eq: &Equation,
    diff_simplified: ExprId,
    var: &str,
    opts: TOptions,
    ctx: &TCtx,
    domain_exclusions: &[ExprId],
    contains_var: FContainsVar,
    collect_steps: FCollectSteps,
    context_ref: FContextRef,
    context_mut: FContextMut,
    render_expr: FRenderExpr,
    mut apply_strategy_with_equation: FApplyStrategyWithEquation,
    simplify_expr: FSimplifyExpr,
    are_equivalent: FAreEquivalent,
) -> Result<
    (
        SolutionSet,
        Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
    ),
    crate::error_model::CasError,
>
where
    TOptions: Copy,
    FContainsVar: FnMut(&mut SState, ExprId, &str) -> bool,
    FCollectSteps: FnMut(&mut SState) -> bool,
    FContextRef: FnMut(&mut SState) -> &cas_ast::Context + Clone,
    FContextMut: FnMut(&mut SState) -> &mut cas_ast::Context,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
    FApplyStrategyWithEquation: FnMut(
        &mut SState,
        crate::strategy_order::SolveStrategyKind,
        &Equation,
        &str,
        TOptions,
        &TCtx,
    ) -> Option<
        Result<
            (
                SolutionSet,
                Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
            ),
            crate::error_model::CasError,
        >,
    >,
    FSimplifyExpr: FnMut(&mut SState, ExprId) -> ExprId,
    FAreEquivalent: FnMut(&mut SState, ExprId, ExprId) -> bool,
{
    let context_mut = std::cell::RefCell::new(context_mut);

    crate::solve_runtime_pipeline_strategy_execute_runtime::execute_strategy_pipeline_with_default_mappers_and_state(
        state,
        original_eq,
        simplified_eq,
        diff_simplified,
        var,
        domain_exclusions,
        contains_var,
        collect_steps,
        context_ref,
        |state| (context_mut.borrow_mut())(state),
        render_expr,
        |state, strategy_kind| {
            apply_strategy_with_equation(state, strategy_kind, simplified_eq, var, opts, ctx)
        },
        crate::solve_analysis::is_soft_strategy_error_by_parts::<crate::error_model::CasError>,
        |state, equation, solve_var, solution| {
            crate::verify_substitution::substitute_equation_sides(
                (context_mut.borrow_mut())(state),
                equation,
                solve_var,
                solution,
            )
        },
        simplify_expr,
        are_equivalent,
    )
}
