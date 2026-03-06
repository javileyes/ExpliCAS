//! Shared runtime adapter for one-strategy dispatch application.

use cas_ast::{Equation, ExprId, SolutionSet};

/// Apply one strategy kind with shared default step/error mappers.
#[allow(clippy::too_many_arguments)]
pub fn apply_strategy_with_default_mappers_and_state<
    T,
    FCollectSteps,
    FContextRef,
    FContextMut,
    FSetCollect,
    FSimplifyExpr,
    FExpandExpr,
    FRenderExpr,
    FRenderExprFromCtx,
    FSolveEquation,
    FIsolateEquation,
    FClassifyLogSolve,
    FNoteAssumption,
    FOnQuadraticScope,
>(
    state: &mut T,
    kind: crate::strategy_order::SolveStrategyKind,
    equation: &Equation,
    var: &str,
    is_real_only: bool,
    mode: crate::log_domain::DomainModeKind,
    wildcard_scope: bool,
    collect_steps: FCollectSteps,
    context_ref: FContextRef,
    context_mut: FContextMut,
    set_collecting: FSetCollect,
    simplify_expr: FSimplifyExpr,
    expand_expr: FExpandExpr,
    render_expr: FRenderExpr,
    render_expr_from_ctx: FRenderExprFromCtx,
    solve_equation: FSolveEquation,
    isolate_equation: FIsolateEquation,
    classify_log_solve: FClassifyLogSolve,
    note_assumption: FNoteAssumption,
    on_quadratic_scope: FOnQuadraticScope,
) -> Option<
    Result<
        (
            SolutionSet,
            Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
        ),
        crate::error_model::CasError,
    >,
>
where
    FCollectSteps: FnMut(&mut T) -> bool,
    FContextRef: Fn(&mut T) -> &cas_ast::Context,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FSetCollect: FnMut(&mut T, bool),
    FSimplifyExpr: FnMut(&mut T, ExprId) -> ExprId,
    FExpandExpr: FnMut(&mut T, ExprId) -> ExprId,
    FRenderExpr: FnMut(&mut T, ExprId) -> String,
    FRenderExprFromCtx: Fn(&cas_ast::Context, ExprId) -> String,
    FSolveEquation: FnMut(
        &mut T,
        &Equation,
        &str,
    ) -> Result<
        (
            SolutionSet,
            Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
        ),
        crate::error_model::CasError,
    >,
    FIsolateEquation: FnMut(
        &mut T,
        &Equation,
        &str,
    ) -> Result<
        (
            SolutionSet,
            Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
        ),
        crate::error_model::CasError,
    >,
    FClassifyLogSolve:
        FnMut(&cas_ast::Context, ExprId, ExprId) -> crate::log_domain::LogSolveDecision,
    FNoteAssumption: FnMut(&mut T, crate::unwrap_plan::LogLinearAssumptionRecord),
    FOnQuadraticScope: FnMut(&mut T),
{
    crate::solve_runtime_flow::apply_strategy_kind_with_default_kernels_and_default_step_and_error_mappers_with_state(
        state,
        kind,
        equation,
        var,
        is_real_only,
        mode,
        wildcard_scope,
        collect_steps,
        context_ref,
        context_mut,
        set_collecting,
        simplify_expr,
        expand_expr,
        render_expr,
        render_expr_from_ctx,
        solve_equation,
        isolate_equation,
        classify_log_solve,
        note_assumption,
        crate::solve_runtime_mapping::medium_step,
        crate::solve_runtime_mapping::attach_substeps,
        crate::solve_runtime_mapping::low_substep,
        crate::solve_runtime_mapping::map_symbolic_inequalities_not_supported_error,
        crate::solve_runtime_mapping::map_variable_not_found_solver_error,
        on_quadratic_scope,
    )
}
