use super::solve_runtime_flow_strategy_dispatch_apply_core::apply_strategy_kind_with_default_kernels_and_state;
use cas_ast::{Equation, ExprId, SolutionSet};

/// Apply one strategy kind using default kernels plus shared wiring for:
/// - default step/substep composition,
/// - quadratic plan error mapping,
/// - variable-not-found mapping,
/// - quadratic-scope side-effects.
#[allow(clippy::too_many_arguments)]
pub fn apply_strategy_kind_with_default_kernels_and_default_step_and_error_mappers_with_state<
    T,
    S,
    SS,
    E,
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
    FMapSimpleStep,
    FAttachSubsteps,
    FMapSubstep,
    FMapPlanError,
    FMapVariableNotFound,
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
    map_simple_step: FMapSimpleStep,
    attach_substeps: FAttachSubsteps,
    map_substep: FMapSubstep,
    map_plan_error: FMapPlanError,
    map_variable_not_found: FMapVariableNotFound,
    on_quadratic_scope: FOnQuadraticScope,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    SS: Clone,
    FCollectSteps: FnMut(&mut T) -> bool,
    FContextRef: Fn(&mut T) -> &cas_ast::Context,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FSetCollect: FnMut(&mut T, bool),
    FSimplifyExpr: FnMut(&mut T, ExprId) -> ExprId,
    FExpandExpr: FnMut(&mut T, ExprId) -> ExprId,
    FRenderExpr: FnMut(&mut T, ExprId) -> String,
    FRenderExprFromCtx: Fn(&cas_ast::Context, ExprId) -> String,
    FSolveEquation: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FIsolateEquation: FnMut(&mut T, &Equation, &str) -> Result<(SolutionSet, Vec<S>), E>,
    FClassifyLogSolve:
        FnMut(&cas_ast::Context, ExprId, ExprId) -> crate::log_domain::LogSolveDecision,
    FNoteAssumption: FnMut(&mut T, crate::unwrap_plan::LogLinearAssumptionRecord),
    FMapSimpleStep: FnMut(String, Equation) -> S,
    FAttachSubsteps: FnMut(S, Vec<SS>) -> S,
    FMapSubstep: FnMut(String, Equation) -> SS,
    FMapPlanError: FnMut(crate::quadratic_formula::QuadraticCoefficientSolvePlanError) -> E,
    FMapVariableNotFound: FnMut(&str) -> E,
    FOnQuadraticScope: FnMut(&mut T),
{
    let map_simple_step = std::cell::RefCell::new(map_simple_step);
    let attach_substeps = std::cell::RefCell::new(attach_substeps);
    let map_substep = std::cell::RefCell::new(map_substep);
    let map_plan_error = std::cell::RefCell::new(map_plan_error);
    let map_variable_not_found = std::cell::RefCell::new(map_variable_not_found);
    let on_quadratic_scope = std::cell::RefCell::new(on_quadratic_scope);

    apply_strategy_kind_with_default_kernels_and_state(
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
        |description, next_eq| (map_simple_step.borrow_mut())(description, next_eq),
        |description, next_eq, substeps: Option<Vec<SS>>| {
            let step = (map_simple_step.borrow_mut())(description, next_eq);
            if let Some(substeps) = substeps {
                (attach_substeps.borrow_mut())(step, substeps)
            } else {
                step
            }
        },
        |description, next_eq| (map_substep.borrow_mut())(description, next_eq),
        |plan_error| (map_plan_error.borrow_mut())(plan_error),
        |missing_var| (map_variable_not_found.borrow_mut())(missing_var),
        |state| (on_quadratic_scope.borrow_mut())(state),
    )
}
