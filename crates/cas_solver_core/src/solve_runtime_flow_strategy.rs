//! Strategy-dispatch and log-runtime wrappers extracted from `solve_runtime_flow_isolation`.

use cas_ast::{Equation, ExprId, SolutionSet};
use crate::solve_runtime_flow::{
    apply_collect_terms_strategy_with_default_kernels_and_state,
    apply_isolation_strategy_with_default_kernels_and_state,
    apply_quadratic_strategy_with_default_kernels_and_state,
    apply_rational_exponent_strategy_with_default_kernels_and_state,
    apply_rational_roots_strategy_with_default_kernels_and_state,
    apply_substitution_strategy_with_default_kernels_and_state,
    apply_unwrap_strategy_with_default_kernels_and_state,
};

#[allow(clippy::too_many_arguments)]
pub fn dispatch_solve_strategy_kind_with_runtime_handlers_with_state<
    T,
    R,
    FRationalExponent,
    FSubstitution,
    FUnwrap,
    FQuadratic,
    FRationalRoots,
    FCollectTerms,
    FIsolation,
>(
    state: &mut T,
    kind: crate::strategy_order::SolveStrategyKind,
    rational_exponent: FRationalExponent,
    substitution: FSubstitution,
    unwrap: FUnwrap,
    quadratic: FQuadratic,
    rational_roots: FRationalRoots,
    collect_terms: FCollectTerms,
    isolation: FIsolation,
) -> R
where
    FRationalExponent: FnOnce(&mut T) -> R,
    FSubstitution: FnOnce(&mut T) -> R,
    FUnwrap: FnOnce(&mut T) -> R,
    FQuadratic: FnOnce(&mut T) -> R,
    FRationalRoots: FnOnce(&mut T) -> R,
    FCollectTerms: FnOnce(&mut T) -> R,
    FIsolation: FnOnce(&mut T) -> R,
{
    crate::strategy_order::dispatch_solve_strategy_kind_with_state(
        state,
        kind,
        rational_exponent,
        substitution,
        unwrap,
        quadratic,
        rational_roots,
        collect_terms,
        isolation,
    )
}

/// Apply one solve strategy kind using the default strategy kernels and
/// runtime-provided callbacks for recursion, rendering, and diagnostics.
#[allow(clippy::too_many_arguments)]
pub fn apply_strategy_kind_with_default_kernels_and_state<
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
    FMapQuadraticStep,
    FMapSubstep,
    FMapPlanError,
    FVariableNotFoundError,
    FOnQuadraticCoefficientPathSolved,
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
    map_quadratic_step: FMapQuadraticStep,
    map_substep: FMapSubstep,
    map_plan_error: FMapPlanError,
    variable_not_found_error: FVariableNotFoundError,
    on_quadratic_coefficient_path_solved: FOnQuadraticCoefficientPathSolved,
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
    FMapQuadraticStep: FnMut(String, Equation, Option<Vec<SS>>) -> S,
    FMapSubstep: FnMut(String, Equation) -> SS,
    FMapPlanError: FnMut(crate::quadratic_formula::QuadraticCoefficientSolvePlanError) -> E,
    FVariableNotFoundError: FnMut(&str) -> E,
    FOnQuadraticCoefficientPathSolved: FnMut(&mut T),
{
    let collect_steps = std::cell::RefCell::new(collect_steps);
    let set_collecting = std::cell::RefCell::new(set_collecting);
    let simplify_expr = std::cell::RefCell::new(simplify_expr);
    let expand_expr = std::cell::RefCell::new(expand_expr);
    let render_expr = std::cell::RefCell::new(render_expr);
    let solve_equation = std::cell::RefCell::new(solve_equation);
    let isolate_equation = std::cell::RefCell::new(isolate_equation);
    let classify_log_solve = std::cell::RefCell::new(classify_log_solve);
    let note_assumption = std::cell::RefCell::new(note_assumption);
    let map_simple_step = std::cell::RefCell::new(map_simple_step);
    let map_quadratic_step = std::cell::RefCell::new(map_quadratic_step);
    let map_substep = std::cell::RefCell::new(map_substep);
    let map_plan_error = std::cell::RefCell::new(map_plan_error);
    let variable_not_found_error = std::cell::RefCell::new(variable_not_found_error);
    let on_quadratic_coefficient_path_solved =
        std::cell::RefCell::new(on_quadratic_coefficient_path_solved);

    dispatch_solve_strategy_kind_with_runtime_handlers_with_state(
        state,
        kind,
        |state| {
            apply_rational_exponent_strategy_with_default_kernels_and_state(
                state,
                equation,
                var,
                |state| (collect_steps.borrow_mut())(state),
                |state| context_mut(state),
                |state, expr| (simplify_expr.borrow_mut())(state, expr),
                |state, next_eq, solve_var| {
                    (solve_equation.borrow_mut())(state, next_eq, solve_var)
                },
                |description, next_eq| (map_simple_step.borrow_mut())(description, next_eq),
            )
        },
        |state| {
            apply_substitution_strategy_with_default_kernels_and_state(
                state,
                equation,
                var,
                |state| (collect_steps.borrow_mut())(state),
                |state| context_mut(state),
                |state, expr| (render_expr.borrow_mut())(state, expr),
                |state, next_eq, solve_var| {
                    (solve_equation.borrow_mut())(state, next_eq, solve_var)
                },
                |description, next_eq| (map_simple_step.borrow_mut())(description, next_eq),
            )
        },
        |state| {
            apply_unwrap_strategy_with_default_kernels_and_state(
                state,
                equation,
                var,
                |state| (collect_steps.borrow_mut())(state),
                |state| context_mut(state),
                mode,
                wildcard_scope,
                |core_ctx, base, other_side| {
                    (classify_log_solve.borrow_mut())(core_ctx, base, other_side)
                },
                |core_ctx, expr| render_expr_from_ctx(core_ctx, expr),
                |state, record| (note_assumption.borrow_mut())(state, record),
                |state, next_eq, solve_var| {
                    (solve_equation.borrow_mut())(state, next_eq, solve_var)
                },
                |description, next_eq| (map_simple_step.borrow_mut())(description, next_eq),
            )
        },
        |state| {
            apply_quadratic_strategy_with_default_kernels_and_state(
                state,
                equation,
                var,
                |state| (collect_steps.borrow_mut())(state),
                is_real_only,
                |state| context_ref(state),
                |state| context_mut(state),
                |state, collecting| (set_collecting.borrow_mut())(state, collecting),
                |state, expr| (simplify_expr.borrow_mut())(state, expr),
                |state, expr| (expand_expr.borrow_mut())(state, expr),
                |core_ctx, expr| render_expr_from_ctx(core_ctx, expr),
                |state, next_eq| (solve_equation.borrow_mut())(state, next_eq, var),
                |description, next_eq, substeps| {
                    (map_quadratic_step.borrow_mut())(description, next_eq, substeps)
                },
                |description, next_eq| (map_substep.borrow_mut())(description, next_eq),
                |plan_error| (map_plan_error.borrow_mut())(plan_error),
                |state| (on_quadratic_coefficient_path_solved.borrow_mut())(state),
            )
        },
        |state| {
            let solved = apply_rational_roots_strategy_with_default_kernels_and_state(
                state,
                equation,
                var,
                |state| (collect_steps.borrow_mut())(state),
                |state| context_mut(state),
                |state, expr| (simplify_expr.borrow_mut())(state, expr),
                |state, expr| (expand_expr.borrow_mut())(state, expr),
                |description, next_eq| (map_simple_step.borrow_mut())(description, next_eq),
            )?;
            Some(Ok(solved))
        },
        |state| {
            apply_collect_terms_strategy_with_default_kernels_and_state(
                state,
                equation,
                var,
                |state| (collect_steps.borrow_mut())(state),
                |state| context_mut(state),
                |state, expr| (simplify_expr.borrow_mut())(state, expr),
                |state, expr| (render_expr.borrow_mut())(state, expr),
                |state, next_eq, solve_var| {
                    (solve_equation.borrow_mut())(state, next_eq, solve_var)
                },
                |description, next_eq| (map_simple_step.borrow_mut())(description, next_eq),
            )
        },
        |state| {
            apply_isolation_strategy_with_default_kernels_and_state(
                state,
                equation,
                var,
                |state| (collect_steps.borrow_mut())(state),
                |state| context_ref(state),
                |state, next_eq, solve_var| {
                    (isolate_equation.borrow_mut())(state, next_eq, solve_var)
                },
                |description, next_eq| (map_simple_step.borrow_mut())(description, next_eq),
                |missing_var| (variable_not_found_error.borrow_mut())(missing_var),
            )
        },
    )
}

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

/// Classify a logarithmic solve decision using:
/// - domain environment (`base > 0`, `rhs > 0`) facts,
/// - runtime value-domain configuration,
/// - runtime positive prover callback returning [`crate::domain_proof::Proof`].
pub fn classify_log_solve_with_domain_env_and_runtime_positive_prover<R, FProvePositive>(
    ctx: &cas_ast::Context,
    base: ExprId,
    rhs: ExprId,
    value_domain: crate::value_domain::ValueDomain,
    mode: crate::log_domain::DomainModeKind,
    env: &crate::domain_env::SolveDomainEnv<R>,
    mut prove_positive: FProvePositive,
) -> crate::log_domain::LogSolveDecision
where
    R: crate::domain_env::RequiredDomainSet,
    FProvePositive: FnMut(
        &cas_ast::Context,
        ExprId,
        crate::value_domain::ValueDomain,
    ) -> crate::domain_proof::Proof,
{
    crate::log_domain::classify_log_solve_with_env_and_tri_prover(
        ctx,
        base,
        rhs,
        value_domain == crate::value_domain::ValueDomain::RealOnly,
        mode,
        env,
        |core_ctx, expr| {
            crate::predicate_proofs::prove_positive_core_with(
                core_ctx,
                expr,
                value_domain,
                |proof_ctx, proof_expr, vd| prove_positive(proof_ctx, proof_expr, vd),
            )
        },
    )
}

/// Map one logarithmic assumption into a runtime assumption-event sink.
pub fn note_log_assumption_with_runtime_sink<FSink>(
    ctx: &cas_ast::Context,
    base: ExprId,
    rhs: ExprId,
    assumption: crate::log_domain::LogAssumption,
    mut sink: FSink,
) where
    FSink: FnMut(crate::assumption_model::AssumptionEvent),
{
    sink(crate::assumption_model::assumption_event_from_log_assumption(ctx, assumption, base, rhs));
}

/// Map one blocked logarithmic hint into a runtime blocked-hint sink.
pub fn note_log_blocked_hint_with_runtime_sink<FSink>(
    ctx: &cas_ast::Context,
    hint: crate::solve_outcome::LogBlockedHintRecord,
    mut sink: FSink,
) where
    FSink: FnMut(crate::blocked_hint::BlockedHint),
{
    sink(crate::assumption_model::map_log_blocked_hint(ctx, hint));
}
