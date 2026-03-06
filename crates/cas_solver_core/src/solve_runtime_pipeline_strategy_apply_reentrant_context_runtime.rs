//! Shared strategy-apply wrapper that binds recursive runtime entrypoints to
//! the current solver options/context.

use cas_ast::{Equation, ExprId, RelOp, SolutionSet};

/// Apply one strategy while wiring recursive `solve_inner` / isolation
/// entrypoints through the current runtime solve context and solver options.
#[allow(clippy::too_many_arguments)]
pub fn apply_strategy_with_runtime_ctx_and_reentrant_entrypoints_and_state<
    T,
    FCollectSteps,
    FContextRef,
    FContextMut,
    FSetCollect,
    FSimplifyExpr,
    FExpandExpr,
    FRenderExpr,
    FRenderExprFromCtx,
    FSolveReentrant,
    FIsolateReentrant,
    FProvePositive,
>(
    state: &mut T,
    kind: crate::strategy_order::SolveStrategyKind,
    equation: &Equation,
    var: &str,
    opts: crate::solver_options::SolverOptions,
    ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    collect_steps: FCollectSteps,
    context_ref: FContextRef,
    context_mut: FContextMut,
    set_collecting: FSetCollect,
    simplify_expr: FSimplifyExpr,
    expand_expr: FExpandExpr,
    render_expr: FRenderExpr,
    render_expr_from_ctx: FRenderExprFromCtx,
    solve_reentrant: FSolveReentrant,
    isolate_reentrant: FIsolateReentrant,
    prove_positive: FProvePositive,
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
    FContextRef: Fn(&mut T) -> &cas_ast::Context + Clone,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FSetCollect: FnMut(&mut T, bool),
    FSimplifyExpr: FnMut(&mut T, ExprId) -> ExprId,
    FExpandExpr: FnMut(&mut T, ExprId) -> ExprId,
    FRenderExpr: FnMut(&mut T, ExprId) -> String,
    FRenderExprFromCtx: Fn(&cas_ast::Context, ExprId) -> String,
    FSolveReentrant: FnMut(
        &Equation,
        &str,
        &mut T,
        crate::solver_options::SolverOptions,
        &crate::solve_runtime_types::RuntimeSolveCtx,
    ) -> Result<
        (
            SolutionSet,
            Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
        ),
        crate::error_model::CasError,
    >,
    FIsolateReentrant: FnMut(
        ExprId,
        ExprId,
        RelOp,
        &str,
        &mut T,
        crate::solver_options::SolverOptions,
        &crate::solve_runtime_types::RuntimeSolveCtx,
    ) -> Result<
        (
            SolutionSet,
            Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
        ),
        crate::error_model::CasError,
    >,
    FProvePositive: FnMut(
        &cas_ast::Context,
        ExprId,
        crate::value_domain::ValueDomain,
    ) -> crate::domain_proof::Proof,
{
    let solve_reentrant = std::cell::RefCell::new(solve_reentrant);
    let isolate_reentrant = std::cell::RefCell::new(isolate_reentrant);

    crate::solve_runtime_pipeline_strategy_apply_context_runtime::apply_strategy_with_runtime_ctx_and_options_and_state(
        state,
        kind,
        equation,
        var,
        opts,
        ctx,
        collect_steps,
        context_ref,
        context_mut,
        set_collecting,
        simplify_expr,
        expand_expr,
        render_expr,
        render_expr_from_ctx,
        |state, recursive_equation, solve_var| {
            (solve_reentrant.borrow_mut())(recursive_equation, solve_var, state, opts, ctx)
        },
        |state, next_eq, solve_var| {
            (isolate_reentrant.borrow_mut())(
                next_eq.lhs,
                next_eq.rhs,
                next_eq.op.clone(),
                solve_var,
                state,
                opts,
                ctx,
            )
        },
        prove_positive,
    )
}
