//! Shared isolation-dispatch wrapper that binds recursive runtime entrypoints
//! to the current solver options/context.

use cas_ast::symbol::SymbolId;
use cas_ast::{Equation, ExprId, RelOp, SolutionSet};

/// Dispatch isolation while wiring recursive `solve_inner` / isolation
/// entrypoints through the current runtime solve context and solver options.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_isolation_with_runtime_ctx_and_reentrant_entrypoints_and_state<
    T,
    FContextRef,
    FContextMut,
    FSimplifyExpr,
    FCollectSteps,
    FProveNonzeroStatus,
    FRenderExpr,
    FSolveReentrant,
    FIsKnownNegative,
    FIsolateReentrant,
    FClearBlockedHints,
    FSimplifyWithTactic,
    FProvePositive,
    FRegisterBlockedHint,
    FSimplifyWithTrace,
    FSymName,
>(
    state: &mut T,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    opts: crate::solver_options::SolverOptions,
    ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    context_ref: FContextRef,
    context_mut: FContextMut,
    simplify_expr: FSimplifyExpr,
    collect_steps: FCollectSteps,
    prove_nonzero_status: FProveNonzeroStatus,
    render_expr: FRenderExpr,
    solve_reentrant: FSolveReentrant,
    is_known_negative: FIsKnownNegative,
    isolate_reentrant: FIsolateReentrant,
    clear_blocked_hints: FClearBlockedHints,
    simplify_with_tactic: FSimplifyWithTactic,
    prove_positive: FProvePositive,
    register_blocked_hint: FRegisterBlockedHint,
    simplify_with_trace: FSimplifyWithTrace,
    sym_name: FSymName,
) -> Result<
    (
        SolutionSet,
        Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
    ),
    crate::error_model::CasError,
>
where
    FContextRef: Fn(&mut T) -> &cas_ast::Context + Clone,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context,
    FSimplifyExpr: FnMut(&mut T, ExprId) -> ExprId,
    FCollectSteps: FnMut(&mut T) -> bool,
    FProveNonzeroStatus: FnMut(&mut T, ExprId) -> crate::linear_solution::NonZeroStatus,
    FRenderExpr: Fn(&cas_ast::Context, ExprId) -> String,
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
    FIsKnownNegative: FnMut(&mut T, ExprId) -> bool,
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
    FClearBlockedHints: FnMut(&mut T),
    FSimplifyWithTactic: FnMut(&mut T, ExprId, &crate::simplify_options::SimplifyOptions) -> ExprId,
    FProvePositive: FnMut(
        &cas_ast::Context,
        ExprId,
        crate::value_domain::ValueDomain,
    ) -> crate::domain_proof::Proof,
    FRegisterBlockedHint: FnMut(crate::blocked_hint::BlockedHint),
    FSimplifyWithTrace: FnMut(&mut T, ExprId) -> (ExprId, Vec<(String, ExprId)>),
    FSymName: FnMut(&mut T, SymbolId) -> String,
{
    let solve_reentrant = std::cell::RefCell::new(solve_reentrant);
    let isolate_reentrant = std::cell::RefCell::new(isolate_reentrant);

    crate::solve_runtime_isolation_dispatch_context_runtime::dispatch_isolation_with_runtime_ctx_and_options_and_state(
        state,
        lhs,
        rhs,
        op,
        var,
        opts,
        ctx,
        context_ref,
        context_mut,
        simplify_expr,
        collect_steps,
        prove_nonzero_status,
        render_expr,
        |state, recursive_equation, solve_var| {
            (solve_reentrant.borrow_mut())(recursive_equation, solve_var, state, opts, ctx)
        },
        is_known_negative,
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
        clear_blocked_hints,
        simplify_with_tactic,
        prove_positive,
        register_blocked_hint,
        simplify_with_trace,
        sym_name,
    )
}
