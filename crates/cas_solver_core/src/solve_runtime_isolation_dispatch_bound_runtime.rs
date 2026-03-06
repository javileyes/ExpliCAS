//! Shared isolation-dispatch wrapper bound to [`RuntimeSolveAdapterState`].

use cas_ast::{Equation, ExprId, RelOp, SolutionSet};

/// Dispatch isolation using the default runtime-state helpers while only
/// requiring recursive solve/isolation entrypoints plus the runtime-specific
/// positive prover and blocked-hint sink.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_isolation_with_runtime_state_and_reentrant_entrypoints_and_state<
    T,
    FSolveReentrant,
    FIsolateReentrant,
    FProvePositive,
    FRegisterBlockedHint,
>(
    state: &mut T,
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    opts: crate::solver_options::SolverOptions,
    ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    solve_reentrant: FSolveReentrant,
    isolate_reentrant: FIsolateReentrant,
    prove_positive: FProvePositive,
    register_blocked_hint: FRegisterBlockedHint,
) -> Result<
    (
        SolutionSet,
        Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
    ),
    crate::error_model::CasError,
>
where
    T: crate::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState,
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
    FRegisterBlockedHint: FnMut(crate::blocked_hint::BlockedHint),
{
    crate::solve_runtime_isolation_dispatch_reentrant_context_runtime::dispatch_isolation_with_runtime_ctx_and_reentrant_entrypoints_and_state(
        state,
        lhs,
        rhs,
        op,
        var,
        opts,
        ctx,
        crate::solve_runtime_adapter_state_runtime::simplifier_context,
        crate::solve_runtime_adapter_state_runtime::simplifier_context_mut,
        crate::solve_runtime_adapter_state_runtime::simplifier_simplify_expr,
        crate::solve_runtime_adapter_state_runtime::simplifier_collect_steps,
        crate::solve_runtime_adapter_state_runtime::simplifier_prove_nonzero_status,
        crate::solve_runtime_adapter_state_runtime::context_render_expr,
        solve_reentrant,
        crate::solve_runtime_adapter_state_runtime::simplifier_is_known_negative,
        isolate_reentrant,
        crate::solve_runtime_adapter_state_runtime::simplifier_clear_blocked_hints,
        crate::solve_runtime_adapter_state_runtime::simplifier_simplify_with_options_expr,
        prove_positive,
        register_blocked_hint,
        crate::solve_runtime_adapter_state_runtime::simplify_rhs_with_step_pairs,
        crate::solve_runtime_adapter_state_runtime::sym_name_as_string,
    )
}
