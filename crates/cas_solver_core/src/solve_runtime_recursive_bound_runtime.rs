//! Shared recursive routing helpers for facade solve runtimes.

use cas_ast::{Equation, ExprId, RelOp, SolutionSet};

/// Run `solve_inner` using default preflight/prepare/pipeline wiring plus the
/// default recursive apply/isolation routes.
#[allow(clippy::too_many_arguments)]
pub fn solve_inner_with_runtime_state_and_default_recursive_routes_and_errors<
    T,
    FSolveReentrant,
    FRegisterBlockedHint,
    FMapDepthError,
    FMapMissingVarError,
>(
    equation: &Equation,
    var: &str,
    state: &mut T,
    opts: crate::solver_options::SolverOptions,
    parent_ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    solve_reentrant: FSolveReentrant,
    register_blocked_hint: FRegisterBlockedHint,
    map_depth_error: FMapDepthError,
    map_missing_var_error: FMapMissingVarError,
) -> Result<
    (
        SolutionSet,
        Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
    ),
    crate::error_model::CasError,
>
where
    T: crate::proof_runtime_bound_runtime::RuntimeProofSimplifierFactory
        + crate::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState,
    FSolveReentrant: Copy
        + Fn(
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
    FRegisterBlockedHint: Copy + Fn(crate::blocked_hint::BlockedHint),
    FMapDepthError: FnOnce() -> crate::error_model::CasError,
    FMapMissingVarError: FnOnce() -> crate::error_model::CasError,
{
    crate::solve_runtime_orchestration_context_runtime::solve_inner_with_runtime_ctx_and_default_rational_preflight_prepare_pipeline_with_state(
        state,
        equation,
        var,
        opts,
        parent_ctx,
        |state| state.runtime_context(),
        map_depth_error,
        map_missing_var_error,
        |state, equation, solve_var, value_domain, solve_ctx| {
            crate::solve_runtime_pipeline_preflight_context_bound_runtime::build_runtime_solve_preflight_state_with_adapter_state_and_default_domain_derivation(
                &*state,
                equation,
                solve_var,
                value_domain,
                solve_ctx,
            )
        },
        |kind, equation, solve_var, state, solve_opts, solve_ctx| {
            apply_strategy_with_runtime_state_and_default_recursive_routes(
                kind,
                equation,
                solve_var,
                state,
                *solve_opts,
                solve_ctx,
                solve_reentrant,
                register_blocked_hint,
            )
        },
        crate::solve_runtime_pipeline_preflight_equation_bound_runtime::prepare_equation_for_strategy_with_adapter_state_and_default_structural_recompose,
        |state, original_eq, prepared_eq, residual, solve_var, solve_opts, solve_ctx, exclusions| {
            crate::solve_runtime_pipeline_strategy_execute_bound_runtime::execute_strategy_pipeline_with_runtime_state_and_apply_entrypoint_with_state(
                state,
                original_eq,
                prepared_eq,
                residual,
                solve_var,
                solve_opts,
                solve_ctx,
                exclusions,
                |kind, equation, solve_var, state, solve_opts, solve_ctx| {
                    apply_strategy_with_runtime_state_and_default_recursive_routes(
                        kind,
                        equation,
                        solve_var,
                        state,
                        *solve_opts,
                        solve_ctx,
                        solve_reentrant,
                        register_blocked_hint,
                    )
                },
            )
        },
    )
}

/// Apply one strategy using the default recursive solve/isolation routes.
#[allow(clippy::too_many_arguments)]
pub fn apply_strategy_with_runtime_state_and_default_recursive_routes<
    T,
    FSolveReentrant,
    FRegisterBlockedHint,
>(
    kind: crate::strategy_order::SolveStrategyKind,
    equation: &Equation,
    var: &str,
    state: &mut T,
    opts: crate::solver_options::SolverOptions,
    ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    solve_reentrant: FSolveReentrant,
    register_blocked_hint: FRegisterBlockedHint,
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
    T: crate::proof_runtime_bound_runtime::RuntimeProofSimplifierFactory
        + crate::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState,
    FSolveReentrant: Copy
        + Fn(
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
    FRegisterBlockedHint: Copy + Fn(crate::blocked_hint::BlockedHint),
{
    crate::solve_runtime_pipeline_strategy_apply_bound_runtime::apply_strategy_with_runtime_state_and_reentrant_entrypoints_and_state(
        state,
        kind,
        equation,
        var,
        opts,
        ctx,
        solve_reentrant,
        |lhs, rhs, op, solve_var, nested_state, nested_opts, nested_ctx| {
            isolate_with_default_depth_with_runtime_state_and_default_recursive_routes(
                lhs,
                rhs,
                op,
                solve_var,
                nested_state,
                nested_opts,
                nested_ctx,
                solve_reentrant,
                register_blocked_hint,
            )
        },
    )
}

/// Guard recursion depth and dispatch isolation using the default recursive
/// solve/isolation routes.
#[allow(clippy::too_many_arguments)]
pub fn isolate_with_default_depth_with_runtime_state_and_default_recursive_routes<
    T,
    FSolveReentrant,
    FRegisterBlockedHint,
>(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    state: &mut T,
    opts: crate::solver_options::SolverOptions,
    ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    solve_reentrant: FSolveReentrant,
    register_blocked_hint: FRegisterBlockedHint,
) -> Result<
    (
        SolutionSet,
        Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
    ),
    crate::error_model::CasError,
>
where
    T: crate::proof_runtime_bound_runtime::RuntimeProofSimplifierFactory
        + crate::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState,
    FSolveReentrant: Copy
        + Fn(
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
    FRegisterBlockedHint: Copy + Fn(crate::blocked_hint::BlockedHint),
{
    crate::solve_runtime_isolation_entry_reentrant_context_runtime::isolate_with_default_depth_guard_and_dispatch_with_state(
        state,
        lhs,
        rhs,
        op,
        var,
        opts,
        ctx,
        |dispatch_lhs, dispatch_rhs, dispatch_op, dispatch_var, dispatch_state, dispatch_opts, dispatch_ctx| {
            dispatch_isolation_with_runtime_state_and_default_recursive_routes(
                dispatch_lhs,
                dispatch_rhs,
                dispatch_op,
                dispatch_var,
                dispatch_state,
                dispatch_opts,
                dispatch_ctx,
                solve_reentrant,
                register_blocked_hint,
            )
        },
    )
}

/// Dispatch isolation using the default recursive solve/isolation routes.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_isolation_with_runtime_state_and_default_recursive_routes<
    T,
    FSolveReentrant,
    FRegisterBlockedHint,
>(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    state: &mut T,
    opts: crate::solver_options::SolverOptions,
    ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    solve_reentrant: FSolveReentrant,
    register_blocked_hint: FRegisterBlockedHint,
) -> Result<
    (
        SolutionSet,
        Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
    ),
    crate::error_model::CasError,
>
where
    T: crate::proof_runtime_bound_runtime::RuntimeProofSimplifierFactory
        + crate::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState,
    FSolveReentrant: Copy
        + Fn(
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
    FRegisterBlockedHint: Copy + Fn(crate::blocked_hint::BlockedHint),
{
    crate::solve_runtime_isolation_dispatch_bound_runtime::dispatch_isolation_with_runtime_state_and_reentrant_entrypoints_and_state(
        state,
        lhs,
        rhs,
        op,
        var,
        opts,
        ctx,
        solve_reentrant,
        |nested_lhs, nested_rhs, nested_op, nested_var, nested_state, nested_opts, nested_ctx| {
            isolate_with_default_depth_with_runtime_state_and_default_recursive_routes(
                nested_lhs,
                nested_rhs,
                nested_op,
                nested_var,
                nested_state,
                nested_opts,
                nested_ctx,
                solve_reentrant,
                register_blocked_hint,
            )
        },
        register_blocked_hint,
    )
}
