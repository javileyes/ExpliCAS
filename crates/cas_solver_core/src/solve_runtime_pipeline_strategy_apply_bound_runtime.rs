//! Shared strategy-apply wrapper bound to [`RuntimeSolveAdapterState`].

use cas_ast::{Equation, ExprId, RelOp, SolutionSet};

/// Apply one strategy using default runtime-state helpers while only requiring
/// a crate-local algebraic expand callback plus recursive solve/isolation
/// entrypoints and the runtime positive prover.
#[allow(clippy::too_many_arguments)]
pub fn apply_strategy_with_runtime_state_and_reentrant_entrypoints_and_state<
    T,
    FExpandExpr,
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
    expand_expr: FExpandExpr,
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
    T: crate::solve_runtime_adapter_state_runtime::RuntimeSolveAdapterState,
    FExpandExpr: FnMut(&mut T, ExprId) -> ExprId,
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
    crate::solve_runtime_pipeline_strategy_apply_reentrant_context_runtime::apply_strategy_with_runtime_ctx_and_reentrant_entrypoints_and_state(
        state,
        kind,
        equation,
        var,
        opts,
        ctx,
        crate::solve_runtime_adapter_state_runtime::simplifier_collect_steps,
        crate::solve_runtime_adapter_state_runtime::simplifier_context,
        crate::solve_runtime_adapter_state_runtime::simplifier_context_mut,
        crate::solve_runtime_adapter_state_runtime::simplifier_set_collect_steps,
        crate::solve_runtime_adapter_state_runtime::simplifier_simplify_expr,
        expand_expr,
        crate::solve_runtime_adapter_state_runtime::simplifier_render_expr,
        crate::solve_runtime_adapter_state_runtime::context_render_expr,
        solve_reentrant,
        isolate_reentrant,
        prove_positive,
    )
}
