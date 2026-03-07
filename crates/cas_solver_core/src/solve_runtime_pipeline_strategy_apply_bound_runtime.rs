//! Shared strategy-apply wrapper bound to [`RuntimeSolveAdapterState`].

use cas_ast::{Equation, ExprId, RelOp, SolutionSet};

/// Apply one strategy using default runtime-state helpers while only requiring
/// a crate-local algebraic expand callback plus recursive solve/isolation
/// entrypoints and the runtime positive prover.
#[allow(clippy::too_many_arguments)]
pub fn apply_strategy_with_runtime_state_and_reentrant_entrypoints_and_state<
    T,
    FSolveReentrant,
    FIsolateReentrant,
>(
    state: &mut T,
    kind: crate::strategy_order::SolveStrategyKind,
    equation: &Equation,
    var: &str,
    opts: crate::solver_options::SolverOptions,
    ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    solve_reentrant: FSolveReentrant,
    isolate_reentrant: FIsolateReentrant,
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
        crate::solve_runtime_adapter_state_runtime::simplifier_expand_expr,
        crate::solve_runtime_adapter_state_runtime::simplifier_render_expr,
        crate::solve_runtime_adapter_state_runtime::context_render_expr,
        solve_reentrant,
        isolate_reentrant,
        crate::proof_runtime_bound_runtime::prove_positive_with_runtime_proof_simplifier::<T>,
    )
}
