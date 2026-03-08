//! Shared `solve_inner` orchestration bound to the current runtime solve
//! context and option model.

use cas_ast::{Equation, ExprId, SolutionSet};

/// Run the default `solve_inner` orchestration using the runtime solve context,
/// runtime solver options, the rational-exponent early strategy, and the
/// standard solved-result / prepared-equation guards.
#[allow(clippy::too_many_arguments)]
pub fn solve_inner_with_runtime_ctx_and_default_rational_preflight_prepare_pipeline_with_state<
    SState,
    FContextRef,
    FMapDepthError,
    FMapMissingVarError,
    FBuildPreflight,
    FApplyStrategy,
    FPrepareEquation,
    FExecutePipeline,
>(
    state: &mut SState,
    equation: &Equation,
    var: &str,
    opts: crate::solver_options::SolverOptions,
    parent_ctx: &crate::solve_runtime_types::RuntimeSolveCtx,
    context_ref: FContextRef,
    map_depth_error: FMapDepthError,
    map_missing_var_error: FMapMissingVarError,
    build_preflight: FBuildPreflight,
    mut apply_strategy: FApplyStrategy,
    prepare_equation: FPrepareEquation,
    execute_pipeline: FExecutePipeline,
) -> Result<
    (
        SolutionSet,
        Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
    ),
    crate::error_model::CasError,
>
where
    FContextRef: FnMut(&mut SState) -> &cas_ast::Context + Clone,
    FMapDepthError: FnOnce() -> crate::error_model::CasError,
    FMapMissingVarError: FnOnce() -> crate::error_model::CasError,
    FBuildPreflight: FnOnce(
        &mut SState,
        &Equation,
        &str,
        crate::value_domain::ValueDomain,
        &crate::solve_runtime_types::RuntimeSolveCtx,
    ) -> crate::solve_analysis::PreflightContext<
        crate::solve_runtime_types::RuntimeSolveCtx,
    >,
    FApplyStrategy: FnMut(
        crate::strategy_order::SolveStrategyKind,
        &Equation,
        &str,
        &mut SState,
        &crate::solver_options::SolverOptions,
        &crate::solve_runtime_types::RuntimeSolveCtx,
    ) -> Option<
        Result<
            (
                SolutionSet,
                Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
            ),
            crate::error_model::CasError,
        >,
    >,
    FPrepareEquation:
        FnOnce(&mut SState, &Equation, &str) -> crate::solve_analysis::PreparedEquationResidual,
    FExecutePipeline: FnOnce(
        &mut SState,
        &Equation,
        &Equation,
        ExprId,
        &str,
        crate::solver_options::SolverOptions,
        &crate::solve_runtime_types::RuntimeSolveCtx,
        &[ExprId],
    ) -> Result<
        (
            SolutionSet,
            Vec<crate::solve_runtime_mapping::DefaultSolveStep>,
        ),
        crate::error_model::CasError,
    >,
{
    let current_depth = parent_ctx.depth().saturating_add(1);
    let mut context_ref_for_debug = context_ref.clone();

    crate::solve_runtime_flow::solve_inner_with_default_entry_preflight_prepare_and_pipeline_with_state(
        state,
        equation,
        var,
        current_depth,
        context_ref,
        map_depth_error,
        map_missing_var_error,
        |state| build_preflight(state, equation, var, opts.value_domain, parent_ctx),
        |state, preflight_eq, solve_var, solve_ctx| {
            apply_strategy(
                crate::strategy_order::SolveStrategyKind::RationalExponent,
                preflight_eq,
                solve_var,
                state,
                &opts,
                solve_ctx,
            )
        },
        crate::solve_analysis::guard_solved_result_with_exclusions,
        prepare_equation,
        |state, prepared_eq| {
            crate::solve_analysis::debug_assert_equation_no_top_level_sub(
                context_ref_for_debug(state),
                prepared_eq,
            );
        },
        |state, original_eq, prepared_eq, residual, solve_var, solve_ctx, exclusions| {
            execute_pipeline(
                state,
                original_eq,
                prepared_eq,
                residual,
                solve_var,
                opts,
                solve_ctx,
                exclusions,
            )
        },
    )
}
