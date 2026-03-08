//! High-level solve-loop orchestration wrappers.
//!
//! These helpers compose entry guards, preflight, equation preparation and
//! strategy-pipeline execution so runtime crates can stay as thin adapters.

use cas_ast::{Equation, ExprId, SolutionSet};

/// Execute the default `solve_inner` orchestration sequence:
/// 1) solve-entry guard,
/// 2) preflight + optional rational-exponent early solve,
/// 3) equation preparation for strategy dispatch,
/// 4) strategy pipeline execution.
///
/// Runtime crates provide only the state accessors and kernel callbacks.
#[allow(clippy::too_many_arguments)]
pub fn solve_inner_with_default_entry_preflight_prepare_and_pipeline_with_state<
    SState,
    Ctx,
    S,
    E,
    FContextRef,
    FMapDepthError,
    FMapMissingVarError,
    FBuildPreflight,
    FApplyRationalExponent,
    FGuardSolved,
    FPrepareEquation,
    FDebugAssertPreparedEquation,
    FExecutePipeline,
>(
    state: &mut SState,
    equation: &Equation,
    var: &str,
    current_depth: usize,
    mut context_ref: FContextRef,
    map_depth_error: FMapDepthError,
    map_missing_var_error: FMapMissingVarError,
    build_preflight: FBuildPreflight,
    apply_rational_exponent: FApplyRationalExponent,
    guard_solved_result: FGuardSolved,
    prepare_equation: FPrepareEquation,
    mut debug_assert_prepared_equation: FDebugAssertPreparedEquation,
    execute_pipeline: FExecutePipeline,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FContextRef: FnMut(&mut SState) -> &cas_ast::Context,
    FMapDepthError: FnOnce() -> E,
    FMapMissingVarError: FnOnce() -> E,
    FBuildPreflight: FnOnce(&mut SState) -> crate::solve_analysis::PreflightContext<Ctx>,
    FApplyRationalExponent:
        FnMut(&mut SState, &Equation, &str, &Ctx) -> Option<Result<(SolutionSet, Vec<S>), E>>,
    FGuardSolved:
        FnMut(Result<(SolutionSet, Vec<S>), E>, &[ExprId]) -> Result<(SolutionSet, Vec<S>), E>,
    FPrepareEquation:
        FnOnce(&mut SState, &Equation, &str) -> crate::solve_analysis::PreparedEquationResidual,
    FDebugAssertPreparedEquation: FnMut(&mut SState, &Equation),
    FExecutePipeline: FnOnce(
        &mut SState,
        &Equation,
        &Equation,
        ExprId,
        &str,
        &Ctx,
        &[ExprId],
    ) -> Result<(SolutionSet, Vec<S>), E>,
{
    crate::solve_runtime_flow_preflight::ensure_default_solve_entry_or_error(
        context_ref(state),
        equation,
        var,
        current_depth,
        map_depth_error,
        map_missing_var_error,
    )?;

    let (domain_exclusions, ctx) = match crate::solve_runtime_flow_preflight::run_preflight_with_default_rational_exponent_prepass_with_state(
        state,
        equation,
        var,
        build_preflight,
        apply_rational_exponent,
        guard_solved_result,
    ) {
        crate::solve_runtime_flow_preflight::PreflightOrSolved::Solved(result) => return result,
        crate::solve_runtime_flow_preflight::PreflightOrSolved::Continue {
            domain_exclusions,
            ctx,
        } => (domain_exclusions, ctx),
    };

    let prepared = prepare_equation(state, equation, var);
    let simplified_equation = prepared.equation;
    let residual = prepared.residual;
    debug_assert_prepared_equation(state, &simplified_equation);

    execute_pipeline(
        state,
        equation,
        &simplified_equation,
        residual,
        var,
        &ctx,
        &domain_exclusions,
    )
}
