//! Preflight and solve-entry runtime wrappers extracted from `solve_runtime_flow`.
//!
//! This module keeps entry guards and equation-preparation helpers isolated
//! from strategy/isolation execution wiring.

use cas_ast::{Equation, ExprId, RelOp, SolutionSet};
use std::hash::Hash;

/// Result of running solve preflight plus optional early rational-exponent prepass.
pub enum PreflightOrSolved<Ctx, S, E> {
    Continue {
        domain_exclusions: Vec<ExprId>,
        ctx: Ctx,
    },
    Solved(Result<(SolutionSet, Vec<S>), E>),
}

/// Enforce the default isolation recursion limit.
pub fn ensure_default_isolation_recursion_depth_or_error<E, FMapError>(
    current_depth: usize,
    map_error: FMapError,
) -> Result<(), E>
where
    FMapError: FnOnce() -> E,
{
    crate::solve_analysis::ensure_recursion_depth_within_limit_or_error(
        current_depth,
        crate::solve_budget::MAX_SOLVE_RECURSION_DEPTH,
        map_error,
    )
}

/// Execute isolation dispatch after enforcing the default recursion guard.
pub fn execute_isolation_with_default_depth_guard_and_dispatch_with_state<
    SState,
    S,
    E,
    FMapDepthError,
    FDispatch,
>(
    state: &mut SState,
    current_depth: usize,
    map_depth_error: FMapDepthError,
    dispatch: FDispatch,
) -> Result<(SolutionSet, Vec<S>), E>
where
    FMapDepthError: FnOnce() -> E,
    FDispatch: FnOnce(&mut SState) -> Result<(SolutionSet, Vec<S>), E>,
{
    ensure_default_isolation_recursion_depth_or_error(current_depth, map_depth_error)?;
    dispatch(state)
}

/// Enforce default solve-entry guards:
/// - recursion depth within solver budget
/// - solve variable present in equation
pub fn ensure_default_solve_entry_or_error<E, FDepthError, FMissingVarError>(
    ctx: &cas_ast::Context,
    equation: &Equation,
    var: &str,
    current_depth: usize,
    map_depth_error: FDepthError,
    map_missing_var_error: FMissingVarError,
) -> Result<(), E>
where
    FDepthError: FnOnce() -> E,
    FMissingVarError: FnOnce() -> E,
{
    crate::solve_analysis::ensure_solve_entry_for_equation_or_error(
        ctx,
        equation,
        var,
        current_depth,
        crate::solve_budget::MAX_SOLVE_RECURSION_DEPTH,
        map_depth_error,
        map_missing_var_error,
    )
}

/// Try the default early rational-exponent prepass:
/// - only for equality equations (`RelOp::Eq`)
/// - if a solve result appears, guard it against domain exclusions
pub fn try_apply_rational_exponent_prepass_with_default_eq_guard_and_exclusion_policy_with_state<
    SState,
    S,
    E,
    FApplyRationalExponent,
    FGuardSolved,
>(
    state: &mut SState,
    equation: &Equation,
    var: &str,
    domain_exclusions: &[ExprId],
    mut apply_rational_exponent: FApplyRationalExponent,
    mut guard_solved_result: FGuardSolved,
) -> Option<Result<(SolutionSet, Vec<S>), E>>
where
    FApplyRationalExponent:
        FnMut(&mut SState, &Equation, &str) -> Option<Result<(SolutionSet, Vec<S>), E>>,
    FGuardSolved:
        FnMut(Result<(SolutionSet, Vec<S>), E>, &[ExprId]) -> Result<(SolutionSet, Vec<S>), E>,
{
    if equation.op != RelOp::Eq {
        return None;
    }
    let result = apply_rational_exponent(state, equation, var)?;
    Some(guard_solved_result(result, domain_exclusions))
}

/// Build preflight context and run the default early rational-exponent prepass.
///
/// Returns:
/// - [`PreflightOrSolved::Solved`] when the prepass solved the equation, or
/// - [`PreflightOrSolved::Continue`] with `(domain_exclusions, solve_ctx)` for
///   the regular strategy pipeline.
pub fn run_preflight_with_default_rational_exponent_prepass_with_state<
    SState,
    Ctx,
    S,
    E,
    FBuildPreflight,
    FApplyRationalExponent,
    FGuardSolved,
>(
    state: &mut SState,
    equation: &Equation,
    var: &str,
    build_preflight: FBuildPreflight,
    mut apply_rational_exponent: FApplyRationalExponent,
    guard_solved_result: FGuardSolved,
) -> PreflightOrSolved<Ctx, S, E>
where
    FBuildPreflight: FnOnce(&mut SState) -> crate::solve_analysis::PreflightContext<Ctx>,
    FApplyRationalExponent:
        FnMut(&mut SState, &Equation, &str, &Ctx) -> Option<Result<(SolutionSet, Vec<S>), E>>,
    FGuardSolved:
        FnMut(Result<(SolutionSet, Vec<S>), E>, &[ExprId]) -> Result<(SolutionSet, Vec<S>), E>,
{
    let preflight = build_preflight(state);
    let domain_exclusions = preflight.domain_exclusions;
    let ctx = preflight.ctx;

    if let Some(result) =
        try_apply_rational_exponent_prepass_with_default_eq_guard_and_exclusion_policy_with_state(
            state,
            equation,
            var,
            &domain_exclusions,
            |state, equation, solve_var| apply_rational_exponent(state, equation, solve_var, &ctx),
            guard_solved_result,
        )
    {
        return PreflightOrSolved::Solved(result);
    }

    PreflightOrSolved::Continue {
        domain_exclusions,
        ctx,
    }
}

/// Enter the default equation-fingerprint cycle guard using a caller-provided
/// context accessor, returning a mapped error on cycle re-entry.
pub fn try_enter_default_equation_cycle_guard_with_context_ref_and_error_with_state<
    SState,
    E,
    FContextRef,
    FMapError,
>(
    state: &mut SState,
    equation: &Equation,
    var: &str,
    mut context_ref: FContextRef,
    map_error: FMapError,
) -> Result<crate::cycle_guard::CycleGuard, E>
where
    FContextRef: FnMut(&mut SState) -> &cas_ast::Context,
    FMapError: FnOnce() -> E,
{
    crate::solve_analysis::try_enter_equation_cycle_guard_with_error(
        context_ref(state),
        equation,
        var,
        map_error,
    )
}

/// Analyze solve preflight and derive equation-level conditions from an
/// existing-condition domain view.
///
/// This wraps:
/// - `analyze_equation_preflight_and_fork_context_with`
/// - `derive_equation_conditions_from_existing_with`
///
/// so runtime crates pass domain hooks once instead of wiring nested callbacks.
#[allow(clippy::too_many_arguments)]
pub fn analyze_preflight_and_fork_context_with_existing_condition_derivation<
    C,
    V,
    Domain,
    DomainEnv,
    Assumption,
    Scope,
    FInferSide,
    FNewDomain,
    FInsertCondition,
    FDeriveFromDomain,
    FInsertRequiredIntoDomainEnv,
>(
    ctx: &cas_ast::Context,
    equation: &Equation,
    var: &str,
    value_domain: V,
    parent_ctx: &crate::solve_context::SolveContext<DomainEnv, C, Assumption, Scope>,
    infer_side_conditions: FInferSide,
    mut new_domain: FNewDomain,
    mut insert_condition: FInsertCondition,
    mut derive_from_domain: FDeriveFromDomain,
    domain_env: DomainEnv,
    insert_required_into_domain_env: FInsertRequiredIntoDomainEnv,
) -> crate::solve_analysis::PreflightContext<
    crate::solve_context::SolveContext<DomainEnv, C, Assumption, Scope>,
>
where
    C: Eq + Hash + Clone,
    V: Copy,
    Assumption: Clone,
    Scope: Clone + PartialEq,
    FInferSide: FnMut(ExprId, V) -> Vec<C>,
    FNewDomain: FnMut() -> Domain,
    FInsertCondition: FnMut(&mut Domain, C),
    FDeriveFromDomain: FnMut(ExprId, ExprId, &Domain, V) -> Vec<C>,
    FInsertRequiredIntoDomainEnv: FnMut(&mut DomainEnv, &C),
{
    crate::solve_analysis::analyze_equation_preflight_and_fork_context_with(
        ctx,
        equation,
        var,
        value_domain,
        parent_ctx,
        infer_side_conditions,
        |lhs, rhs, existing, eval_domain| {
            crate::solve_analysis::derive_equation_conditions_from_existing_with(
                lhs,
                rhs,
                existing,
                eval_domain,
                &mut new_domain,
                &mut insert_condition,
                &mut derive_from_domain,
            )
        },
        domain_env,
        insert_required_into_domain_env,
    )
}

/// Prepare one equation for strategy dispatch using default residual-candidate
/// acceptance (`accept when smaller or when variable is eliminated`).
#[allow(clippy::too_many_arguments)]
pub fn prepare_equation_for_strategy_with_default_residual_acceptance_and_state<
    SState,
    FContainsVar,
    FSimplifyForSolve,
    FRecomposePowQuotient,
    FStructuralRewrite,
    FSemanticRewrite,
    FBuildDifference,
    FExpandAlgebraic,
    FExpandTrig,
    FContext,
    FZeroExpr,
>(
    state: &mut SState,
    equation: &Equation,
    var: &str,
    contains_var: FContainsVar,
    simplify_for_solve: FSimplifyForSolve,
    try_recompose_pow_quotient: FRecomposePowQuotient,
    structural_rewrite: FStructuralRewrite,
    semantic_rewrite: FSemanticRewrite,
    build_difference: FBuildDifference,
    expand_algebraic: FExpandAlgebraic,
    expand_trig: FExpandTrig,
    mut context: FContext,
    zero_expr: FZeroExpr,
) -> crate::solve_analysis::PreparedEquationResidual
where
    FContainsVar: FnMut(&mut SState, ExprId, &str) -> bool,
    FSimplifyForSolve: FnMut(&mut SState, ExprId) -> ExprId,
    FRecomposePowQuotient: FnMut(&mut SState, ExprId) -> Option<ExprId>,
    FStructuralRewrite: FnMut(&mut SState, ExprId, ExprId) -> Option<(ExprId, ExprId)>,
    FSemanticRewrite: FnMut(&mut SState, ExprId, ExprId) -> Option<(ExprId, ExprId)>,
    FBuildDifference: FnMut(&mut SState, ExprId, ExprId) -> ExprId,
    FExpandAlgebraic: FnMut(&mut SState, ExprId) -> ExprId,
    FExpandTrig: FnMut(&mut SState, ExprId) -> ExprId,
    FContext: FnMut(&mut SState) -> &cas_ast::Context,
    FZeroExpr: FnMut(&mut SState) -> ExprId,
{
    crate::solve_analysis::prepare_equation_for_strategy_with_state(
        state,
        equation,
        var,
        contains_var,
        simplify_for_solve,
        try_recompose_pow_quotient,
        structural_rewrite,
        semantic_rewrite,
        build_difference,
        expand_algebraic,
        expand_trig,
        |state, current, candidate, var_name| {
            crate::solve_analysis::accept_residual_rewrite_candidate(
                context(state),
                current,
                candidate,
                var_name,
            )
        },
        zero_expr,
    )
}

/// Prepare one equation for strategy dispatch using:
/// - default residual acceptance policy,
/// - default `pow`-quotient recomposition,
/// - default structural additive cancellation.
///
/// Runtime crates only provide semantic fallback rewrite and runtime kernels.
#[allow(clippy::too_many_arguments)]
pub fn prepare_equation_for_strategy_with_default_structural_recompose_and_cancel_and_default_residual_acceptance_with_state<
    SState,
    FContainsVar,
    FSimplifyForSolve,
    FSemanticRewrite,
    FBuildDifference,
    FExpandAlgebraic,
    FExpandTrig,
    FContext,
    FContextMut,
    FZeroExpr,
>(
    state: &mut SState,
    equation: &Equation,
    var: &str,
    contains_var: FContainsVar,
    simplify_for_solve: FSimplifyForSolve,
    semantic_rewrite: FSemanticRewrite,
    build_difference: FBuildDifference,
    expand_algebraic: FExpandAlgebraic,
    expand_trig: FExpandTrig,
    context: FContext,
    context_mut: FContextMut,
    zero_expr: FZeroExpr,
) -> crate::solve_analysis::PreparedEquationResidual
where
    FContainsVar: FnMut(&mut SState, ExprId, &str) -> bool,
    FSimplifyForSolve: FnMut(&mut SState, ExprId) -> ExprId,
    FSemanticRewrite: FnMut(&mut SState, ExprId, ExprId) -> Option<(ExprId, ExprId)>,
    FBuildDifference: FnMut(&mut SState, ExprId, ExprId) -> ExprId,
    FExpandAlgebraic: FnMut(&mut SState, ExprId) -> ExprId,
    FExpandTrig: FnMut(&mut SState, ExprId) -> ExprId,
    FContext: FnMut(&mut SState) -> &cas_ast::Context,
    FContextMut: FnMut(&mut SState) -> &mut cas_ast::Context,
    FZeroExpr: FnMut(&mut SState) -> ExprId,
{
    let context_mut = std::cell::RefCell::new(context_mut);

    prepare_equation_for_strategy_with_default_residual_acceptance_and_state(
        state,
        equation,
        var,
        contains_var,
        simplify_for_solve,
        |state, expr| {
            crate::isolation_utils::try_recompose_pow_quotient((context_mut.borrow_mut())(state), expr)
        },
        |state, lhs, rhs| {
            crate::cancel_common_terms::cancel_common_additive_terms(
                (context_mut.borrow_mut())(state),
                lhs,
                rhs,
            )
            .map(|rewrite| (rewrite.new_lhs, rewrite.new_rhs))
        },
        semantic_rewrite,
        build_difference,
        expand_algebraic,
        expand_trig,
        context,
        zero_expr,
    )
}
