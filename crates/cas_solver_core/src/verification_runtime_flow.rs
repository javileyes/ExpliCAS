//! Shared verification runtime orchestration for integration crates.
//!
//! This keeps conservative-domain simplify wiring out of runtime wrappers
//! (`cas_engine`, `cas_solver`) so they only provide state-specific kernels.

use crate::domain_mode::DomainMode;
use crate::verification::{VerifyResult, VerifyStatus};
use cas_ast::{Equation, ExprId};

/// Verify one candidate solution using conservative domain-mode simplify wiring.
#[allow(clippy::too_many_arguments)]
pub fn verify_solution_with_conservative_domain_simplify_with_state<
    T,
    FSubstituteDiff,
    FSimplifyWithOptions,
    FContainsVariable,
    FFoldNumericIslands,
    FIsZero,
    FRenderExpr,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    solution: ExprId,
    substitute_diff: FSubstituteDiff,
    mut simplify_with_options: FSimplifyWithOptions,
    contains_variable: FContainsVariable,
    fold_numeric_islands: FFoldNumericIslands,
    is_zero: FIsZero,
    render_expr: FRenderExpr,
) -> VerifyStatus
where
    FSubstituteDiff: FnMut(&mut T, &Equation, &str, ExprId) -> ExprId,
    FSimplifyWithOptions: FnMut(&mut T, ExprId, crate::simplify_options::SimplifyOptions) -> ExprId,
    FContainsVariable: FnMut(&mut T, ExprId) -> bool,
    FFoldNumericIslands: FnMut(&mut T, ExprId) -> ExprId,
    FIsZero: FnMut(&mut T, ExprId) -> bool,
    FRenderExpr: FnMut(&mut T, ExprId) -> String,
{
    crate::verification_flow::verify_solution_with_domain_modes_with_state(
        state,
        equation,
        var,
        solution,
        substitute_diff,
        |state, expr, domain_mode: DomainMode| {
            let opts = crate::conservative_eval_config::simplify_options_for_domain(domain_mode);
            simplify_with_options(state, expr, opts)
        },
        contains_variable,
        fold_numeric_islands,
        is_zero,
        render_expr,
    )
}

/// Verify one candidate solution using shared runtime kernels:
/// - domain-mode simplify options from conservative config
/// - numeric-island fold with default guard/limits
/// - variable/zero/render checks from caller-provided context accessors.
#[allow(clippy::too_many_arguments)]
pub fn verify_solution_with_runtime_kernels_with_state<
    T,
    FContext,
    FContextMut,
    FSimplifyWithOptions,
    FGroundEvalCandidate,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    solution: ExprId,
    context: FContext,
    context_mut: FContextMut,
    simplify_with_options: FSimplifyWithOptions,
    ground_eval_candidate: FGroundEvalCandidate,
) -> VerifyStatus
where
    FContext: Fn(&T) -> &cas_ast::Context + Copy,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context + Copy,
    FSimplifyWithOptions:
        Fn(&mut T, ExprId, crate::simplify_options::SimplifyOptions) -> ExprId + Copy,
    FGroundEvalCandidate: Fn(
            &cas_ast::Context,
            ExprId,
            &crate::simplify_options::SimplifyOptions,
        ) -> Option<(cas_ast::Context, ExprId)>
        + Copy,
{
    verify_solution_with_conservative_domain_simplify_with_state(
        state,
        equation,
        var,
        solution,
        |state, eq, solve_var, candidate| {
            crate::verify_substitution::substitute_equation_diff(
                context_mut(state),
                eq,
                solve_var,
                candidate,
            )
        },
        simplify_with_options,
        |state, expr| cas_math::expr_predicates::contains_variable(context(state), expr),
        |state, root| {
            crate::verification_runtime_helpers::fold_numeric_islands_with_default_guard_and_conservative_options(
                context_mut(state),
                root,
                ground_eval_candidate,
            )
        },
        |state, expr| crate::isolation_utils::is_numeric_zero(context(state), expr),
        |state, expr| cas_formatter::render_expr(context(state), expr),
    )
}

/// Verify a full solution set using the same runtime kernels as
/// [`verify_solution_with_runtime_kernels_with_state`].
#[allow(clippy::too_many_arguments)]
pub fn verify_solution_set_with_runtime_kernels_with_state<
    T,
    FContext,
    FContextMut,
    FSimplifyWithOptions,
    FGroundEvalCandidate,
>(
    state: &mut T,
    equation: &Equation,
    var: &str,
    solutions: &cas_ast::SolutionSet,
    context: FContext,
    context_mut: FContextMut,
    simplify_with_options: FSimplifyWithOptions,
    ground_eval_candidate: FGroundEvalCandidate,
) -> VerifyResult
where
    FContext: Fn(&T) -> &cas_ast::Context + Copy,
    FContextMut: Fn(&mut T) -> &mut cas_ast::Context + Copy,
    FSimplifyWithOptions:
        Fn(&mut T, ExprId, crate::simplify_options::SimplifyOptions) -> ExprId + Copy,
    FGroundEvalCandidate: Fn(
            &cas_ast::Context,
            ExprId,
            &crate::simplify_options::SimplifyOptions,
        ) -> Option<(cas_ast::Context, ExprId)>
        + Copy,
{
    crate::verification_flow::verify_solution_set_for_equation_with_state(
        state,
        equation,
        var,
        solutions,
        |state, equation, var, solution| {
            verify_solution_with_runtime_kernels_with_state(
                state,
                equation,
                var,
                solution,
                context,
                context_mut,
                simplify_with_options,
                ground_eval_candidate,
            )
        },
    )
}
