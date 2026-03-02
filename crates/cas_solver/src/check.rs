//! Solution verification helpers.
//!
//! Verifies solver outputs by substituting candidates back into the original
//! equation and checking whether residuals simplify to zero.

use cas_ast::Context;
use cas_ast::{Equation, ExprId, SolutionSet};
use cas_math::expr_predicates::contains_variable;
use cas_math::ground_eval_guard::GroundEvalGuard;
use cas_solver_core::isolation_utils::is_numeric_zero;
use cas_solver_core::verification::{
    verify_solution_set_with_state,
    verify_solution_with_strict_fold_and_generic_fallback_with_default_stats_and_state,
};
use cas_solver_core::verify_substitution::substitute_equation_diff;

use crate::{DomainMode, EvalConfig, Simplifier, SimplifyOptions, ValueDomain};

pub use cas_solver_core::verification::{VerifyResult, VerifyStatus, VerifySummary};

/// Verify a single solution by substituting into the equation.
pub fn verify_solution(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solution: ExprId,
) -> VerifyStatus {
    let strict_opts = simplify_options_for_domain(DomainMode::Strict);
    let generic_opts = simplify_options_for_domain(DomainMode::Generic);

    verify_solution_with_strict_fold_and_generic_fallback_with_default_stats_and_state(
        simplifier,
        equation,
        var,
        solution,
        |state, eq, solve_var, candidate| {
            substitute_equation_diff(&mut state.context, eq, solve_var, candidate)
        },
        |state, expr| state.simplify_with_stats(expr, strict_opts.clone()).0,
        |state, expr| state.simplify_with_stats(expr, generic_opts.clone()).0,
        |state, expr| contains_variable(&state.context, expr),
        |state, expr| fold_numeric_islands(&mut state.context, expr),
        |state, expr| is_numeric_zero(&state.context, expr),
        render_expr,
    )
}

/// Verify a solution set, handling all [`SolutionSet`] variants.
pub fn verify_solution_set(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solutions: &SolutionSet,
) -> VerifyResult {
    let mut verify_discrete =
        |state: &mut Simplifier, solution: ExprId| verify_solution(state, equation, var, solution);
    verify_solution_set_with_state(simplifier, solutions, &mut verify_discrete)
}

fn simplify_options_for_domain(domain_mode: DomainMode) -> SimplifyOptions {
    SimplifyOptions {
        shared: crate::SharedSemanticConfig {
            semantics: EvalConfig {
                domain_mode,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fold_numeric_islands(ctx: &mut Context, root: ExprId) -> ExprId {
    let fold_opts = SimplifyOptions {
        collect_steps: false,
        expand_mode: false,
        shared: crate::SharedSemanticConfig {
            semantics: EvalConfig {
                domain_mode: DomainMode::Generic,
                value_domain: ValueDomain::RealOnly,
                ..Default::default()
            },
            ..Default::default()
        },
        budgets: crate::phase::PhaseBudgets {
            core_iters: 4,
            transform_iters: 2,
            rationalize_iters: 0,
            post_iters: 2,
            max_total_rewrites: 50,
        },
        ..Default::default()
    };

    cas_solver_core::verification_numeric_islands::fold_numeric_islands_guarded_with_default_limits_and_candidate_evaluator(
        ctx,
        root,
        GroundEvalGuard::enter,
        |src_ctx, id| {
            let mut tmp = Simplifier::with_context(src_ctx.clone());
            tmp.set_collect_steps(false);
            let (result, _, _) = tmp.simplify_with_stats(id, fold_opts.clone());
            Some((tmp.context, result))
        },
    )
}

fn render_expr(simplifier: &mut Simplifier, expr: ExprId) -> String {
    cas_formatter::render_expr(&simplifier.context, expr)
}
