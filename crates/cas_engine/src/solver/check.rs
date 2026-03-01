//! Solution verification module.
//!
//! Verifies solver solutions by substituting back into the original equation
//! and checking if the result simplifies to zero.
//!
//! Uses a 2-phase approach:
//! - Phase 1 (Strict): domain-honest simplification.
//! - Phase 2 (Generic fallback): only when strict residual is variable-free.

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

use crate::engine::Simplifier;
use crate::solver::runtime_adapters::simplifier_render_expr;

pub use cas_solver_core::verification::{VerifyResult, VerifyStatus, VerifySummary};

/// Verify a single solution by substituting into the equation.
pub fn verify_solution(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solution: ExprId,
) -> VerifyStatus {
    let strict_opts = simplify_options_for_domain(crate::domain::DomainMode::Strict);
    let generic_opts = simplify_options_for_domain(crate::domain::DomainMode::Generic);

    verify_solution_with_strict_fold_and_generic_fallback_with_default_stats_and_state(
        simplifier,
        equation,
        var,
        solution,
        |simplifier, eq, solve_var, candidate| {
            substitute_equation_diff(&mut simplifier.context, eq, solve_var, candidate)
        },
        |simplifier, expr| simplifier.simplify_with_stats(expr, strict_opts.clone()).0,
        |simplifier, expr| simplifier.simplify_with_stats(expr, generic_opts.clone()).0,
        |simplifier, expr| contains_variable(&simplifier.context, expr),
        |simplifier, expr| fold_numeric_islands(&mut simplifier.context, expr),
        |simplifier, expr| is_numeric_zero(&simplifier.context, expr),
        simplifier_render_expr,
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

fn simplify_options_for_domain(domain_mode: crate::domain::DomainMode) -> crate::SimplifyOptions {
    crate::SimplifyOptions {
        shared: crate::phase::SharedSemanticConfig {
            semantics: crate::semantics::EvalConfig {
                domain_mode,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

fn fold_numeric_islands(ctx: &mut Context, root: ExprId) -> ExprId {
    let fold_opts = crate::SimplifyOptions {
        collect_steps: false,
        expand_mode: false,
        shared: crate::phase::SharedSemanticConfig {
            semantics: crate::semantics::EvalConfig {
                domain_mode: crate::domain::DomainMode::Generic,
                value_domain: crate::semantics::ValueDomain::RealOnly,
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
