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
use cas_solver_core::domain_mode::DomainMode;
use cas_solver_core::isolation_utils::is_numeric_zero;
use cas_solver_core::verification_flow::{
    verify_solution_set_for_equation_with_state, verify_solution_with_domain_modes_with_state,
};
use cas_solver_core::verify_substitution::substitute_equation_diff;

use crate::engine::Simplifier;
use crate::solver::simplifier_render_expr;

pub use cas_solver_core::verification::{VerifyResult, VerifyStatus, VerifySummary};

/// Verify a single solution by substituting into the equation.
pub fn verify_solution(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
    solution: ExprId,
) -> VerifyStatus {
    verify_solution_with_domain_modes_with_state(
        simplifier,
        equation,
        var,
        solution,
        |state, eq, solve_var, candidate| {
            substitute_equation_diff(&mut state.context, eq, solve_var, candidate)
        },
        |state, expr, domain_mode| {
            let opts = simplify_options_for_domain(domain_mode);
            state.simplify_with_stats(expr, opts).0
        },
        |state, expr| contains_variable(&state.context, expr),
        |state, expr| fold_numeric_islands(&mut state.context, expr),
        |state, expr| is_numeric_zero(&state.context, expr),
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
    verify_solution_set_for_equation_with_state(
        simplifier,
        equation,
        var,
        solutions,
        verify_solution,
    )
}

fn simplify_options_for_domain(domain_mode: DomainMode) -> crate::SimplifyOptions {
    crate::SimplifyOptions {
        shared: crate::SharedSemanticConfig {
            semantics: crate::EvalConfig {
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
        shared: crate::SharedSemanticConfig {
            semantics: crate::EvalConfig {
                domain_mode: crate::DomainMode::Generic,
                value_domain: crate::ValueDomain::RealOnly,
                ..Default::default()
            },
            ..Default::default()
        },
        budgets: crate::PhaseBudgets {
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
