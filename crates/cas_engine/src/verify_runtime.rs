//! Solution verification runtime helpers.

use crate::conservative_simplify::{
    conservative_numeric_fold_options, simplify_options_for_domain,
};
use crate::engine::Simplifier;
use cas_ast::{Equation, ExprId, SolutionSet};
use cas_math::expr_predicates::contains_variable;
use cas_solver_core::isolation_utils::is_numeric_zero;
use cas_solver_core::verification::{VerifyResult, VerifyStatus};
use cas_solver_core::verification_flow::{
    verify_solution_set_for_equation_with_state, verify_solution_with_domain_modes_with_state,
};
use cas_solver_core::verify_substitution::substitute_equation_diff;

fn fold_numeric_islands(ctx: &mut cas_ast::Context, root: ExprId) -> ExprId {
    let fold_opts = conservative_numeric_fold_options();

    cas_solver_core::verification_runtime_helpers::fold_numeric_islands_with_default_guard_and_candidate(
        ctx,
        root,
        |src_ctx, id| {
            let mut tmp = Simplifier::with_context(src_ctx.clone());
            tmp.set_collect_steps(false);
            let (result, _, _) = tmp.simplify_with_stats(id, fold_opts.clone());
            Some((tmp.context, result))
        },
    )
}

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
        |state, expr| cas_formatter::render_expr(&state.context, expr),
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
