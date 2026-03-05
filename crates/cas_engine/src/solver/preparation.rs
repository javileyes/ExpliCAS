use crate::engine::Simplifier;
use cas_ast::{Equation, Expr};
use cas_solver_core::solve_analysis::PreparedEquationResidual;

use super::{
    cancel_common_terms::cancel_additive_terms_semantic, simplifier_contains_var,
    simplifier_expand_expr, simplifier_zero_expr,
};

pub(crate) fn prepare_equation_for_strategy(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
) -> PreparedEquationResidual {
    cas_solver_core::solve_analysis::prepare_equation_for_strategy_with_state(
        simplifier,
        equation,
        var,
        simplifier_contains_var,
        |state, expr| state.simplify_for_solve(expr),
        |state, expr| {
            cas_solver_core::isolation_utils::try_recompose_pow_quotient(&mut state.context, expr)
        },
        |state, lhs, rhs| {
            cas_solver_core::cancel_common_terms::cancel_common_additive_terms(
                &mut state.context,
                lhs,
                rhs,
            )
            .map(|rewrite| (rewrite.new_lhs, rewrite.new_rhs))
        },
        |state, lhs, rhs| {
            cancel_additive_terms_semantic(state, lhs, rhs)
                .map(|rewrite| (rewrite.new_lhs, rewrite.new_rhs))
        },
        |state, lhs, rhs| state.context.add(Expr::Sub(lhs, rhs)),
        simplifier_expand_expr,
        |state, expr| state.expand(expr).0,
        |state, current, candidate, var_name| {
            cas_solver_core::solve_analysis::accept_residual_rewrite_candidate(
                &state.context,
                current,
                candidate,
                var_name,
            )
        },
        simplifier_zero_expr,
    )
}
