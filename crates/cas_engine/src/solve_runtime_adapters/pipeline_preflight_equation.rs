use crate::engine::Simplifier;
use crate::solve_runtime_adapters::{
    simplifier_contains_var, simplifier_context_mut, simplifier_expand_expr, simplifier_zero_expr,
};
use cas_ast::{Equation, Expr};
use cas_solver_core::solve_analysis::PreparedEquationResidual;

pub(crate) fn prepare_equation_for_strategy(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
) -> PreparedEquationResidual {
    cas_solver_core::solve_runtime_flow::prepare_equation_for_strategy_with_default_structural_recompose_and_cancel_and_default_residual_acceptance_with_state(
        simplifier,
        equation,
        var,
        simplifier_contains_var,
        |state, expr| state.simplify_for_solve(expr),
        |state, lhs, rhs| {
            crate::cancel_common_terms_runtime::cancel_additive_terms_semantic(state, lhs, rhs)
                .map(|rewrite| (rewrite.new_lhs, rewrite.new_rhs))
        },
        |state, lhs, rhs| state.context.add(Expr::Sub(lhs, rhs)),
        simplifier_expand_expr,
        |state, expr| state.expand(expr).0,
        |state| &state.context,
        simplifier_context_mut,
        simplifier_zero_expr,
    )
}
