use crate::solve_runtime_adapters::{
    simplifier_contains_var, simplifier_context_mut, simplifier_expand_expr, simplifier_zero_expr,
};
use crate::Simplifier;
use cas_ast::Equation;
use cas_solver_core::solve_analysis::PreparedEquationResidual;

pub(crate) fn prepare_equation_for_strategy(
    simplifier: &mut Simplifier,
    equation: &Equation,
    var: &str,
) -> PreparedEquationResidual {
    cas_solver_core::solve_runtime_pipeline_preflight_equation_runtime::prepare_equation_for_strategy_with_default_structural_recompose_and_cancel_with_state(
        simplifier,
        equation,
        var,
        simplifier_contains_var,
        |state, expr| state.simplify_for_solve(expr),
        |state, lhs, rhs| {
            crate::cancel_runtime::cancel_additive_terms_semantic(state, lhs, rhs)
                .map(|rewrite| (rewrite.new_lhs, rewrite.new_rhs))
        },
        simplifier_expand_expr,
        |state, expr| state.expand(expr).0,
        |state| &state.context,
        simplifier_context_mut,
        simplifier_zero_expr,
    )
}
