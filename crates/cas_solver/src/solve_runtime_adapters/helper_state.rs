use crate::Simplifier;
use cas_ast::ExprId;

// Solver facade keeps its own algebraic expand entrypoint (`crate::expand`),
// so this adapter remains local while the rest of the state helpers come from
// `cas_solver_core::solve_runtime_adapter_state_runtime`.
pub(crate) fn simplifier_expand_expr(simplifier: &mut Simplifier, expr: ExprId) -> ExprId {
    crate::expand(&mut simplifier.context, expr)
}
