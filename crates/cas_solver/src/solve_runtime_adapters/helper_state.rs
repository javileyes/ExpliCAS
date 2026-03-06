use crate::Simplifier;
use cas_ast::symbol::SymbolId;
use cas_ast::ExprId;
use cas_solver_core::external_proof::map_external_nonzero_status_with;
use cas_solver_core::solve_runtime_state_helpers as state_helpers;

pub(crate) fn simplifier_context(simplifier: &mut Simplifier) -> &cas_ast::Context {
    &simplifier.context
}

pub(crate) fn simplifier_context_mut(simplifier: &mut Simplifier) -> &mut cas_ast::Context {
    &mut simplifier.context
}

pub(crate) fn simplifier_contains_var(
    simplifier: &mut Simplifier,
    expr: ExprId,
    var: &str,
) -> bool {
    state_helpers::contains_var_in_context(&simplifier.context, expr, var)
}

pub(crate) fn simplifier_simplify_expr(simplifier: &mut Simplifier, expr: ExprId) -> ExprId {
    simplifier.simplify(expr).0
}

pub(crate) fn simplifier_expand_expr(simplifier: &mut Simplifier, expr: ExprId) -> ExprId {
    crate::expand(&mut simplifier.context, expr)
}

pub(crate) fn simplifier_render_expr(simplifier: &mut Simplifier, expr: ExprId) -> String {
    state_helpers::render_expr_in_context(&simplifier.context, expr)
}

pub(crate) fn context_render_expr(ctx: &cas_ast::Context, expr: ExprId) -> String {
    state_helpers::render_expr_in_context(ctx, expr)
}

pub(crate) fn simplifier_zero_expr(simplifier: &mut Simplifier) -> ExprId {
    state_helpers::zero_expr_in_context(&mut simplifier.context)
}

pub(crate) fn simplifier_collect_steps(simplifier: &mut Simplifier) -> bool {
    simplifier.collect_steps()
}

pub(crate) fn simplifier_prove_nonzero_status(
    simplifier: &mut Simplifier,
    expr: ExprId,
) -> cas_solver_core::linear_solution::NonZeroStatus {
    map_external_nonzero_status_with(
        crate::proof_runtime::prove_nonzero(&simplifier.context, expr),
        |proof| matches!(proof, crate::Proof::Proven | crate::Proof::ProvenImplicit),
        |proof| matches!(proof, crate::Proof::Disproven),
    )
}

pub(crate) fn simplifier_is_known_negative(simplifier: &mut Simplifier, expr: ExprId) -> bool {
    state_helpers::is_known_negative_in_context(&simplifier.context, expr)
}

pub(crate) fn simplify_rhs_with_step_pairs(
    simplifier: &mut Simplifier,
    rhs_expr: ExprId,
) -> (ExprId, Vec<(String, ExprId)>) {
    let (simplified_rhs, sim_steps) = simplifier.simplify(rhs_expr);
    state_helpers::simplify_rhs_with_step_pairs(simplified_rhs, sim_steps)
}

pub(crate) fn sym_name_as_string(simplifier: &mut Simplifier, fn_symbol: SymbolId) -> String {
    state_helpers::sym_name_as_string(&simplifier.context, fn_symbol)
}
