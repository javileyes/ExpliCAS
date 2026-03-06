//! Shared runtime state adapter helpers for solve runtime wrappers.

use crate::external_proof::map_external_nonzero_status_with;
use crate::solve_runtime_state_helpers as state_helpers;
use cas_ast::symbol::SymbolId;
use cas_ast::ExprId;

/// Runtime state contract used by solve runtime adapters.
pub trait RuntimeSolveAdapterState {
    fn runtime_context(&self) -> &cas_ast::Context;
    fn runtime_context_mut(&mut self) -> &mut cas_ast::Context;
    fn runtime_simplify_expr(&mut self, expr: ExprId) -> ExprId;
    fn runtime_expand_expr(&mut self, expr: ExprId) -> ExprId;
    fn runtime_expand_full_expr(&mut self, expr: ExprId) -> ExprId;
    fn runtime_collect_steps(&self) -> bool;
    fn runtime_set_collect_steps(&mut self, collect: bool);
    fn runtime_simplify_for_solve(&mut self, expr: ExprId) -> ExprId;
    fn runtime_simplify_with_options_expr(
        &mut self,
        expr: ExprId,
        options: crate::simplify_options::SimplifyOptions,
    ) -> ExprId;
    fn runtime_are_equivalent(&mut self, lhs: ExprId, rhs: ExprId) -> bool;
    fn runtime_clear_blocked_hints(&mut self);
    fn runtime_prove_nonzero(&self, expr: ExprId) -> crate::domain_proof::Proof;
    fn runtime_simplify_with_steps(
        &mut self,
        expr: ExprId,
    ) -> (ExprId, Vec<crate::step_model::Step>);
}

pub fn simplifier_context<S: RuntimeSolveAdapterState>(simplifier: &mut S) -> &cas_ast::Context {
    simplifier.runtime_context()
}

pub fn simplifier_context_mut<S: RuntimeSolveAdapterState>(
    simplifier: &mut S,
) -> &mut cas_ast::Context {
    simplifier.runtime_context_mut()
}

pub fn simplifier_contains_var<S: RuntimeSolveAdapterState>(
    simplifier: &mut S,
    expr: ExprId,
    var: &str,
) -> bool {
    state_helpers::contains_var_in_context(simplifier.runtime_context(), expr, var)
}

pub fn simplifier_simplify_expr<S: RuntimeSolveAdapterState>(
    simplifier: &mut S,
    expr: ExprId,
) -> ExprId {
    simplifier.runtime_simplify_expr(expr)
}

pub fn simplifier_expand_expr<S: RuntimeSolveAdapterState>(
    simplifier: &mut S,
    expr: ExprId,
) -> ExprId {
    simplifier.runtime_expand_expr(expr)
}

pub fn simplifier_expand_full_expr<S: RuntimeSolveAdapterState>(
    simplifier: &mut S,
    expr: ExprId,
) -> ExprId {
    simplifier.runtime_expand_full_expr(expr)
}

pub fn simplifier_render_expr<S: RuntimeSolveAdapterState>(
    simplifier: &mut S,
    expr: ExprId,
) -> String {
    state_helpers::render_expr_in_context(simplifier.runtime_context(), expr)
}

pub fn context_render_expr(ctx: &cas_ast::Context, expr: ExprId) -> String {
    state_helpers::render_expr_in_context(ctx, expr)
}

pub fn simplifier_zero_expr<S: RuntimeSolveAdapterState>(simplifier: &mut S) -> ExprId {
    state_helpers::zero_expr_in_context(simplifier.runtime_context_mut())
}

pub fn simplifier_collect_steps<S: RuntimeSolveAdapterState>(simplifier: &mut S) -> bool {
    simplifier.runtime_collect_steps()
}

pub fn simplifier_set_collect_steps<S: RuntimeSolveAdapterState>(
    simplifier: &mut S,
    collect: bool,
) {
    simplifier.runtime_set_collect_steps(collect);
}

pub fn simplifier_simplify_for_solve<S: RuntimeSolveAdapterState>(
    simplifier: &mut S,
    expr: ExprId,
) -> ExprId {
    simplifier.runtime_simplify_for_solve(expr)
}

pub fn simplifier_simplify_with_options_expr<S: RuntimeSolveAdapterState>(
    simplifier: &mut S,
    expr: ExprId,
    options: &crate::simplify_options::SimplifyOptions,
) -> ExprId {
    simplifier.runtime_simplify_with_options_expr(expr, options.clone())
}

pub fn simplifier_are_equivalent<S: RuntimeSolveAdapterState>(
    simplifier: &mut S,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    simplifier.runtime_are_equivalent(lhs, rhs)
}

pub fn simplifier_clear_blocked_hints<S: RuntimeSolveAdapterState>(simplifier: &mut S) {
    simplifier.runtime_clear_blocked_hints();
}

pub fn simplifier_prove_nonzero_status<S: RuntimeSolveAdapterState>(
    simplifier: &mut S,
    expr: ExprId,
) -> crate::linear_solution::NonZeroStatus {
    map_external_nonzero_status_with(
        simplifier.runtime_prove_nonzero(expr),
        |proof| {
            matches!(
                proof,
                crate::domain_proof::Proof::Proven | crate::domain_proof::Proof::ProvenImplicit
            )
        },
        |proof| matches!(proof, crate::domain_proof::Proof::Disproven),
    )
}

pub fn simplifier_is_known_negative<S: RuntimeSolveAdapterState>(
    simplifier: &mut S,
    expr: ExprId,
) -> bool {
    state_helpers::is_known_negative_in_context(simplifier.runtime_context(), expr)
}

pub fn simplify_rhs_with_step_pairs<S: RuntimeSolveAdapterState>(
    simplifier: &mut S,
    rhs_expr: ExprId,
) -> (ExprId, Vec<(String, ExprId)>) {
    let (simplified_rhs, sim_steps) = simplifier.runtime_simplify_with_steps(rhs_expr);
    state_helpers::simplify_rhs_with_step_pairs(simplified_rhs, sim_steps)
}

pub fn sym_name_as_string<S: RuntimeSolveAdapterState>(
    simplifier: &mut S,
    fn_symbol: SymbolId,
) -> String {
    state_helpers::sym_name_as_string(simplifier.runtime_context(), fn_symbol)
}
