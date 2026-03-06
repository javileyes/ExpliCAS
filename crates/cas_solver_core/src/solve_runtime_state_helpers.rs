//! Shared context/state helpers for solve runtime adapters.

use cas_ast::{symbol::SymbolId, Context, ExprId};

pub fn contains_var_in_context(ctx: &Context, expr: ExprId, var: &str) -> bool {
    crate::isolation_utils::contains_var(ctx, expr, var)
}

pub fn render_expr_in_context(ctx: &Context, expr: ExprId) -> String {
    cas_formatter::render_expr(ctx, expr)
}

pub fn zero_expr_in_context(ctx: &mut Context) -> ExprId {
    ctx.num(0)
}

pub fn is_known_negative_in_context(ctx: &Context, expr: ExprId) -> bool {
    crate::isolation_utils::is_known_negative(ctx, expr)
}

pub fn sym_name_as_string(ctx: &Context, fn_symbol: SymbolId) -> String {
    ctx.sym_name(fn_symbol).to_string()
}

pub fn simplify_rhs_with_step_pairs(
    simplified_rhs: ExprId,
    sim_steps: Vec<crate::step_model::Step>,
) -> (ExprId, Vec<(String, ExprId)>) {
    let entries = sim_steps
        .into_iter()
        .map(|step| (step.description, step.after))
        .collect::<Vec<_>>();
    (simplified_rhs, entries)
}
