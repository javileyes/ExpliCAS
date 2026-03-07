use cas_ast::{Context, ExprId};

use super::expr::display_expr;

pub(super) fn display_discrete_solution_set(ctx: &Context, exprs: &[ExprId]) -> String {
    let s: Vec<String> = exprs.iter().map(|e| display_expr(ctx, *e)).collect();
    format!("{{ {} }}", s.join(", "))
}
