use cas_ast::{Context, ExprId};
use cas_formatter::LaTeXExpr;

pub(super) fn render_expr_to_latex(context: &Context, expr: ExprId) -> String {
    LaTeXExpr { context, id: expr }.to_latex()
}
