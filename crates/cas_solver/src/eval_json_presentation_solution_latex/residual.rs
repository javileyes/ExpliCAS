use cas_ast::{Context, ExprId};
use cas_formatter::LaTeXExpr;

pub(super) fn render_residual_solution(ctx: &Context, expr: ExprId) -> String {
    let expr_latex = LaTeXExpr {
        context: ctx,
        id: expr,
    }
    .to_latex();
    format!(r"\text{{Solve: }} {} = 0", expr_latex)
}
