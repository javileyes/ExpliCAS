use cas_ast::{Context, Expr, ExprId};
use cas_formatter::LaTeXExpr;

pub(super) fn render_residual_solution(ctx: &Context, expr: ExprId) -> String {
    let expr_latex = LaTeXExpr {
        context: ctx,
        id: expr,
    }
    .to_latex();
    // Scout cycle-3 honesty contract: a residual that is itself a
    // `solve(...)` call is self-describing — it already carries the full
    // relation (operator included), matching the `integrate(...)` residual
    // convention. The old "Solve: … = 0" wrapper appended a dangling "= 0"
    // that misdescribed inequalities.
    if matches!(ctx.get(expr), Expr::Function(name, _) if ctx.sym_name(*name) == "solve") {
        expr_latex
    } else {
        format!(r"\text{{Solve: }} {} = 0", expr_latex)
    }
}
