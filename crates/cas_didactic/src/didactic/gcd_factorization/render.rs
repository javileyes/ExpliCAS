use cas_ast::{Context, ExprId};
use cas_formatter::DisplayExpr;

pub(super) fn render_expr(context: &Context, id: ExprId) -> String {
    format!("{}", DisplayExpr { context, id })
}

pub(super) fn format_division_expr(numerator: &str, denominator: &str) -> String {
    format!(
        "{} / {}",
        maybe_parenthesize(numerator),
        maybe_parenthesize(denominator)
    )
}

fn maybe_parenthesize(expr: &str) -> String {
    if expr.contains('+') || expr.contains('-') || expr.contains(' ') {
        format!("({})", expr)
    } else {
        expr.to_string()
    }
}
