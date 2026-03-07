use cas_ast::{BoundType, Context, Interval};
use cas_formatter::LaTeXExpr;

pub(super) fn render_interval_to_latex(context: &Context, interval: &Interval) -> String {
    let left = match interval.min_type {
        BoundType::Open => "(",
        BoundType::Closed => "[",
    };
    let right = match interval.max_type {
        BoundType::Open => ")",
        BoundType::Closed => "]",
    };
    let min_latex = LaTeXExpr {
        context,
        id: interval.min,
    }
    .to_latex();
    let max_latex = LaTeXExpr {
        context,
        id: interval.max,
    }
    .to_latex();
    format!(r"{}{}, {}{}", left, min_latex, max_latex, right)
}
