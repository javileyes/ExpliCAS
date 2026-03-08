use cas_ast::{Context, Interval};
use cas_formatter::LaTeXExpr;

pub(super) fn render_continuous_interval(ctx: &Context, interval: &Interval) -> String {
    let min_latex = LaTeXExpr {
        context: ctx,
        id: interval.min,
    }
    .to_latex();
    let max_latex = LaTeXExpr {
        context: ctx,
        id: interval.max,
    }
    .to_latex();
    format!(r"\left[{}, {}\right]", min_latex, max_latex)
}

pub(super) fn render_interval_union(ctx: &Context, intervals: &[Interval]) -> String {
    let parts: Vec<String> = intervals
        .iter()
        .map(|int| render_continuous_interval(ctx, int))
        .collect();
    parts.join(r" \cup ")
}
