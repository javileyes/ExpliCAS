use cas_ast::{BoundType, Context, Interval};
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
    // Respect each endpoint's bound type — `[`/`]` closed, `(`/`)` open — so a
    // half-open interval (`[0, 4)`) and infinite ends (`(-∞, 1)`) render correctly
    // instead of the previous hardcoded `\left[ … \right]`.
    let left = if interval.min_type == BoundType::Closed {
        '['
    } else {
        '('
    };
    let right = if interval.max_type == BoundType::Closed {
        ']'
    } else {
        ')'
    };
    format!(r"\left{}{}, {}\right{}", left, min_latex, max_latex, right)
}

pub(super) fn render_interval_union(ctx: &Context, intervals: &[Interval]) -> String {
    let parts: Vec<String> = intervals
        .iter()
        .map(|int| render_continuous_interval(ctx, int))
        .collect();
    parts.join(r" \cup ")
}
