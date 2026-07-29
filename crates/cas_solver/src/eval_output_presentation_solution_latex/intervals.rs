use cas_ast::{BoundType, Context, Interval};
use cas_formatter::{LaTeXExprStyled, StylePreferences};

pub(super) fn render_continuous_interval(
    ctx: &Context,
    interval: &Interval,
    style: &StylePreferences,
) -> String {
    let min_latex = LaTeXExprStyled {
        context: ctx,
        id: interval.min,
        style_prefs: style,
    }
    .to_latex();
    let max_latex = LaTeXExprStyled {
        context: ctx,
        id: interval.max,
        style_prefs: style,
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

pub(super) fn render_interval_union(
    ctx: &Context,
    intervals: &[Interval],
    style: &StylePreferences,
) -> String {
    let parts: Vec<String> = intervals
        .iter()
        .map(|int| render_continuous_interval(ctx, int, style))
        .collect();
    parts.join(r" \cup ")
}
