use cas_ast::{BoundType, Context, Interval};
use cas_formatter::DisplayExpr;

pub(super) fn display_interval(context: &Context, interval: &Interval) -> String {
    let min_bracket = match interval.min_type {
        BoundType::Open => "(",
        BoundType::Closed => "[",
    };
    let max_bracket = match interval.max_type {
        BoundType::Open => ")",
        BoundType::Closed => "]",
    };
    format!(
        "{}{}, {}{}",
        min_bracket,
        DisplayExpr {
            context,
            id: interval.min
        },
        DisplayExpr {
            context,
            id: interval.max
        },
        max_bracket
    )
}
