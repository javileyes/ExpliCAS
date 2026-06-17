use cas_ast::{BoundType, Context, Interval};
use cas_formatter::DisplayExpr;

pub(super) fn format_output_interval(ctx: &Context, interval: &Interval) -> String {
    // Respect the interval's open/closed bounds: `x > 1` is `(1, ∞)`, not
    // `[1, ∞]` — a closed bracket would wrongly claim the endpoint (and ∞,
    // which is never a member) belongs to the solution set.
    let open = match interval.min_type {
        BoundType::Open => "(",
        BoundType::Closed => "[",
    };
    let close = match interval.max_type {
        BoundType::Open => ")",
        BoundType::Closed => "]",
    };
    format!(
        "{}{}, {}{}",
        open,
        DisplayExpr {
            context: ctx,
            id: interval.min
        },
        DisplayExpr {
            context: ctx,
            id: interval.max
        },
        close
    )
}

#[cfg(test)]
mod tests {
    use super::format_output_interval;
    use cas_ast::{BoundType, Context, Interval};

    #[test]
    fn renders_open_and_closed_bounds_distinctly() {
        let mut ctx = Context::new();
        let one = ctx.num(1);
        let three = ctx.num(3);

        // Strict `x > 1` style: open at both the finite bound and (always) ∞.
        let open = Interval {
            min: one,
            min_type: BoundType::Open,
            max: three,
            max_type: BoundType::Open,
        };
        assert_eq!(format_output_interval(&ctx, &open), "(1, 3)");

        // `1 ≤ x < 3`: closed-open must not collapse to the same string.
        let half_open = Interval {
            min: one,
            min_type: BoundType::Closed,
            max: three,
            max_type: BoundType::Open,
        };
        assert_eq!(format_output_interval(&ctx, &half_open), "[1, 3)");

        let closed = Interval {
            min: one,
            min_type: BoundType::Closed,
            max: three,
            max_type: BoundType::Closed,
        };
        assert_eq!(format_output_interval(&ctx, &closed), "[1, 3]");
    }
}
