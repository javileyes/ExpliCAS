//! `taylor(...)` / `series(...)` command: Maclaurin (expansion point 0) series to a
//! requested order, delegating to the same analytic series engine the limit evaluator uses.

use cas_ast::{Context, Expr, ExprId};

use crate::define_rule;
use crate::rule::Rewrite;

/// Parse `taylor(f, x, n)` and `taylor(f, x, point, n)` (and the `series` alias).
/// Returns `(target, var_name, order)` for an expansion at `point = 0` with a
/// non-negative integer `order`. Non-zero expansion points and non-integer / negative
/// orders are declined (left as honest residuals).
fn try_extract_taylor_call(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, String, usize)> {
    if !ctx.is_call_named(expr, "taylor") && !ctx.is_call_named(expr, "series") {
        return None;
    }
    let Expr::Function(_, args) = ctx.get(expr) else {
        return None;
    };
    let args = args.clone();
    let (target, var_expr, point_expr, order_expr) = match args.len() {
        3 => (args[0], args[1], None, args[2]),
        4 => (args[0], args[1], Some(args[2]), args[3]),
        _ => return None,
    };

    let Expr::Variable(var_sym) = ctx.get(var_expr) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_sym).to_string();

    // Only the Maclaurin point (0) is supported; a non-zero expansion point is a residual.
    if let Some(point) = point_expr {
        if cas_math::numeric::as_i64(ctx, point) != Some(0) {
            return None;
        }
    }

    let order = cas_math::numeric::as_i64(ctx, order_expr)?;
    if order < 0 {
        return None;
    }
    Some((target, var_name, order as usize))
}

define_rule!(TaylorRule, "Taylor Series", |ctx, expr| {
    let (target, var_name, order) = try_extract_taylor_call(ctx, expr)?;
    let series =
        cas_math::limits_support::taylor_series_at_zero_expr(ctx, target, &var_name, order)?;
    Some(Rewrite::new(series).desc("serie de Taylor (Maclaurin)"))
});

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn extract(src: &str) -> Option<(String, usize)> {
        let mut ctx = Context::new();
        let expr = parse(src, &mut ctx).expect("parse");
        try_extract_taylor_call(&mut ctx, expr).map(|(_, v, o)| (v, o))
    }

    #[test]
    fn extracts_three_and_four_argument_forms() {
        assert_eq!(extract("taylor(exp(x), x, 4)"), Some(("x".to_string(), 4)));
        assert_eq!(
            extract("taylor(sin(x), x, 0, 5)"),
            Some(("x".to_string(), 5))
        );
        assert_eq!(extract("series(cos(t), t, 6)"), Some(("t".to_string(), 6)));
    }

    #[test]
    fn declines_nonzero_point_and_bad_order() {
        assert_eq!(extract("taylor(exp(x), x, 1, 4)"), None); // non-zero expansion point
        assert_eq!(extract("taylor(exp(x), x, -1)"), None); // negative order
        assert_eq!(extract("taylor(exp(x), x)"), None); // too few args
        assert_eq!(extract("diff(exp(x), x)"), None); // not a taylor/series call
    }

    #[test]
    fn expands_exponential_to_requested_order() {
        let mut ctx = Context::new();
        let expr = parse("taylor(exp(x), x, 0, 4)", &mut ctx).expect("parse");
        let (target, var, order) = try_extract_taylor_call(&mut ctx, expr).expect("taylor call");
        let series =
            cas_math::limits_support::taylor_series_at_zero_expr(&mut ctx, target, &var, order)
                .expect("series");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: series
            }
        );
        // 1 + x + x^2/2 + x^3/6 + x^4/24 — the coefficients must all be present.
        for needle in ["x^2", "x^3", "x^4"] {
            assert!(rendered.contains(needle), "{rendered}");
        }
    }

    #[test]
    fn expands_geometric_rational_function() {
        let mut ctx = Context::new();
        let expr = parse("taylor(1/(1-x), x, 0, 4)", &mut ctx).expect("parse");
        let (target, var, order) = try_extract_taylor_call(&mut ctx, expr).expect("taylor call");
        let series =
            cas_math::limits_support::taylor_series_at_zero_expr(&mut ctx, target, &var, order)
                .expect("series");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: series
            }
        );
        // 1/(1-x) = 1 + x + x^2 + x^3 + x^4.
        for needle in ["x^2", "x^3", "x^4"] {
            assert!(rendered.contains(needle), "{rendered}");
        }
    }
}
