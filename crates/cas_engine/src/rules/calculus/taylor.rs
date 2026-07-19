//! `taylor(...)` / `series(...)` command: Taylor/Maclaurin series to a requested order. The
//! Maclaurin (point 0) case delegates to the limit evaluator's analytic series engine; a
//! non-zero expansion point is built from the definition by repeated differentiation.

use cas_ast::{Context, Expr, ExprId};

use crate::define_rule;
use crate::rule::Rewrite;

/// Order used for the 2-argument form `taylor(f, x)` / `series(f, x)`, where the caller omits
/// the truncation order. Matches the common textbook default (a Maclaurin expansion up to `x^6`).
const DEFAULT_TAYLOR_ORDER: usize = 6;

/// Default TOTAL degree for the multivariate list form `taylor(f, [x,y])`: the
/// quadratic approximation — the multivariate textbook default (and it keeps
/// the default term count small: C(2+d,d)). An explicit order can go higher,
/// under the caps.
const DEFAULT_MULTIVAR_TAYLOR_ORDER: usize = 2;

/// Explicit ceiling for the requested order (F1, Fase 3): beyond it the command declines to an
/// honest residual instead of building an ever-deeper series tree. 32 sits far above any
/// curricular use; note that from roughly order 20 the simplifier's depth budget already leaves
/// some coefficients partially unfolded (cosmetic — the values are exact), which is the
/// documented ceiling behaviour rather than a bug.
const MAX_TAYLOR_ORDER: usize = 32;

/// Parse `taylor(f, x)`, `taylor(f, x, n)` and `taylor(f, x, point, n)` (and the `series` alias).
/// Returns `(target, var_name, point_expr, order)`. `point_expr` is the expansion point (the
/// literal `0` node when the 2/3-argument forms omit it); the order defaults to
/// [`DEFAULT_TAYLOR_ORDER`] in the 2-argument form. A negative/non-integer explicit order is
/// declined as an honest residual.
fn try_extract_taylor_call(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, String, ExprId, usize)> {
    if !ctx.is_call_named(expr, "taylor") && !ctx.is_call_named(expr, "series") {
        return None;
    }
    let Expr::Function(_, args) = ctx.get(expr) else {
        return None;
    };
    let args = args.clone();
    // 2 args: `taylor(f, x)` (Maclaurin, default order); 3 args: `taylor(f, x, n)` (order n at 0);
    // 4 args: `taylor(f, x, a, n)` (order n around a). The 2-argument form makes the most natural
    // invocation return a series instead of erroring on an unrecognized arity.
    let (target, var_expr, point_expr, order_expr) = match args.len() {
        2 => (args[0], args[1], None, None),
        3 => (args[0], args[1], None, Some(args[2])),
        4 => (args[0], args[1], Some(args[2]), Some(args[3])),
        _ => return None,
    };

    let Expr::Variable(var_sym) = ctx.get(var_expr) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_sym).to_string();

    let point = point_expr.unwrap_or_else(|| ctx.num(0));
    let order = match order_expr {
        Some(order_expr) => {
            let order = cas_math::numeric::as_i64(ctx, order_expr)?;
            if order < 0 || order as usize > MAX_TAYLOR_ORDER {
                return None;
            }
            order as usize
        }
        None => DEFAULT_TAYLOR_ORDER,
    };
    Some((target, var_name, point, order))
}

/// Parse the MULTIVARIATE forms (F2, Fase 3): `taylor(f, [x,y])` (Maclaurin,
/// default order), `taylor(f, [x,y], n)` and `taylor(f, [x,y], [a,b], n)` (and
/// the `series` alias). The variable list must be an n×1|1×n matrix of pure
/// Variables (anything else declines — the malformed-list pin); an explicit
/// point list must match the variable count and its entries may not mention the
/// expansion variables. Returns `(target, var_names, points, order)`.
fn try_extract_taylor_multivar_call(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<String>, Vec<ExprId>, usize)> {
    if !ctx.is_call_named(expr, "taylor") && !ctx.is_call_named(expr, "series") {
        return None;
    }
    let Expr::Function(_, args) = ctx.get(expr) else {
        return None;
    };
    let args = args.clone();
    let (target, vars_expr, points_expr, order_expr) = match args.len() {
        2 => (args[0], args[1], None, None),
        3 => (args[0], args[1], None, Some(args[2])),
        4 => (args[0], args[1], Some(args[2]), Some(args[3])),
        _ => return None,
    };
    let var_names = extract_pure_variable_list(ctx, vars_expr)?;
    let points = match points_expr {
        Some(points_expr) => {
            let Expr::Matrix { rows, cols, data } = ctx.get(points_expr) else {
                return None;
            };
            if (*rows != 1 && *cols != 1) || data.len() != var_names.len() {
                return None;
            }
            let points = data.clone();
            // A point entry mentioning an expansion variable is not a point.
            for &p in &points {
                if expr_mentions_any_var(ctx, p, &var_names) {
                    return None;
                }
            }
            points
        }
        None => {
            let zero = ctx.num(0);
            vec![zero; var_names.len()]
        }
    };
    let order = match order_expr {
        Some(order_expr) => {
            let order = cas_math::numeric::as_i64(ctx, order_expr)?;
            if order < 0 || order as usize > MAX_TAYLOR_ORDER {
                return None;
            }
            order as usize
        }
        None => DEFAULT_MULTIVAR_TAYLOR_ORDER,
    };
    Some((target, var_names, points, order))
}

/// An n×1 | 1×n matrix whose entries are ALL pure `Variable`s, as names.
fn extract_pure_variable_list(ctx: &Context, expr: ExprId) -> Option<Vec<String>> {
    let Expr::Matrix { rows, cols, data } = ctx.get(expr) else {
        return None;
    };
    if (*rows != 1 && *cols != 1) || data.is_empty() {
        return None;
    }
    let mut names = Vec::with_capacity(data.len());
    for &v in data {
        let Expr::Variable(sym) = ctx.get(v) else {
            return None;
        };
        names.push(ctx.sym_name(*sym).to_string());
    }
    Some(names)
}

/// True when `expr` mentions any of the named variables.
fn expr_mentions_any_var(ctx: &Context, expr: ExprId, names: &[String]) -> bool {
    let mut stack = vec![expr];
    while let Some(e) = stack.pop() {
        match ctx.get(e) {
            Expr::Variable(sym) => {
                let name = ctx.sym_name(*sym);
                if names.iter().any(|n| n == name) {
                    return true;
                }
            }
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Pow(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }
    }
    false
}

define_rule!(TaylorRule, "Taylor Series", |ctx, expr| {
    // Multivariate list form first (F2): its 2nd argument is a Matrix, which
    // the univariate extractor rejects, so the two forms cannot overlap.
    if let Some((target, var_names, points, order)) = try_extract_taylor_multivar_call(ctx, expr) {
        let series = cas_math::limits_support::taylor_multivar_series_expr(
            ctx, target, &var_names, &points, order,
        )?;
        return Some(Rewrite::new(series).desc("serie de Taylor multivariable (grado total)"));
    }
    let (target, var_name, point, order) = try_extract_taylor_call(ctx, expr)?;
    // The Maclaurin point uses the analytic engine (nicer closed forms) when it can. When that
    // declines (e.g. a fractional binomial (1+x)^α), AND for any non-zero point, expand from the
    // definition by repeated differentiation — which handles any function smooth at the point.
    if cas_math::numeric::as_i64(ctx, point) == Some(0) {
        if let Some(series) =
            cas_math::limits_support::taylor_series_at_zero_expr(ctx, target, &var_name, order)
        {
            return Some(Rewrite::new(series).desc("serie de Taylor (Maclaurin)"));
        }
    }
    let series = cas_math::limits_support::taylor_series_at_point_expr(
        ctx, target, &var_name, point, order,
    )?;
    Some(Rewrite::new(series).desc("serie de Taylor"))
});

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn extract(src: &str) -> Option<(String, usize)> {
        let mut ctx = Context::new();
        let expr = parse(src, &mut ctx).expect("parse");
        try_extract_taylor_call(&mut ctx, expr).map(|(_, v, _, o)| (v, o))
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
    fn accepts_nonzero_point_and_declines_bad_order() {
        assert_eq!(
            extract("taylor(exp(x), x, 1, 4)"),
            Some(("x".to_string(), 4))
        ); // point a = 1
        assert_eq!(extract("taylor(exp(x), x, -1)"), None); // negative order
        assert_eq!(extract("taylor(exp(x), x, 33)"), None); // beyond MAX_TAYLOR_ORDER
        assert_eq!(
            extract("taylor(exp(x), x, 32)"),
            Some(("x".to_string(), 32))
        ); // at the ceiling
        assert_eq!(extract("diff(exp(x), x)"), None); // not a taylor/series call
        assert_eq!(extract("taylor(exp(x))"), None); // too few args (no variable)
    }

    #[test]
    fn multivar_extractor_validates_list_point_and_order() {
        let mut ctx = Context::new();
        let extract_mv = |ctx: &mut Context, src: &str| {
            let expr = parse(src, ctx).expect(src);
            try_extract_taylor_multivar_call(ctx, expr).map(|(_, v, p, o)| (v, p.len(), o))
        };
        assert_eq!(
            extract_mv(&mut ctx, "taylor(e^(x+y), [x,y], [0,0], 3)"),
            Some((vec!["x".to_string(), "y".to_string()], 2, 3))
        );
        // 2-args: orden default multivar (la aproximación cuadrática).
        assert_eq!(
            extract_mv(&mut ctx, "taylor(e^(x+y), [x,y])"),
            Some((
                vec!["x".to_string(), "y".to_string()],
                2,
                DEFAULT_MULTIVAR_TAYLOR_ORDER
            ))
        );
        // Lista malformada (no-Variables) → decline (pin del scoping).
        assert_eq!(extract_mv(&mut ctx, "taylor(x*y, [x, 2*y], 2)"), None);
        // Punto que menciona una variable de expansión → decline.
        assert_eq!(
            extract_mv(&mut ctx, "taylor(e^(x+y), [x,y], [x,0], 2)"),
            None
        );
        // Longitud de punto distinta → decline.
        assert_eq!(extract_mv(&mut ctx, "taylor(e^(x+y), [x,y], [0], 2)"), None);
    }

    #[test]
    fn two_argument_form_defaults_the_order() {
        // `taylor(f, x)` / `series(f, x)` (no order) is a Maclaurin expansion to the default order.
        assert_eq!(
            extract("taylor(exp(x), x)"),
            Some(("x".to_string(), DEFAULT_TAYLOR_ORDER))
        );
        assert_eq!(
            extract("series(sin(t), t)"),
            Some(("t".to_string(), DEFAULT_TAYLOR_ORDER))
        );
    }

    #[test]
    fn expands_around_a_nonzero_point() {
        let mut ctx = Context::new();
        // d/dx of the Taylor of exp around 1 must reproduce the integrand value e at x=1.
        let expr = parse("taylor(exp(x), x, 1, 3)", &mut ctx).expect("parse");
        let (target, var, point, order) =
            try_extract_taylor_call(&mut ctx, expr).expect("taylor call");
        let series = cas_math::limits_support::taylor_series_at_point_expr(
            &mut ctx, target, &var, point, order,
        )
        .expect("series");
        let rendered = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: series
            }
        );
        // The expansion is in powers of (x − 1), with the e coefficient.
        assert!(rendered.contains("x - 1"), "{rendered}");
        assert!(rendered.contains('e'), "{rendered}");
    }

    #[test]
    fn expands_exponential_to_requested_order() {
        let mut ctx = Context::new();
        let expr = parse("taylor(exp(x), x, 0, 4)", &mut ctx).expect("parse");
        let (target, var, _point, order) =
            try_extract_taylor_call(&mut ctx, expr).expect("taylor call");
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
        let (target, var, _point, order) =
            try_extract_taylor_call(&mut ctx, expr).expect("taylor call");
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
