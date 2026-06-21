use cas_ast::{Context, Expr, ExprId};

/// Pull perfect-square numeric factors out of a square-root expression.
///
/// Converts `sqrt(k * expr) -> m * sqrt(expr)` when `k = m^2`.
/// Also handles additive forms with common factor, e.g.:
/// `sqrt(4*a + 4*b) -> 2*sqrt(a + b)`.
pub fn pull_square_from_sqrt(ctx: &mut Context, sqrt_expr: ExprId) -> ExprId {
    let expr_data = ctx.get(sqrt_expr).clone();
    let Expr::Pow(base, exp) = expr_data else {
        return sqrt_expr;
    };

    let is_half = matches!(ctx.get(exp), Expr::Div(n, d)
        if matches!(ctx.get(*n), Expr::Number(num) if num == &num_rational::BigRational::from_integer(1.into()))
        && matches!(ctx.get(*d), Expr::Number(den) if den == &num_rational::BigRational::from_integer(2.into()))
    );
    if !is_half {
        return sqrt_expr;
    }

    let (factor, rest) = split_numeric_factor(ctx, base);
    let Some(k) = factor else {
        return sqrt_expr;
    };
    if k <= 0 {
        return sqrt_expr;
    }

    let sqrt_k = (k as f64).sqrt();
    let m = sqrt_k.round() as i64;
    if m * m != k || m == 1 {
        return sqrt_expr;
    }

    // Whether `rest` already has the factor `k` removed.
    //   - Additive base (`16 + 4·e`): `split_numeric_factor` returns the WHOLE sum as
    //     `rest`, so we must divide by `k` to leave `√((16+4e)/4) = √(4+e)` under the root.
    //   - Multiplicative base (`4·(4+e)`): `rest` is ALREADY the cofactor `(4+e)`, so dividing
    //     again would halve the radical (`2·√((4+e)/4) = √(4+e)`, a wrong, smaller value) — the
    //     factor must be taken as-is so the result is `2·√(4+e)`.
    let base_is_additive = matches!(ctx.get(base), Expr::Add(_, _) | Expr::Sub(_, _));
    let actual_rest = if base_is_additive {
        let k_expr = ctx.add(Expr::Number(num_rational::BigRational::from_integer(
            k.into(),
        )));
        ctx.add(Expr::Div(rest, k_expr))
    } else {
        rest
    };

    let one = ctx.num(1);
    let two = ctx.num(2);
    let half = ctx.add(Expr::Div(one, two));
    let sqrt_rest = ctx.add(Expr::Pow(actual_rest, half));
    let m_expr = ctx.add(Expr::Number(num_rational::BigRational::from_integer(
        m.into(),
    )));
    ctx.add(Expr::Mul(m_expr, sqrt_rest))
}

fn split_numeric_factor(ctx: &Context, expr: ExprId) -> (Option<i64>, ExprId) {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            if let Expr::Number(n) = ctx.get(*l) {
                if n.is_integer() {
                    if let Ok(k) = i64::try_from(n.to_integer()) {
                        return (Some(k), *r);
                    }
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if n.is_integer() {
                    if let Ok(k) = i64::try_from(n.to_integer()) {
                        return (Some(k), *l);
                    }
                }
            }
            (None, expr)
        }
        Expr::Add(_, _) | Expr::Sub(_, _) => {
            let terms = collect_additive_terms(ctx, expr);
            if terms.is_empty() {
                return (None, expr);
            }

            let coeffs: Vec<i64> = terms
                .iter()
                .filter_map(|(id, _)| get_term_coefficient(ctx, *id))
                .collect();
            if coeffs.len() != terms.len() || coeffs.is_empty() {
                return (None, expr);
            }

            let gcd = coeffs.iter().fold(0i64, |acc, &c| gcd_i64(acc, c.abs()));
            if gcd <= 1 {
                return (None, expr);
            }

            let sqrt_gcd = (gcd as f64).sqrt();
            let m = sqrt_gcd.round() as i64;
            if m * m != gcd {
                return (None, expr);
            }

            (Some(gcd), expr)
        }
        _ => (None, expr),
    }
}

fn get_term_coefficient(ctx: &Context, expr: ExprId) -> Option<i64> {
    match ctx.get(expr) {
        Expr::Mul(l, r) => {
            if let Expr::Number(n) = ctx.get(*l) {
                if n.is_integer() {
                    return i64::try_from(n.to_integer()).ok();
                }
            }
            if let Expr::Number(n) = ctx.get(*r) {
                if n.is_integer() {
                    return i64::try_from(n.to_integer()).ok();
                }
            }
            Some(1)
        }
        Expr::Neg(inner) => get_term_coefficient(ctx, *inner).map(|c| -c),
        Expr::Number(n) if n.is_integer() => i64::try_from(n.to_integer()).ok(),
        _ => Some(1),
    }
}

fn collect_additive_terms(ctx: &Context, expr: ExprId) -> Vec<(ExprId, bool)> {
    let mut terms = Vec::new();
    collect_additive_terms_recursive(ctx, expr, true, &mut terms);
    terms
}

fn collect_additive_terms_recursive(
    ctx: &Context,
    expr: ExprId,
    positive: bool,
    terms: &mut Vec<(ExprId, bool)>,
) {
    match ctx.get(expr) {
        Expr::Add(l, r) => {
            collect_additive_terms_recursive(ctx, *l, positive, terms);
            collect_additive_terms_recursive(ctx, *r, positive, terms);
        }
        Expr::Sub(l, r) => {
            collect_additive_terms_recursive(ctx, *l, positive, terms);
            collect_additive_terms_recursive(ctx, *r, !positive, terms);
        }
        Expr::Neg(inner) => {
            collect_additive_terms_recursive(ctx, *inner, !positive, terms);
        }
        _ => terms.push((expr, positive)),
    }
}

fn gcd_i64(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd_i64(b, a % b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pulls_square_factor_from_simple_sqrt() {
        let mut ctx = Context::new();
        let four = ctx.num(4);
        let x = ctx.var("x");
        let mul = ctx.add(Expr::Mul(four, x));
        let one = ctx.num(1);
        let two = ctx.num(2);
        let half = ctx.add(Expr::Div(one, two));
        let sqrt = ctx.add(Expr::Pow(mul, half));
        let out = pull_square_from_sqrt(&mut ctx, sqrt);
        assert!(matches!(ctx.get(out), Expr::Mul(_, _)));
    }

    /// Numeric value of a constant `expr` (`e` folds to `2.718…`), for asserting the
    /// radical was preserved (not silently halved). `None` if the expr is not foldable.
    fn approx(ctx: &Context, expr: ExprId) -> Option<f64> {
        let map = std::collections::HashMap::new();
        cas_math::evaluator_f64::eval_f64(ctx, expr, &map)
    }

    #[test]
    fn pulls_square_from_factored_additive_base_keeps_coefficient() {
        // REGRESSION: `√(4·(4+e))` must be `2·√(4+e)` (≈ 5.18), NOT `√(4+e)` (≈ 2.59).
        // The discriminant of `x²−4x−e` simplifies to the FACTORED form `4·(4+e)`; the old
        // code divided the already-extracted cofactor `(4+e)` by `4` a second time, halving the
        // radical and yielding wrong quadratic roots (`(4+√(4+e))/2` instead of `2+√(4+e)`).
        let mut ctx = Context::new();
        let four = ctx.num(4);
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let four_plus_e = ctx.add(Expr::Add(four, e));
        let four2 = ctx.num(4);
        let base = ctx.add(Expr::Mul(four2, four_plus_e)); // 4·(4+e)
        let one = ctx.num(1);
        let two = ctx.num(2);
        let half = ctx.add(Expr::Div(one, two));
        let sqrt = ctx.add(Expr::Pow(base, half));

        let out = pull_square_from_sqrt(&mut ctx, sqrt);
        let got = approx(&ctx, out).expect("foldable");
        let want = (4.0 * (4.0 + std::f64::consts::E)).sqrt(); // √(4·(4+e)) ≈ 5.18
        assert!(
            (got - want).abs() < 1e-9,
            "expected √(4·(4+e)) ≈ {want}, got {got} (radical was halved)"
        );
    }

    #[test]
    fn pulls_square_from_unfactored_additive_base_divides_once() {
        // `√(16+4e)` (unfactored sum) must still reduce to `2·√(4+e)` ≈ 5.18 — the Add branch of
        // `split_numeric_factor` returns the whole sum, so dividing by the gcd `4` once is correct.
        let mut ctx = Context::new();
        let sixteen = ctx.num(16);
        let four = ctx.num(4);
        let e = ctx.add(Expr::Constant(cas_ast::Constant::E));
        let four_e = ctx.add(Expr::Mul(four, e));
        let base = ctx.add(Expr::Add(sixteen, four_e)); // 16 + 4e
        let one = ctx.num(1);
        let two = ctx.num(2);
        let half = ctx.add(Expr::Div(one, two));
        let sqrt = ctx.add(Expr::Pow(base, half));

        let out = pull_square_from_sqrt(&mut ctx, sqrt);
        let got = approx(&ctx, out).expect("foldable");
        let want = (16.0 + 4.0 * std::f64::consts::E).sqrt();
        assert!(
            (got - want).abs() < 1e-9,
            "expected √(16+4e) ≈ {want}, got {got}"
        );
    }
}
