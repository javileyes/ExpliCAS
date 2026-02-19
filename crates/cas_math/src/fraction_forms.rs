//! Fraction-shape helpers over AST expressions.

use cas_ast::{Context, Expr, ExprId};

fn as_div(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Div(n, d) => Some((*n, *d)),
        _ => None,
    }
}

fn as_mul(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr) {
        Expr::Mul(l, r) => Some((*l, *r)),
        _ => None,
    }
}

fn exprs_equal(ctx: &Context, a: ExprId, b: ExprId) -> bool {
    if a == b {
        return true;
    }
    match (ctx.get(a), ctx.get(b)) {
        (Expr::Number(n1), Expr::Number(n2)) => n1 == n2,
        (Expr::Variable(v1), Expr::Variable(v2)) => v1 == v2,
        (Expr::Constant(c1), Expr::Constant(c2)) => c1 == c2,
        (Expr::Add(l1, r1), Expr::Add(l2, r2))
        | (Expr::Sub(l1, r1), Expr::Sub(l2, r2))
        | (Expr::Mul(l1, r1), Expr::Mul(l2, r2))
        | (Expr::Div(l1, r1), Expr::Div(l2, r2))
        | (Expr::Pow(l1, r1), Expr::Pow(l2, r2)) => {
            exprs_equal(ctx, *l1, *l2) && exprs_equal(ctx, *r1, *r2)
        }
        (Expr::Neg(e1), Expr::Neg(e2)) => exprs_equal(ctx, *e1, *e2),
        _ => false,
    }
}

fn try_extract_factor(ctx: &Context, expr: ExprId, factor: ExprId) -> Option<ExprId> {
    if expr == factor {
        return None;
    }

    if let Some((l, r)) = as_mul(ctx, expr) {
        if l == factor || exprs_equal(ctx, l, factor) {
            return Some(r);
        }
        if r == factor || exprs_equal(ctx, r, factor) {
            return Some(l);
        }
    }

    None
}

/// Extract `(numerator, denominator, is_fraction)` from an expression.
///
/// Recognizes:
/// - `Div(num, den)`
/// - `Mul(Number(±1/n), x)` / `Mul(x, Number(±1/n))`
/// - `Mul(Div(±1, den), x)` / `Mul(x, Div(±1, den))`
pub fn extract_as_fraction(ctx: &mut Context, expr: ExprId) -> (ExprId, ExprId, bool) {
    use num_bigint::BigInt;
    use num_rational::BigRational;
    use num_traits::Signed;

    if let Some((num, den)) = as_div(ctx, expr) {
        return (num, den, true);
    }

    if let Some((l, r)) = as_mul(ctx, expr) {
        let check_unit_fraction = |n: &BigRational| -> Option<(BigInt, bool)> {
            if n.is_integer() {
                return None;
            }
            let numer = n.numer();
            let abs_numer: BigInt = if numer < &BigInt::from(0) {
                -numer.clone()
            } else {
                numer.clone()
            };
            if abs_numer == BigInt::from(1) {
                return Some((n.denom().clone(), numer.is_negative()));
            }
            None
        };

        let check_unit_div = |factor: ExprId| -> Option<(ExprId, bool)> {
            let (num, den) = as_div(ctx, factor)?;
            let n = crate::numeric::as_number(ctx, num)?;
            if n.is_integer() {
                let v = n.numer();
                if *v == BigInt::from(1) {
                    return Some((den, false));
                }
                if *v == BigInt::from(-1) {
                    return Some((den, true));
                }
            }
            None
        };

        if let Some(n) = crate::numeric::as_number(ctx, l) {
            if let Some((denom, is_neg)) = check_unit_fraction(n) {
                let den_expr = ctx.add(Expr::Number(BigRational::from_integer(denom)));
                if is_neg {
                    return (ctx.add(Expr::Neg(r)), den_expr, true);
                }
                return (r, den_expr, true);
            }
        }
        if let Some(n) = crate::numeric::as_number(ctx, r) {
            if let Some((denom, is_neg)) = check_unit_fraction(n) {
                let den_expr = ctx.add(Expr::Number(BigRational::from_integer(denom)));
                if is_neg {
                    return (ctx.add(Expr::Neg(l)), den_expr, true);
                }
                return (l, den_expr, true);
            }
        }

        if let Some((den, is_neg)) = check_unit_div(l) {
            if is_neg {
                return (ctx.add(Expr::Neg(r)), den, true);
            }
            return (r, den, true);
        }
        if let Some((den, is_neg)) = check_unit_div(r) {
            if is_neg {
                return (ctx.add(Expr::Neg(l)), den, true);
            }
            return (l, den, true);
        }
    }

    (expr, ctx.num(1), false)
}

/// Check if one denominator divides the other.
///
/// Returns `(new_n1, new_n2, common_den, is_divisible)`.
pub fn check_divisible_denominators(
    ctx: &mut Context,
    n1: ExprId,
    n2: ExprId,
    d1: ExprId,
    d2: ExprId,
) -> (ExprId, ExprId, ExprId, bool) {
    if let Some(k) = try_extract_factor(ctx, d2, d1) {
        let new_n1 = ctx.add_raw(Expr::Mul(n1, k));
        return (new_n1, n2, d2, true);
    }

    if let Some(k) = try_extract_factor(ctx, d1, d2) {
        let new_n2 = ctx.add_raw(Expr::Mul(n2, k));
        return (n1, new_n2, d1, true);
    }

    (n1, n2, d1, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly_compare::poly_eq;
    use cas_parser::parse;

    #[test]
    fn extract_as_fraction_detects_div() {
        let mut ctx = Context::new();
        let expr = parse("x/2", &mut ctx).expect("parse");
        let (num, den, ok) = extract_as_fraction(&mut ctx, expr);
        assert!(ok);
        let expected_num = parse("x", &mut ctx).expect("parse num");
        let expected_den = parse("2", &mut ctx).expect("parse den");
        assert!(poly_eq(&ctx, num, expected_num));
        assert!(poly_eq(&ctx, den, expected_den));
    }

    #[test]
    fn extract_as_fraction_detects_unit_fraction_mul() {
        let mut ctx = Context::new();
        let expr = parse("(1/3)*x", &mut ctx).expect("parse");
        let (num, den, ok) = extract_as_fraction(&mut ctx, expr);
        assert!(ok);
        let expected_num = parse("x", &mut ctx).expect("parse num");
        let expected_den = parse("3", &mut ctx).expect("parse den");
        assert!(poly_eq(&ctx, num, expected_num));
        assert!(poly_eq(&ctx, den, expected_den));
    }

    #[test]
    fn divisible_denominators_scales_left_numerator() {
        let mut ctx = Context::new();
        let n1 = parse("a", &mut ctx).expect("parse n1");
        let n2 = parse("b", &mut ctx).expect("parse n2");
        let d1 = parse("2", &mut ctx).expect("parse d1");
        let d2 = parse("2*x", &mut ctx).expect("parse d2");

        let (nn1, nn2, cd, ok) = check_divisible_denominators(&mut ctx, n1, n2, d1, d2);
        assert!(ok);
        let exp_n1 = parse("a*x", &mut ctx).expect("parse exp_n1");
        assert!(poly_eq(&ctx, nn1, exp_n1));
        assert!(poly_eq(&ctx, nn2, n2));
        assert!(poly_eq(&ctx, cd, d2));
    }
}
