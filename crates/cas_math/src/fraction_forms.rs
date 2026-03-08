//! Fraction-shape helpers over AST expressions.

use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_traits::One;
use std::cmp::Ordering;

/// Recognizes ±1 in various AST forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignOne {
    PlusOne,
    MinusOne,
}

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

/// Check if expression is `+1` or `-1` (including `Neg(1)` shape).
pub fn sign_one(ctx: &Context, id: ExprId) -> Option<SignOne> {
    use num_rational::BigRational;
    match ctx.get(id) {
        Expr::Number(n) => {
            if n == &BigRational::from_integer((-1).into()) {
                Some(SignOne::MinusOne)
            } else if n.is_one() {
                Some(SignOne::PlusOne)
            } else {
                None
            }
        }
        Expr::Neg(inner) => match ctx.get(*inner) {
            Expr::Number(n) if n.is_one() => Some(SignOne::MinusOne),
            _ => None,
        },
        _ => None,
    }
}

/// Normalize binomial denominator into `(left, right_norm, is_add, right_is_abs_one)`.
///
/// Examples:
/// - `a + 1` => `(a, 1, true,  true)`
/// - `a - 1` => `(a, 1, false, true)`
/// - `a + (-1)` => `(a, 1, false, true)` (normalized to subtraction)
pub fn split_binomial_den(ctx: &mut Context, den: ExprId) -> Option<(ExprId, ExprId, bool, bool)> {
    let one = ctx.num(1);
    match ctx.get(den) {
        Expr::Add(l, r) => match sign_one(ctx, *r) {
            Some(SignOne::PlusOne) => Some((*l, one, true, true)),
            Some(SignOne::MinusOne) => Some((*l, one, false, true)),
            None => Some((*l, *r, true, false)),
        },
        Expr::Sub(l, r) => match sign_one(ctx, *r) {
            Some(SignOne::PlusOne) => Some((*l, one, false, true)),
            Some(SignOne::MinusOne) => Some((*l, one, true, true)),
            None => Some((*l, *r, false, false)),
        },
        _ => None,
    }
}

/// Check if two denominator expressions are structural opposites.
///
/// Examples:
/// - `(a - b)` vs `(b - a)`
/// - `(-a + b)` vs `(a - b)`
/// - `(a-b)(a-c)` vs `(a-c)(b-a)`
pub fn are_denominators_opposite(ctx: &Context, e1: ExprId, e2: ExprId) -> bool {
    match (ctx.get(e1), ctx.get(e2)) {
        // Case 1: (a - b) vs (b - a)
        (Expr::Sub(l1, r1), Expr::Sub(l2, r2)) => {
            compare_expr(ctx, *l1, *r2) == Ordering::Equal
                && compare_expr(ctx, *r1, *l2) == Ordering::Equal
        }
        // Case 2: (-a + b) vs (a - b), including Number(-n) normalization
        (Expr::Add(l1, r1), Expr::Sub(l2, r2)) => {
            if let Expr::Neg(neg_inner) = ctx.get(*l1) {
                if compare_expr(ctx, *neg_inner, *l2) == Ordering::Equal
                    && compare_expr(ctx, *r1, *r2) == Ordering::Equal
                {
                    return true;
                }
            }
            if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(*l1), ctx.get(*l2)) {
                let neg_n2 = -n2.clone();
                if n1 == &neg_n2 && compare_expr(ctx, *r1, *r2) == Ordering::Equal {
                    return true;
                }
            }
            false
        }
        // Case 3: reverse of case 2
        (Expr::Sub(_, _), Expr::Add(_, _)) => are_denominators_opposite(ctx, e2, e1),
        // Case 4: both Add, multiple opposite layouts
        (Expr::Add(l1, r1), Expr::Add(l2, r2)) => {
            if let (Expr::Neg(neg_l1), Expr::Neg(neg_l2)) = (ctx.get(*l1), ctx.get(*l2)) {
                if compare_expr(ctx, *neg_l1, *r2) == Ordering::Equal
                    && compare_expr(ctx, *r1, *neg_l2) == Ordering::Equal
                {
                    return true;
                }
            }

            if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(*l1), ctx.get(*l2)) {
                if let Expr::Neg(neg_r2) = ctx.get(*r2) {
                    let neg_n2 = -n2.clone();
                    if n1 == &neg_n2 && compare_expr(ctx, *r1, *neg_r2) == Ordering::Equal {
                        return true;
                    }
                }
                if let Expr::Neg(neg_r1) = ctx.get(*r1) {
                    let neg_n2 = -n2.clone();
                    if n1 == &neg_n2 && compare_expr(ctx, *neg_r1, *r2) == Ordering::Equal {
                        return true;
                    }
                }
            }

            if let (Expr::Neg(neg_r1), Expr::Neg(neg_r2)) = (ctx.get(*r1), ctx.get(*r2)) {
                if compare_expr(ctx, *l1, *neg_r2) == Ordering::Equal
                    && compare_expr(ctx, *l2, *neg_r1) == Ordering::Equal
                {
                    return true;
                }
            }
            false
        }
        // Case 5: multiplicative expressions where one factor pair is opposite
        (Expr::Mul(m1_l, m1_r), Expr::Mul(m2_l, m2_r)) => {
            let factors_opposite =
                |f1: ExprId, f2: ExprId| -> bool { are_denominators_opposite(ctx, f1, f2) };

            if compare_expr(ctx, *m1_l, *m2_l) == Ordering::Equal && factors_opposite(*m1_r, *m2_r)
            {
                return true;
            }
            if compare_expr(ctx, *m1_l, *m2_r) == Ordering::Equal && factors_opposite(*m1_r, *m2_l)
            {
                return true;
            }
            if compare_expr(ctx, *m1_r, *m2_l) == Ordering::Equal && factors_opposite(*m1_l, *m2_r)
            {
                return true;
            }
            if compare_expr(ctx, *m1_r, *m2_r) == Ordering::Equal && factors_opposite(*m1_l, *m2_l)
            {
                return true;
            }
            false
        }
        _ => false,
    }
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

    #[test]
    fn sign_one_detects_plus_and_minus_one() {
        let mut ctx = Context::new();
        let plus = parse("1", &mut ctx).expect("parse +1");
        let minus = parse("-1", &mut ctx).expect("parse -1");
        assert_eq!(sign_one(&ctx, plus), Some(SignOne::PlusOne));
        assert_eq!(sign_one(&ctx, minus), Some(SignOne::MinusOne));
    }

    #[test]
    fn split_binomial_den_normalizes_signs() {
        let mut ctx = Context::new();
        let den_add = parse("sqrt(x)+(-1)", &mut ctx).expect("parse add");
        let den_sub = parse("sqrt(x)-(-1)", &mut ctx).expect("parse sub");

        let (_, r1, add1, abs_one1) = split_binomial_den(&mut ctx, den_add).expect("split add");
        let (_, r2, add2, abs_one2) = split_binomial_den(&mut ctx, den_sub).expect("split sub");

        assert!(matches!(ctx.get(r1), Expr::Number(n) if n.is_one()));
        assert!(!add1);
        assert!(abs_one1);
        assert!(matches!(ctx.get(r2), Expr::Number(n) if n.is_one()));
        assert!(add2);
        assert!(abs_one2);
    }

    #[test]
    fn opposite_denominator_detection() {
        let mut ctx = Context::new();
        let d1 = parse("a-b", &mut ctx).expect("parse d1");
        let d2 = parse("b-a", &mut ctx).expect("parse d2");
        assert!(are_denominators_opposite(&ctx, d1, d2));
    }
}
