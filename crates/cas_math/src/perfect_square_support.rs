use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{Signed, Zero};
use std::cmp::Ordering;

/// Try to compute the square root of a rational number.
/// Returns `Some(√r)` only when numerator and denominator are perfect squares.
pub fn rational_sqrt(r: &BigRational) -> Option<BigRational> {
    if r.is_negative() {
        return None;
    }

    if r.is_zero() {
        return Some(BigRational::from_integer(0.into()));
    }

    let numer = r.numer().clone();
    let denom = r.denom().clone();

    let numer_sqrt = numer.sqrt();
    if &numer_sqrt * &numer_sqrt != numer {
        return None;
    }

    let denom_sqrt = denom.sqrt();
    if &denom_sqrt * &denom_sqrt != denom {
        return None;
    }

    Some(BigRational::new(numer_sqrt, denom_sqrt))
}

/// Extract the square root of a term if it is a perfect square.
///
/// Recognizes:
/// - `Pow(base, 2k)` -> `base^k`
/// - `Mul(n, Pow(base, 2k))` with perfect-square integer `n` -> `sqrt(n) * base^k`
/// - `Number(n)` with perfect-square integer `n` -> `sqrt(n)`
pub fn extract_square_root_of_term(ctx: &mut Context, term: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, exp) = ctx.get(term) {
        let (base, exp) = (*base, *exp);
        if let Expr::Number(n) = ctx.get(exp) {
            let n = n.clone();
            if n.is_integer() {
                let int_val = n.to_integer();
                let two: num_bigint::BigInt = 2.into();
                if &int_val % &two == 0.into() && int_val > 0.into() {
                    let half_exp = &int_val / &two;
                    let half_exp_rat = BigRational::from_integer(half_exp);
                    if half_exp_rat == BigRational::from_integer(1.into()) {
                        return Some(base);
                    }
                    let half_exp_id = ctx.add(Expr::Number(half_exp_rat));
                    return Some(ctx.add(Expr::Pow(base, half_exp_id)));
                }
            }
        }
        return None;
    }

    if let Expr::Mul(l, r) = ctx.get(term) {
        let (l, r) = (*l, *r);
        for (maybe_coeff, maybe_pow) in [(l, r), (r, l)] {
            if let Expr::Number(coeff) = ctx.get(maybe_coeff) {
                if coeff.is_integer() && *coeff > BigRational::from_integer(0.into()) {
                    let coeff_int = coeff.to_integer();
                    let coeff_root = coeff_int.sqrt();
                    if &coeff_root * &coeff_root == coeff_int {
                        if let Some(pow_root) = extract_square_root_of_term(ctx, maybe_pow) {
                            let root_num =
                                ctx.add(Expr::Number(BigRational::from_integer(coeff_root)));
                            return Some(ctx.add(Expr::Mul(root_num, pow_root)));
                        }
                    }
                }
            }
        }
    }

    if let Expr::Number(n) = ctx.get(term) {
        if n.is_integer() && *n > BigRational::zero() {
            let int_val = n.to_integer();
            let root = int_val.sqrt();
            if &root * &root == int_val {
                return Some(ctx.add(Expr::Number(BigRational::from_integer(root))));
            }
        }
    }

    None
}

/// Try to match a 3-term additive expression as `(A ± B)^2`.
///
/// Returns `(A, B, is_sub)` where `is_sub=true` means `(A - B)^2`.
pub fn try_match_perfect_square_trinomial(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<(ExprId, ExprId, bool)> {
    use crate::expr_nary::{AddView, Sign};

    let terms = AddView::from_expr(ctx, arg).terms;
    if terms.len() != 3 {
        return None;
    }

    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                continue;
            }
            let k = 3 - i - j;

            let (term_a_sq, sign_a_sq) = terms[i];
            let (term_mid, sign_mid) = terms[k];
            let (term_b_sq, sign_b_sq) = terms[j];

            if sign_a_sq != Sign::Pos || sign_b_sq != Sign::Pos {
                continue;
            }

            let Some(a) = extract_square_root_of_term(ctx, term_a_sq) else {
                continue;
            };
            let Some(b) = extract_square_root_of_term(ctx, term_b_sq) else {
                continue;
            };

            let b_val = if let Expr::Number(bn) = ctx.get(b) {
                Some(bn.clone())
            } else {
                None
            };

            let mut effective_neg_mid = sign_mid == Sign::Neg;
            let mut effective_mid = term_mid;

            if let Expr::Mul(l, r) = ctx.get(effective_mid) {
                let (l, r) = (*l, *r);
                if let Expr::Number(n) = ctx.get(l) {
                    if n.is_negative() {
                        let abs_n = ctx.add(Expr::Number(-n.clone()));
                        effective_mid = ctx.add(Expr::Mul(abs_n, r));
                        effective_neg_mid = !effective_neg_mid;
                    }
                } else if let Expr::Number(n) = ctx.get(r) {
                    if n.is_negative() {
                        let abs_n = ctx.add(Expr::Number(-n.clone()));
                        effective_mid = ctx.add(Expr::Mul(l, abs_n));
                        effective_neg_mid = !effective_neg_mid;
                    }
                }
            } else if let Expr::Number(n) = ctx.get(effective_mid) {
                if n.is_negative() {
                    effective_mid = ctx.add(Expr::Number(-n.clone()));
                    effective_neg_mid = !effective_neg_mid;
                }
            } else if let Expr::Neg(inner) = ctx.get(effective_mid) {
                effective_mid = *inner;
                effective_neg_mid = !effective_neg_mid;
            }

            if !check_middle_term_2ab(ctx, effective_mid, a, b, &b_val) {
                continue;
            }

            return Some((a, b, effective_neg_mid));
        }
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqrtPerfectSquareRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Rewrite sqrt of perfect-square trinomials:
/// - `sqrt(A^2 ± 2AB + B^2) -> abs(A ± B)`
///
/// Assumes canonicalized sqrt representation as `Pow(arg, 1/2)`.
pub fn try_rewrite_sqrt_perfect_square_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<SqrtPerfectSquareRewrite> {
    let arg = match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            let half = BigRational::new(1.into(), 2.into());
            match ctx.get(*exp) {
                Expr::Number(n) if *n == half => *base,
                _ => return None,
            }
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            args[0]
        }
        _ => return None,
    };

    let (a, b, is_sub) = try_match_perfect_square_trinomial(ctx, arg)?;
    let inner = if is_sub {
        ctx.add(Expr::Sub(a, b))
    } else {
        ctx.add(Expr::Add(a, b))
    };
    let rewritten = ctx.call_builtin(cas_ast::BuiltinFn::Abs, vec![inner]);
    Some(SqrtPerfectSquareRewrite {
        rewritten,
        desc: "sqrt(A^2 ± 2AB + B^2) = |A ± B|",
    })
}

fn check_middle_term_2ab(
    ctx: &mut Context,
    term: ExprId,
    a: ExprId,
    b: ExprId,
    b_val: &Option<BigRational>,
) -> bool {
    let two = BigRational::from_integer(2.into());

    let factors = crate::expr_nary::mul_factors(ctx, term);

    if let Some(bv) = b_val {
        let expected_coeff = &two * bv;
        if factors.len() == 2 {
            for perm in [(0, 1), (1, 0)] {
                if let Expr::Number(n) = ctx.get(factors[perm.0]) {
                    if *n == expected_coeff
                        && compare_expr(ctx, factors[perm.1], a) == Ordering::Equal
                    {
                        return true;
                    }
                }
            }
        }
        if factors.len() == 3 {
            let mut found_two = false;
            let mut found_bv = false;
            let mut found_a = false;
            for &f in &factors {
                if !found_two {
                    if let Expr::Number(n) = ctx.get(f) {
                        if *n == two {
                            found_two = true;
                            continue;
                        }
                    }
                }
                if !found_bv {
                    if let Expr::Number(n) = ctx.get(f) {
                        if n == bv {
                            found_bv = true;
                            continue;
                        }
                    }
                }
                if !found_a && compare_expr(ctx, f, a) == Ordering::Equal {
                    found_a = true;
                    continue;
                }
            }
            if found_two && found_bv && found_a {
                return true;
            }
        }
    }

    if factors.len() == 3 {
        let mut found_two = false;
        let mut found_a = false;
        let mut found_b = false;
        for &f in &factors {
            if !found_two {
                if let Expr::Number(n) = ctx.get(f) {
                    if *n == two {
                        found_two = true;
                        continue;
                    }
                }
            }
            if !found_a && compare_expr(ctx, f, a) == Ordering::Equal {
                found_a = true;
                continue;
            }
            if !found_b && compare_expr(ctx, f, b) == Ordering::Equal {
                found_b = true;
                continue;
            }
        }
        if found_two && found_a && found_b {
            return true;
        }
    }

    let two_id = ctx.add(Expr::Number(two.clone()));
    let ab = ctx.add(Expr::Mul(a, b));
    let expected_1 = ctx.add(Expr::Mul(two_id, ab));
    if compare_expr(ctx, term, expected_1) == Ordering::Equal {
        return true;
    }

    let two_a = ctx.add(Expr::Mul(two_id, a));
    let expected_2 = ctx.add(Expr::Mul(two_a, b));
    if compare_expr(ctx, term, expected_2) == Ordering::Equal {
        return true;
    }

    let two_b = ctx.add(Expr::Mul(two_id, b));
    let expected_3 = ctx.add(Expr::Mul(a, two_b));
    if compare_expr(ctx, term, expected_3) == Ordering::Equal {
        return true;
    }

    if let Some(bv) = b_val {
        let expected_coeff_val = two * bv;
        let coeff_id = ctx.add(Expr::Number(expected_coeff_val));
        let expected_4 = ctx.add(Expr::Mul(coeff_id, a));
        if compare_expr(ctx, term, expected_4) == Ordering::Equal {
            return true;
        }
        let expected_5 = ctx.add(Expr::Mul(a, coeff_id));
        if compare_expr(ctx, term, expected_5) == Ordering::Equal {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_sqrt_returns_exact_root() {
        let x = BigRational::new(9.into(), 16.into());
        assert_eq!(
            rational_sqrt(&x),
            Some(BigRational::new(3.into(), 4.into()))
        );
    }

    #[test]
    fn extract_square_root_of_term_handles_coeff_times_even_power() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("4*x^2", &mut ctx).expect("parse");
        let got = extract_square_root_of_term(&mut ctx, expr).expect("root");
        let expected = cas_parser::parse("2*x", &mut ctx).expect("parse expected");
        assert_eq!(compare_expr(&ctx, got, expected), Ordering::Equal);
    }

    #[test]
    fn match_perfect_square_trinomial_symbolic() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("x^2 + 2*x*y + y^2", &mut ctx).expect("parse");
        let (a, b, is_sub) = try_match_perfect_square_trinomial(&mut ctx, expr).expect("match");
        assert!(!is_sub);
        assert_ne!(a, b);
    }

    #[test]
    fn match_perfect_square_trinomial_with_coefficients() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("x^2 + 2*x + 1", &mut ctx).expect("parse");
        let (a, b, is_sub) = try_match_perfect_square_trinomial(&mut ctx, expr).expect("match");
        assert!(!is_sub);
        assert_ne!(a, b);
    }

    #[test]
    fn rewrite_sqrt_perfect_square_expr_symbolic() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("sqrt(x^2 + 2*x*y + y^2)", &mut ctx).expect("parse");
        let expected = cas_parser::parse("abs(x + y)", &mut ctx).expect("expected");
        let rw = try_rewrite_sqrt_perfect_square_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(compare_expr(&ctx, rw.rewritten, expected), Ordering::Equal);
    }

    #[test]
    fn rewrite_sqrt_perfect_square_expr_rejects_non_sqrt() {
        let mut ctx = Context::new();
        let expr = cas_parser::parse("x^2 + 2*x + 1", &mut ctx).expect("parse");
        assert!(try_rewrite_sqrt_perfect_square_expr(&mut ctx, expr).is_none());
    }
}
