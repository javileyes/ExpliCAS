use crate::expr_destructure::{as_add, as_mul};
use crate::expr_nary::{AddView, Sign};
use crate::pi_helpers::{is_pi, is_pi_over_n};
use cas_ast::{Context, Expr, ExprId};
use num_traits::One;

/// Extract the coefficient of π from an expression.
/// - π -> 1
/// - k*π -> k
/// - π*k -> k
pub fn extract_pi_coefficient(ctx: &Context, expr: ExprId) -> Option<i32> {
    if is_pi(ctx, expr) {
        return Some(1);
    }

    if let Expr::Mul(l, r) = ctx.get(expr) {
        if is_pi(ctx, *r) {
            if let Expr::Number(n) = ctx.get(*l) {
                if n.is_integer() {
                    return n.to_integer().try_into().ok();
                }
            }
        }
        if is_pi(ctx, *l) {
            if let Expr::Number(n) = ctx.get(*r) {
                if n.is_integer() {
                    return n.to_integer().try_into().ok();
                }
            }
        }
    }

    None
}

/// Extract `k` from expressions equivalent to `k*π/2` with integer `k`.
pub fn extract_pi_half_multiple(ctx: &Context, expr: ExprId) -> Option<i32> {
    if is_pi_over_n(ctx, expr, 2) {
        return Some(1);
    }

    if is_pi(ctx, expr) {
        return Some(2);
    }

    if let Expr::Mul(l, r) = ctx.get(expr) {
        if let Expr::Number(n) = ctx.get(*l) {
            if is_pi_over_n(ctx, *r, 2) && n.is_integer() {
                if let Ok(k) = n.to_integer().try_into() {
                    return Some(k);
                }
            }
            if is_pi(ctx, *r) && n.is_integer() {
                if let Ok(k_half) = n.to_integer().try_into() {
                    let k: i32 = k_half;
                    return Some(k * 2);
                }
            }
        }

        if let Expr::Number(n) = ctx.get(*r) {
            if is_pi_over_n(ctx, *l, 2) && n.is_integer() {
                if let Ok(k) = n.to_integer().try_into() {
                    return Some(k);
                }
            }
            if is_pi(ctx, *l) && n.is_integer() {
                if let Ok(k_half) = n.to_integer().try_into() {
                    let k: i32 = k_half;
                    return Some(k * 2);
                }
            }
        }
    }

    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Number(d) = ctx.get(*den) {
            if d.is_integer() && *d == num_rational::BigRational::from_integer(2.into()) {
                if is_pi(ctx, *num) {
                    return Some(1);
                }
                if let Expr::Mul(l, r) = ctx.get(*num) {
                    if let Expr::Number(n) = ctx.get(*l) {
                        if is_pi(ctx, *r) && n.is_integer() {
                            if let Ok(k) = n.to_integer().try_into() {
                                return Some(k);
                            }
                        }
                    }
                    if let Expr::Number(n) = ctx.get(*r) {
                        if is_pi(ctx, *l) && n.is_integer() {
                            if let Ok(k) = n.to_integer().try_into() {
                                return Some(k);
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

/// Extract `(base_term, k)` from `expr` such that:
/// `expr = base_term + k*π/2`.
///
/// Handles canonical and n-ary forms:
/// - `Div(Add(n*x, k*pi), m)` when `m | (2k)`
/// - `Mul(1/n, Add(..., k*pi))`
/// - Any n-ary additive expression containing a `k*π/2` term
pub fn extract_phase_shift(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, i32)> {
    // Form 1: Div((coeff*x + k*pi), denom) - canonical quotient form
    if let Expr::Div(num, den) = ctx.get(expr) {
        let num = *num;
        let den = *den;

        let denom_val: i32 = if let Expr::Number(n) = ctx.get(den) {
            if n.is_integer() {
                n.to_integer().try_into().ok()?
            } else {
                return None;
            }
        } else {
            return None;
        };

        if let Some((l, r)) = as_add(ctx, num) {
            if is_pi(ctx, r) {
                let k = 2 / denom_val;
                if 2 % denom_val == 0 {
                    let base = ctx.add(Expr::Div(l, den));
                    return Some((base, k));
                }
            }

            if is_pi(ctx, l) {
                let k = 2 / denom_val;
                if 2 % denom_val == 0 {
                    let base = ctx.add(Expr::Div(r, den));
                    return Some((base, k));
                }
            }

            if let Some(pi_coeff) = extract_pi_coefficient(ctx, r) {
                let k_times_2 = 2 * pi_coeff;
                if k_times_2 % denom_val == 0 {
                    let k = k_times_2 / denom_val;
                    let base = ctx.add(Expr::Div(l, den));
                    return Some((base, k));
                }
            }

            if let Some(pi_coeff) = extract_pi_coefficient(ctx, l) {
                let k_times_2 = 2 * pi_coeff;
                if k_times_2 % denom_val == 0 {
                    let k = k_times_2 / denom_val;
                    let base = ctx.add(Expr::Div(r, den));
                    return Some((base, k));
                }
            }
        }
    }

    // Form 1b: Mul(1/n, Add(coeff*x, k*pi)) - canonical lowered division form
    if let Some((coeff_id, inner)) = as_mul(ctx, expr) {
        if let Expr::Number(coeff) = ctx.get(coeff_id) {
            if coeff.numer() == &num_bigint::BigInt::from(1) && !coeff.denom().is_one() {
                let denom_val: i32 = coeff.denom().try_into().ok().unwrap_or(0);
                if denom_val > 0 {
                    if let Some((l, r)) = as_add(ctx, inner) {
                        if is_pi(ctx, r) {
                            let k = 2 / denom_val;
                            if 2 % denom_val == 0 {
                                let base = ctx.add(Expr::Mul(coeff_id, l));
                                return Some((base, k));
                            }
                        }

                        if is_pi(ctx, l) {
                            let k = 2 / denom_val;
                            if 2 % denom_val == 0 {
                                let base = ctx.add(Expr::Mul(coeff_id, r));
                                return Some((base, k));
                            }
                        }

                        if let Some(pi_coeff) = extract_pi_coefficient(ctx, r) {
                            let k_times_2 = 2 * pi_coeff;
                            if k_times_2 % denom_val == 0 {
                                let k = k_times_2 / denom_val;
                                let base = ctx.add(Expr::Mul(coeff_id, l));
                                return Some((base, k));
                            }
                        }
                    }
                }
            }
        }
    }

    // Form 2/3: n-ary additive scan
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() >= 2 {
        for (i, (term, sign)) in view.terms.iter().enumerate() {
            if let Some(mut k) = extract_pi_half_multiple(ctx, *term) {
                if *sign == Sign::Neg {
                    k = -k;
                }
                let remaining: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
                    .terms
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, t)| *t)
                    .collect();
                let rest_view = AddView {
                    root: expr,
                    terms: remaining,
                };
                let base = rest_view.rebuild(ctx);
                return Some((base, k));
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn extract_pi_coefficient_matches_mul_forms() {
        let mut ctx = Context::new();
        let pi = parse("pi", &mut ctx).expect("pi");
        let three_pi = parse("3*pi", &mut ctx).expect("3*pi");
        let pi_four = parse("pi*4", &mut ctx).expect("pi*4");
        let x = parse("x", &mut ctx).expect("x");

        assert_eq!(extract_pi_coefficient(&ctx, pi), Some(1));
        assert_eq!(extract_pi_coefficient(&ctx, three_pi), Some(3));
        assert_eq!(extract_pi_coefficient(&ctx, pi_four), Some(4));
        assert_eq!(extract_pi_coefficient(&ctx, x), None);
    }

    #[test]
    fn extract_pi_half_multiple_matches_common_forms() {
        let mut ctx = Context::new();
        let pi_half = parse("pi/2", &mut ctx).expect("pi/2");
        let pi = parse("pi", &mut ctx).expect("pi");
        let three_pi_half = parse("3*pi/2", &mut ctx).expect("3*pi/2");
        let five_pi = parse("5*pi", &mut ctx).expect("5*pi");

        assert_eq!(extract_pi_half_multiple(&ctx, pi_half), Some(1));
        assert_eq!(extract_pi_half_multiple(&ctx, pi), Some(2));
        assert_eq!(extract_pi_half_multiple(&ctx, three_pi_half), Some(3));
        assert_eq!(extract_pi_half_multiple(&ctx, five_pi), Some(10));
    }

    #[test]
    fn extract_phase_shift_from_div_add_form() {
        let mut ctx = Context::new();
        let expr = parse("(2*x + pi)/2", &mut ctx).expect("expr");
        let expected_base = parse("(2*x)/2", &mut ctx).expect("(2*x)/2");

        let (base, k) = extract_phase_shift(&mut ctx, expr).expect("phase shift");
        assert_eq!(base, expected_base);
        assert_eq!(k, 1);
    }

    #[test]
    fn extract_phase_shift_from_sub_form() {
        let mut ctx = Context::new();
        let expr = parse("x - pi/2", &mut ctx).expect("expr");
        let expected_base = parse("x", &mut ctx).expect("x");

        let (base, k) = extract_phase_shift(&mut ctx, expr).expect("phase shift");
        assert_eq!(base, expected_base);
        assert_eq!(k, -1);
    }

    #[test]
    fn extract_phase_shift_from_nary_add_form() {
        let mut ctx = Context::new();
        let expr = parse("x + y + pi/2", &mut ctx).expect("expr");
        let expected_base = parse("x + y", &mut ctx).expect("x+y");

        let (base, k) = extract_phase_shift(&mut ctx, expr).expect("phase shift");
        assert_eq!(base, expected_base);
        assert_eq!(k, 1);
    }
}
