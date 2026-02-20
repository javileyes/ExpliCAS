use crate::pi_helpers::{is_pi, is_pi_over_n};
use cas_ast::{Context, Expr, ExprId};

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
}
