//! Support for quotient-of-powers rewrites.

use cas_ast::ordering::compare_expr;
use cas_ast::views::FractionParts;
use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Zero};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub struct QuotientOfPowersRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Try to rewrite fractional-exponent quotient-of-powers patterns.
///
/// `allow_zero_cancel(ctx, base)` is consulted for the `a^n / a^n -> 1` case.
pub fn try_rewrite_quotient_of_powers_expr_with<FAllow>(
    ctx: &mut Context,
    expr: ExprId,
    mut allow_zero_cancel: FAllow,
) -> Option<QuotientOfPowersRewrite>
where
    FAllow: FnMut(&Context, ExprId) -> bool,
{
    let fp = FractionParts::from(&*ctx, expr);
    if !fp.is_fraction() {
        return None;
    }

    let (num, den, _) = fp.to_num_den(ctx);
    let num_pow = match ctx.get(num) {
        Expr::Pow(b, e) => Some((*b, *e)),
        _ => None,
    };
    let den_pow = match ctx.get(den) {
        Expr::Pow(b, e) => Some((*b, *e)),
        _ => None,
    };

    if let (Some((b_n, e_n)), Some((b_d, e_d))) = (num_pow, den_pow) {
        if compare_expr(ctx, b_n, b_d) == Ordering::Equal {
            if let (Expr::Number(n), Expr::Number(m)) = (ctx.get(e_n), ctx.get(e_d)) {
                if n.is_integer() && m.is_integer() {
                    return None;
                }
                let diff = n - m;
                if diff.is_zero() {
                    if !allow_zero_cancel(ctx, b_n) {
                        return None;
                    }
                    return Some(QuotientOfPowersRewrite {
                        rewritten: ctx.num(1),
                        desc: "a^n / a^n = 1",
                    });
                } else if diff.is_one() {
                    return Some(QuotientOfPowersRewrite {
                        rewritten: b_n,
                        desc: "a^n / a^m = a^(n-m)",
                    });
                } else {
                    if diff < num_rational::BigRational::zero() && !diff.is_integer() {
                        return None;
                    }
                    let new_exp = ctx.add(Expr::Number(diff));
                    return Some(QuotientOfPowersRewrite {
                        rewritten: ctx.add(Expr::Pow(b_n, new_exp)),
                        desc: "a^n / a^m = a^(n-m)",
                    });
                }
            }
        }
    }

    if let Some((b_n, e_n)) = num_pow {
        if compare_expr(ctx, b_n, den) == Ordering::Equal {
            if let Expr::Number(n) = ctx.get(e_n) {
                if !n.is_integer() {
                    let new_exp_val = n - num_rational::BigRational::one();
                    if new_exp_val < num_rational::BigRational::zero() {
                        return None;
                    }
                    if new_exp_val.is_one() {
                        return Some(QuotientOfPowersRewrite {
                            rewritten: b_n,
                            desc: "a^n / a = a^(n-1)",
                        });
                    } else {
                        let new_exp = ctx.add(Expr::Number(new_exp_val));
                        return Some(QuotientOfPowersRewrite {
                            rewritten: ctx.add(Expr::Pow(b_n, new_exp)),
                            desc: "a^n / a = a^(n-1)",
                        });
                    }
                }
            }
        }
    }

    if let Some((b_d, e_d)) = den_pow {
        if compare_expr(ctx, num, b_d) == Ordering::Equal {
            if let Expr::Number(m) = ctx.get(e_d) {
                if !m.is_integer() {
                    let new_exp_val = num_rational::BigRational::one() - m;
                    if new_exp_val < num_rational::BigRational::zero() {
                        return None;
                    }
                    let new_exp = ctx.add(Expr::Number(new_exp_val));
                    return Some(QuotientOfPowersRewrite {
                        rewritten: ctx.add(Expr::Pow(num, new_exp)),
                        desc: "a / a^m = a^(1-m)",
                    });
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_quotient_of_powers_expr_with;
    use cas_ast::ordering::compare_expr;
    use cas_ast::Context;
    use cas_parser::parse;
    use std::cmp::Ordering;

    #[test]
    fn rewrites_fractional_pow_ratio() {
        let mut ctx = Context::new();
        let x = parse("x", &mut ctx).expect("parse x");
        let three_halves = ctx.rational(3, 2);
        let one_half = ctx.rational(1, 2);
        let num = ctx.add(cas_ast::Expr::Pow(x, three_halves));
        let den = ctx.add(cas_ast::Expr::Pow(x, one_half));
        let expr = ctx.add(cas_ast::Expr::Div(num, den));
        let rewrite = try_rewrite_quotient_of_powers_expr_with(&mut ctx, expr, |_ctx, _base| true)
            .expect("rewrite");
        let expected = parse("x", &mut ctx).expect("expected");
        assert_eq!(
            compare_expr(&ctx, rewrite.rewritten, expected),
            Ordering::Equal
        );
    }

    #[test]
    fn rejects_zero_cancel_when_callback_blocks() {
        let mut ctx = Context::new();
        let expr = parse("x^(1/2) / x^(1/2)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_quotient_of_powers_expr_with(&mut ctx, expr, |_ctx, _base| false);
        assert!(rewrite.is_none());
    }
}
