//! Support for multiplication/division fraction rewrites.

use crate::expr_destructure::as_mul;
use cas_ast::ordering::compare_expr;
use cas_ast::views::{Factor, FractionParts};
use cas_ast::{Context, Expr, ExprId};

#[derive(Debug, Clone, Copy)]
pub struct MulDivRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Simplify multiplication expressions that include fraction-like factors.
///
/// Main behaviors:
/// - `(a/b) * b -> a`
/// - `a * (b/a) -> b`
/// - `(p/q) * (a/b) -> (p*a)/(q*b)` for rational `p/q`
/// - general fraction combination `(n1/d1) * (n2/d2) -> (n1*n2)/(d1*d2)`
pub fn try_rewrite_simplify_mul_div_expr(ctx: &mut Context, expr: ExprId) -> Option<MulDivRewrite> {
    let (l, r) = as_mul(ctx, expr)?;

    let fp_l = FractionParts::from(&*ctx, l);
    let fp_r = FractionParts::from(&*ctx, r);
    if !fp_l.is_fraction() && !fp_r.is_fraction() {
        return None;
    }

    if fp_l.is_fraction() && fp_l.den.len() == 1 && fp_l.den[0].exp == 1 {
        let den_base = fp_l.den[0].base;
        if compare_expr(ctx, den_base, r) == std::cmp::Ordering::Equal {
            let rewritten = if fp_l.num.is_empty() {
                ctx.num(fp_l.sign as i64)
            } else {
                let num_prod = FractionParts::build_product_static(ctx, &fp_l.num);
                if fp_l.sign < 0 {
                    ctx.add(Expr::Neg(num_prod))
                } else {
                    num_prod
                }
            };
            return Some(MulDivRewrite {
                rewritten,
                desc: "Cancel division: (a/b)*b -> a",
            });
        }
    }

    if fp_r.is_fraction() && fp_r.den.len() == 1 && fp_r.den[0].exp == 1 {
        let den_base = fp_r.den[0].base;
        if compare_expr(ctx, den_base, l) == std::cmp::Ordering::Equal {
            let rewritten = if fp_r.num.is_empty() {
                ctx.num(fp_r.sign as i64)
            } else {
                let num_prod = FractionParts::build_product_static(ctx, &fp_r.num);
                if fp_r.sign < 0 {
                    ctx.add(Expr::Neg(num_prod))
                } else {
                    num_prod
                }
            };
            return Some(MulDivRewrite {
                rewritten,
                desc: "Cancel division: a*(b/a) -> b",
            });
        }
    }

    {
        let check_rational_with_fraction =
            |num_side: ExprId,
             frac_side: ExprId|
             -> Option<(num_rational::BigRational, FractionParts)> {
                if let Expr::Number(n) = ctx.get(num_side) {
                    if !n.is_integer() {
                        let fp = FractionParts::from(&*ctx, frac_side);
                        if fp.is_fraction() {
                            return Some((n.clone(), fp));
                        }
                    }
                }
                None
            };

        if let Some((rational, fp_frac)) =
            check_rational_with_fraction(l, r).or_else(|| check_rational_with_fraction(r, l))
        {
            let (p, q) = (rational.numer().clone(), rational.denom().clone());
            let p_expr = ctx.add(Expr::Number(num_rational::BigRational::from_integer(p)));
            let q_expr = ctx.add(Expr::Number(num_rational::BigRational::from_integer(q)));

            let mut new_num = vec![Factor {
                base: p_expr,
                exp: 1,
            }];
            new_num.extend(fp_frac.num.iter().cloned());

            let mut new_den = vec![Factor {
                base: q_expr,
                exp: 1,
            }];
            new_den.extend(fp_frac.den.iter().cloned());

            let result_fp = FractionParts {
                sign: fp_frac.sign,
                num: new_num,
                den: new_den,
            };

            let new_expr = result_fp.build_as_div(ctx);
            if new_expr != expr {
                return Some(MulDivRewrite {
                    rewritten: new_expr,
                    desc: "Combine fractions in multiplication",
                });
            }
        }
    }

    if matches!(ctx.get(l), Expr::Number(_) | Expr::Constant(_))
        || matches!(ctx.get(r), Expr::Number(_) | Expr::Constant(_))
    {
        return None;
    }

    if fp_l.is_fraction() || fp_r.is_fraction() {
        let mut combined_num = Vec::new();
        combined_num.extend(fp_l.num.iter().cloned());
        combined_num.extend(fp_r.num.iter().cloned());

        let mut combined_den = Vec::new();
        combined_den.extend(fp_l.den.iter().cloned());
        combined_den.extend(fp_r.den.iter().cloned());

        {
            let mut new_num = Vec::with_capacity(combined_num.len());
            for f in &combined_num {
                if f.exp == 1 {
                    if let Expr::Number(n) = ctx.get(f.base) {
                        let n = n.clone();
                        if !n.is_integer() {
                            let (p, q) = (n.numer().clone(), n.denom().clone());
                            let p_expr =
                                ctx.add(Expr::Number(num_rational::BigRational::from_integer(p)));
                            let q_expr =
                                ctx.add(Expr::Number(num_rational::BigRational::from_integer(q)));
                            new_num.push(Factor {
                                base: p_expr,
                                exp: 1,
                            });
                            combined_den.push(Factor {
                                base: q_expr,
                                exp: 1,
                            });
                            continue;
                        }
                    }
                }
                new_num.push(*f);
            }
            combined_num = new_num;
        }

        let combined_sign = (fp_l.sign as i16 * fp_r.sign as i16) as i8;
        let result_fp = FractionParts {
            sign: combined_sign,
            num: combined_num,
            den: combined_den,
        };

        let new_expr = result_fp.build_as_div(ctx);
        if new_expr == expr {
            return None;
        }

        return Some(MulDivRewrite {
            rewritten: new_expr,
            desc: "Combine fractions in multiplication",
        });
    }

    None
}

#[cfg(test)]
mod tests {
    use super::try_rewrite_simplify_mul_div_expr;
    use crate::poly_compare::poly_eq;
    use cas_ast::Context;
    use cas_ast::{collect_variables, Expr};
    use cas_parser::parse;

    #[test]
    fn cancel_left_fraction_times_denominator() {
        let mut ctx = Context::new();
        let expr = parse("(a/b)*b", &mut ctx).expect("parse");
        let rw = try_rewrite_simplify_mul_div_expr(&mut ctx, expr).expect("rewrite");
        let expected = parse("a", &mut ctx).expect("expected");
        assert!(poly_eq(&ctx, rw.rewritten, expected));
    }

    #[test]
    fn combine_two_fraction_factors() {
        let mut ctx = Context::new();
        let expr = parse("(a/b)*(c/d)", &mut ctx).expect("parse");
        let rw = try_rewrite_simplify_mul_div_expr(&mut ctx, expr).expect("rewrite");
        assert!(matches!(ctx.get(rw.rewritten), Expr::Div(_, _)));
        let vars = collect_variables(&ctx, rw.rewritten);
        assert_eq!(vars.len(), 4);
    }

    #[test]
    fn keep_constant_times_fraction_unchanged() {
        let mut ctx = Context::new();
        let expr = parse("2*(a/b)", &mut ctx).expect("parse");
        assert!(try_rewrite_simplify_mul_div_expr(&mut ctx, expr).is_none());
    }
}
