//! Support for power evaluation and normalization rewrites.

use crate::expr_destructure::{as_neg, as_pow};
use crate::root_forms::extract_root_factor;
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr, ExprId};
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive};
use std::cmp::Ordering;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerEvalRewrite {
    pub rewritten: ExprId,
    pub desc: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowerEvalStaticRewrite {
    pub rewritten: ExprId,
    pub kind: PowerEvalStaticRewriteKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerEvalStaticRewriteKind {
    NegativeExponentNormalization,
    NegativeBaseEven,
    NegativeBaseOdd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvenPowSubSwapRewrite {
    pub rewritten: ExprId,
    pub old_base: ExprId,
    pub new_base: ExprId,
}

/// Try `x^(-n) -> 1/x^n` for integer negative exponents.
pub fn try_rewrite_negative_exponent_normalization_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerEvalStaticRewrite> {
    let (base, exp) = as_pow(ctx, expr)?;
    if let Expr::Number(n) = ctx.get(exp) {
        if n.is_integer() && n.is_negative() {
            let pos_n = -n.clone();
            let pos_exp = ctx.add(Expr::Number(pos_n));
            let one = ctx.num(1);
            let pos_pow = ctx.add(Expr::Pow(base, pos_exp));
            let rewritten = ctx.add(Expr::Div(one, pos_pow));
            return Some(PowerEvalStaticRewrite {
                rewritten,
                kind: PowerEvalStaticRewriteKind::NegativeExponentNormalization,
            });
        }
    }
    None
}

/// Try `(-x)^n` rewrites for integer exponents:
/// - `(-x)^even -> x^even`
/// - `(-x)^odd -> -(x^odd)`
pub fn try_rewrite_negative_base_power_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerEvalStaticRewrite> {
    let (base, exp) = as_pow(ctx, expr)?;
    let inner = as_neg(ctx, base)?;
    if let Expr::Number(n) = ctx.get(exp) {
        if n.is_integer() {
            if n.to_integer().is_even() {
                let rewritten = ctx.add(Expr::Pow(inner, exp));
                return Some(PowerEvalStaticRewrite {
                    rewritten,
                    kind: PowerEvalStaticRewriteKind::NegativeBaseEven,
                });
            }
            let pow = ctx.add(Expr::Pow(inner, exp));
            let rewritten = ctx.add(Expr::Neg(pow));
            return Some(PowerEvalStaticRewrite {
                rewritten,
                kind: PowerEvalStaticRewriteKind::NegativeBaseOdd,
            });
        }
    }
    None
}

/// Canonicalize even-power subtraction bases:
/// `(b-a)^even -> (a-b)^even` when `a < b`.
pub fn try_rewrite_even_pow_sub_swap_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<EvenPowSubSwapRewrite> {
    let (base, exp) = as_pow(ctx, expr)?;

    let is_even = match ctx.get(exp) {
        Expr::Number(n) => n.is_integer() && n.to_integer().is_even(),
        _ => false,
    };
    if !is_even {
        return None;
    }

    let (pos_term, neg_term) = match ctx.get(base) {
        Expr::Sub(a, b) => (*a, *b),
        Expr::Add(l, r) => {
            if let Expr::Neg(inner) = ctx.get(*r) {
                (*l, *inner)
            } else if let Expr::Neg(inner) = ctx.get(*l) {
                (*r, *inner)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    if compare_expr(ctx, neg_term, pos_term) != Ordering::Less {
        return None;
    }

    let new_base = ctx.add(Expr::Sub(neg_term, pos_term));
    let rewritten = ctx.add(Expr::Pow(new_base, exp));

    Some(EvenPowSubSwapRewrite {
        rewritten,
        old_base: base,
        new_base,
    })
}

/// Try numeric power evaluation:
/// - direct literal evaluation through `const_eval`
/// - root-factor extraction simplification for rational exponents
pub fn try_rewrite_evaluate_power_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerEvalRewrite> {
    let (base, exp) = as_pow(ctx, expr)?;

    if let Some(result) = crate::const_eval::try_eval_pow_literal(ctx, base, exp) {
        return Some(PowerEvalRewrite {
            rewritten: result,
            desc: "Evaluate literal power".to_string(),
        });
    }

    if let (Expr::Number(b), Expr::Number(e)) = (ctx.get(base), ctx.get(exp)) {
        let (b, e) = (b.clone(), e.clone());
        let numer = e.numer();
        let denom = e.denom();

        if let Some(n) = denom.to_u32() {
            let b_num = b.numer();
            let b_den = b.denom();

            let (out_n, in_n) = extract_root_factor(b_num, n);
            let (out_d, in_d) = extract_root_factor(b_den, n);

            if !out_n.is_one() || !out_d.is_one() {
                if let Some(pow_num) = numer.to_i32() {
                    use num_rational::BigRational;
                    let outside_rat = BigRational::new(out_n.clone(), out_d.clone());
                    let outside_val = if pow_num == 1 {
                        outside_rat
                    } else {
                        num_traits::Pow::pow(&outside_rat, pow_num.unsigned_abs())
                    };
                    let outside_val = if pow_num < 0 {
                        outside_val.recip()
                    } else {
                        outside_val
                    };

                    let inside_rat = BigRational::new(in_n, in_d);
                    if inside_rat.is_one() {
                        let rewritten = ctx.add(Expr::Number(outside_val));
                        return Some(PowerEvalRewrite {
                            rewritten,
                            desc: format!("Simplify root: {}^{}", b, e),
                        });
                    } else {
                        let outside_expr = ctx.add(Expr::Number(outside_val));
                        let inside_expr = ctx.add(Expr::Number(inside_rat));
                        let exp_expr = ctx.add(Expr::Number(e.clone()));
                        let root_part = ctx.add(Expr::Pow(inside_expr, exp_expr));
                        let rewritten = ctx.add(Expr::Mul(outside_expr, root_part));
                        return Some(PowerEvalRewrite {
                            rewritten,
                            desc: format!("Simplify root: {}^{}", b, e),
                        });
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::{
        try_rewrite_evaluate_power_expr, try_rewrite_even_pow_sub_swap_expr,
        try_rewrite_negative_base_power_expr, try_rewrite_negative_exponent_normalization_expr,
    };
    use cas_ast::{Context, Expr};
    use cas_parser::parse;

    #[test]
    fn rewrites_negative_integer_exponent() {
        let mut ctx = Context::new();
        let expr = parse("x^(-3)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_negative_exponent_normalization_expr(&mut ctx, expr);
        assert!(rewrite.is_some());
    }

    #[test]
    fn evaluates_literal_power() {
        let mut ctx = Context::new();
        let expr = parse("2^3", &mut ctx).expect("parse");
        let rewrite = try_rewrite_evaluate_power_expr(&mut ctx, expr).expect("rewrite");
        assert!(matches!(ctx.get(rewrite.rewritten), Expr::Number(_)));
    }

    #[test]
    fn rewrites_negative_base_power() {
        let mut ctx = Context::new();
        let expr = parse("(-x)^3", &mut ctx).expect("parse");
        assert!(try_rewrite_negative_base_power_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn rewrites_even_pow_sub_swap() {
        let mut ctx = Context::new();
        let expr = parse("(b-a)^2", &mut ctx).expect("parse");
        assert!(try_rewrite_even_pow_sub_swap_expr(&mut ctx, expr).is_some());
    }
}
