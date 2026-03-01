//! Support for product/quotient rewrites over powers.

use crate::build::mul2_raw;
use crate::canonical_forms::is_canonical_form;
use crate::exponents_support::{add_exp, has_numeric_factor};
use crate::expr_destructure::{as_div, as_mul, as_pow};
use crate::expr_nary::mul_leaves;
use crate::expr_predicates::is_e_constant_expr;
use crate::root_forms::can_distribute_root_safely;
use cas_ast::ordering::compare_expr;
use cas_ast::views::FractionParts;
use cas_ast::{Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::Signed;
use num_traits::{One, Zero};
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PowerProductRewrite {
    pub rewritten: ExprId,
    pub desc: &'static str,
}

/// Try product-of-powers rewrites:
/// - `x^a * x^b -> x^(a+b)`
/// - `x^a * x -> x^(a+1)` and symmetric variants
/// - selected nested variants preserving multiplicative tails
pub fn try_rewrite_product_power_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerProductRewrite> {
    let should_combine = |ctx: &Context, base: ExprId, e1: ExprId, e2: ExprId| -> bool {
        if let Expr::Number(_) = ctx.get(base) {
            if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(e1), ctx.get(e2)) {
                let sum = n1 + n2;
                if sum.is_integer() {
                    return true;
                }
                let num = sum.numer().abs();
                let den = sum.denom().abs();
                return num < den;
            }
        }
        true
    };

    if let Some((lhs, rhs)) = as_mul(ctx, expr) {
        let lhs_pow = as_pow(ctx, lhs);
        let rhs_pow = as_pow(ctx, rhs);

        if let (Some((base1, exp1)), Some((base2, exp2))) = (lhs_pow, rhs_pow) {
            if compare_expr(ctx, base1, base2) == Ordering::Equal
                && should_combine(ctx, base1, exp1, exp2)
            {
                let sum_exp = add_exp(ctx, exp1, exp2);
                let rewritten = ctx.add(Expr::Pow(base1, sum_exp));
                return Some(PowerProductRewrite {
                    rewritten,
                    desc: "Combine powers with same base",
                });
            }
        }

        if let Some((base1, exp1)) = lhs_pow {
            if compare_expr(ctx, base1, rhs) == Ordering::Equal {
                let one = ctx.num(1);
                if should_combine(ctx, base1, exp1, one) {
                    let sum_exp = add_exp(ctx, exp1, one);
                    let rewritten = ctx.add(Expr::Pow(base1, sum_exp));
                    return Some(PowerProductRewrite {
                        rewritten,
                        desc: "Combine power and base",
                    });
                }
            }
        }

        if let Some((base2, exp2)) = rhs_pow {
            if compare_expr(ctx, base2, lhs) == Ordering::Equal {
                let one = ctx.num(1);
                if should_combine(ctx, base2, one, exp2) {
                    let sum_exp = add_exp(ctx, one, exp2);
                    let rewritten = ctx.add(Expr::Pow(base2, sum_exp));
                    return Some(PowerProductRewrite {
                        rewritten,
                        desc: "Combine base and power",
                    });
                }
            }
        }

        if compare_expr(ctx, lhs, rhs) == Ordering::Equal {
            let two = ctx.num(2);
            let rewritten = ctx.add(Expr::Pow(lhs, two));
            return Some(PowerProductRewrite {
                rewritten,
                desc: "Multiply identical terms",
            });
        }

        if let Some((rl, rr)) = as_mul(ctx, rhs) {
            if compare_expr(ctx, lhs, rl) == Ordering::Equal {
                let two = ctx.num(2);
                let x_squared = ctx.add(Expr::Pow(lhs, two));
                let rewritten = mul2_raw(ctx, x_squared, rr);
                return Some(PowerProductRewrite {
                    rewritten,
                    desc: "Combine nested identical terms",
                });
            }

            let rl_pow = as_pow(ctx, rl);

            if let (Some((base1, exp1)), Some((base2, exp2))) = (lhs_pow, rl_pow) {
                if compare_expr(ctx, base1, base2) == Ordering::Equal {
                    let sum_exp = add_exp(ctx, exp1, exp2);
                    let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                    let rewritten = mul2_raw(ctx, new_pow, rr);
                    return Some(PowerProductRewrite {
                        rewritten,
                        desc: "Combine nested powers",
                    });
                }
            }

            if let Some((base2, exp2)) = rl_pow {
                if compare_expr(ctx, lhs, base2) == Ordering::Equal {
                    let one = ctx.num(1);
                    let sum_exp = add_exp(ctx, exp2, one);
                    let new_pow = ctx.add(Expr::Pow(base2, sum_exp));
                    let rewritten = mul2_raw(ctx, new_pow, rr);
                    return Some(PowerProductRewrite {
                        rewritten,
                        desc: "Combine base and nested power",
                    });
                }
            }

            if let Some((base1, exp1)) = lhs_pow {
                if compare_expr(ctx, base1, rl) == Ordering::Equal {
                    let one = ctx.num(1);
                    let sum_exp = ctx.add(Expr::Add(exp1, one));
                    let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                    let rewritten = mul2_raw(ctx, new_pow, rr);
                    return Some(PowerProductRewrite {
                        rewritten,
                        desc: "Combine power and nested base",
                    });
                }
            }

            if let Some((ll, lr)) = as_mul(ctx, lhs) {
                if let Expr::Number(_) = ctx.get(ll) {
                    let lr_pow = as_pow(ctx, lr);

                    if let (Some((base1, exp1)), Some((base2, exp2))) = (lr_pow, rhs_pow) {
                        if compare_expr(ctx, base1, base2) == Ordering::Equal {
                            let sum_exp = add_exp(ctx, exp1, exp2);
                            let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                            let rewritten = mul2_raw(ctx, ll, new_pow);
                            return Some(PowerProductRewrite {
                                rewritten,
                                desc: "Combine coeff-power and power",
                            });
                        }
                    }

                    if let Some((base1, exp1)) = lr_pow {
                        if compare_expr(ctx, base1, rhs) == Ordering::Equal {
                            let one = ctx.num(1);
                            let sum_exp = ctx.add(Expr::Add(exp1, one));
                            let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                            let rewritten = mul2_raw(ctx, ll, new_pow);
                            return Some(PowerProductRewrite {
                                rewritten,
                                desc: "Combine coeff-power and base",
                            });
                        }
                    }
                }
            }

            if let Some((ll, lr)) = as_mul(ctx, lhs) {
                if let Expr::Number(_) = ctx.get(ll) {
                    if let Some((base2, exp2)) = rhs_pow {
                        if compare_expr(ctx, lr, base2) == Ordering::Equal {
                            let one = ctx.num(1);
                            let sum_exp = add_exp(ctx, exp2, one);
                            let new_pow = ctx.add(Expr::Pow(base2, sum_exp));
                            let rewritten = mul2_raw(ctx, ll, new_pow);
                            return Some(PowerProductRewrite {
                                rewritten,
                                desc: "Combine coeff-base and power",
                            });
                        }
                    }
                }
            }

            if let Some((base2, exp2)) = rl_pow {
                if compare_expr(ctx, lhs, base2) == Ordering::Equal {
                    let one = ctx.num(1);
                    let sum_exp = ctx.add(Expr::Add(one, exp2));
                    let new_pow = ctx.add(Expr::Pow(base2, sum_exp));
                    let rewritten = mul2_raw(ctx, new_pow, rr);
                    return Some(PowerProductRewrite {
                        rewritten,
                        desc: "Combine nested base and power",
                    });
                }
            }
        }
    }
    None
}

/// Try distribution of power over product:
/// `(a*b)^n -> a^n * b^n`.
pub fn try_rewrite_power_product_distribution_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerProductRewrite> {
    if is_canonical_form(ctx, expr) {
        return None;
    }

    if let Some((base, exp)) = as_pow(ctx, expr) {
        if let Some((a, b)) = as_mul(ctx, base) {
            if let Expr::Number(exp_num) = ctx.get(exp) {
                let denom = exp_num.denom();
                if denom > &num_bigint::BigInt::from(1)
                    && !can_distribute_root_safely(ctx, base, denom)
                {
                    return None;
                }
            }

            let a_is_num = matches!(ctx.get(a), Expr::Number(_));
            let b_is_num = matches!(ctx.get(b), Expr::Number(_));
            let exp_is_numeric = matches!(ctx.get(exp), Expr::Number(_));
            if (a_is_num || b_is_num) && !exp_is_numeric {
                return None;
            }

            let a_pow = ctx.add(Expr::Pow(a, exp));
            let b_pow = ctx.add(Expr::Pow(b, exp));
            let rewritten = mul2_raw(ctx, a_pow, b_pow);
            return Some(PowerProductRewrite {
                rewritten,
                desc: "Distribute power over product",
            });
        }
    }

    None
}

/// Try same-exponent product rewrites:
/// - `a^n * b^n -> (a*b)^n`
/// - nested variant: `a^n * (b^n * c) -> (a*b)^n * c`
pub fn try_rewrite_product_same_exponent_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerProductRewrite> {
    if let Some((lhs, rhs)) = as_mul(ctx, expr) {
        let lhs_pow = as_pow(ctx, lhs);
        let rhs_pow = as_pow(ctx, rhs);

        if let (Some((base1, exp1)), Some((base2, exp2))) = (lhs_pow, rhs_pow) {
            if compare_expr(ctx, exp1, exp2) == Ordering::Equal {
                let base1_is_num = matches!(ctx.get(base1), Expr::Number(_));
                let base2_is_num = matches!(ctx.get(base2), Expr::Number(_));
                let base1_has_num = base1_is_num || has_numeric_factor(ctx, base1);
                let base2_has_num = base2_is_num || has_numeric_factor(ctx, base2);
                if !base1_has_num && !base2_has_num {
                    return None;
                }

                let new_base = mul2_raw(ctx, base1, base2);
                let rewritten = ctx.add(Expr::Pow(new_base, exp1));
                return Some(PowerProductRewrite {
                    rewritten,
                    desc: "Combine powers with same exponent",
                });
            }
        }

        if let Some((base1, exp1)) = lhs_pow {
            if let Some((rl, rr)) = as_mul(ctx, rhs) {
                if let Some((base2, exp2)) = as_pow(ctx, rl) {
                    if compare_expr(ctx, exp1, exp2) == Ordering::Equal {
                        let base1_is_num = matches!(ctx.get(base1), Expr::Number(_));
                        let base2_is_num = matches!(ctx.get(base2), Expr::Number(_));
                        let base1_has_num = base1_is_num || has_numeric_factor(ctx, base1);
                        let base2_has_num = base2_is_num || has_numeric_factor(ctx, base2);
                        if !base1_has_num && !base2_has_num {
                            return None;
                        }

                        let new_base = mul2_raw(ctx, base1, base2);
                        let combined_pow = ctx.add(Expr::Pow(new_base, exp1));
                        let rewritten = mul2_raw(ctx, combined_pow, rr);
                        return Some(PowerProductRewrite {
                            rewritten,
                            desc: "Combine nested powers with same exponent",
                        });
                    }
                }
            }
        }
    }

    None
}

/// Try same-exponent quotient rewrite:
/// `a^n / b^n -> (a/b)^n`.
pub fn try_rewrite_quotient_same_exponent_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerProductRewrite> {
    let fp = FractionParts::from(&*ctx, expr);
    if !fp.is_fraction() {
        return None;
    }
    let (num, den, _) = fp.to_num_den(ctx);

    if let (Expr::Pow(base_num, exp_num), Expr::Pow(base_den, exp_den)) =
        (ctx.get(num), ctx.get(den))
    {
        let (base_num, exp_num, base_den, exp_den) = (*base_num, *exp_num, *base_den, *exp_den);
        if compare_expr(ctx, exp_num, exp_den) == Ordering::Equal {
            let base_num_is_num = matches!(ctx.get(base_num), Expr::Number(_));
            let base_den_is_num = matches!(ctx.get(base_den), Expr::Number(_));
            let base_num_has_num = base_num_is_num || has_numeric_factor(ctx, base_num);
            let base_den_has_num = base_den_is_num || has_numeric_factor(ctx, base_den);
            if !base_num_has_num && !base_den_has_num {
                return None;
            }

            let new_base = ctx.add(Expr::Div(base_num, base_den));
            let rewritten = ctx.add(Expr::Pow(new_base, exp_num));
            return Some(PowerProductRewrite {
                rewritten,
                desc: "a^n / b^n = (a/b)^n",
            });
        }
    }

    None
}

/// Try quotient-power distribution rewrite:
/// `(a/b)^n -> a^n / b^n`.
pub fn try_rewrite_power_quotient_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerProductRewrite> {
    if let Some((base, exp)) = as_pow(ctx, expr) {
        if let Expr::Number(exp_num) = ctx.get(exp) {
            let denom = exp_num.denom();
            if denom > &num_bigint::BigInt::from(1) && !can_distribute_root_safely(ctx, base, denom)
            {
                return None;
            }
        }

        if let Some((num, den)) = as_div(ctx, base) {
            let new_num = ctx.add(Expr::Pow(num, exp));
            let new_den = ctx.add(Expr::Pow(den, exp));
            let rewritten = ctx.add(Expr::Div(new_num, new_den));
            return Some(PowerProductRewrite {
                rewritten,
                desc: "Distribute power over quotient",
            });
        }
    }
    None
}

/// Try exponential quotient rewrites for base `e`:
/// - `e^a / e^b -> e^(a-b)`
/// - `e / e^b -> e^(1-b)`
/// - `e^a / e -> e^(a-1)`
pub fn try_rewrite_exp_quotient_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerProductRewrite> {
    if let Some((num, den)) = as_div(ctx, expr) {
        let num_pow = as_pow(ctx, num);
        let den_pow = as_pow(ctx, den);

        if let (Some((num_base, num_exp)), Some((den_base, den_exp))) = (num_pow, den_pow) {
            if is_e_constant_expr(ctx, num_base) && is_e_constant_expr(ctx, den_base) {
                let diff = ctx.add(Expr::Sub(num_exp, den_exp));
                let e = ctx.add(Expr::Constant(Constant::E));
                let rewritten = ctx.add(Expr::Pow(e, diff));
                return Some(PowerProductRewrite {
                    rewritten,
                    desc: "e^a / e^b = e^(a-b)",
                });
            }
        }

        if is_e_constant_expr(ctx, num) {
            if let Some((den_base, den_exp)) = den_pow {
                if is_e_constant_expr(ctx, den_base) {
                    let one = ctx.num(1);
                    let diff = ctx.add(Expr::Sub(one, den_exp));
                    let e = ctx.add(Expr::Constant(Constant::E));
                    let rewritten = ctx.add(Expr::Pow(e, diff));
                    return Some(PowerProductRewrite {
                        rewritten,
                        desc: "e / e^b = e^(1-b)",
                    });
                }
            }
        }

        if is_e_constant_expr(ctx, den) {
            if let Some((num_base, num_exp)) = num_pow {
                if is_e_constant_expr(ctx, num_base) {
                    let one = ctx.num(1);
                    let diff = ctx.add(Expr::Sub(num_exp, one));
                    let e = ctx.add(Expr::Constant(Constant::E));
                    let rewritten = ctx.add(Expr::Pow(e, diff));
                    return Some(PowerProductRewrite {
                        rewritten,
                        desc: "e^a / e = e^(a-1)",
                    });
                }
            }
        }
    }
    None
}

/// Try combining powers with the same base in n-ary multiplication trees.
pub fn try_rewrite_mul_nary_combine_powers_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<PowerProductRewrite> {
    if !matches!(ctx.get(expr), Expr::Mul(_, _)) {
        return None;
    }

    let factors = mul_leaves(ctx, expr);
    if factors.len() > 12 || factors.len() < 2 {
        return None;
    }

    let mut base_exp_pairs: Vec<(ExprId, Option<BigRational>, bool)> = Vec::new();
    for &factor in &factors {
        match ctx.get(factor) {
            Expr::Pow(base, exp) => {
                if let Expr::Number(n) = ctx.get(*exp) {
                    base_exp_pairs.push((*base, Some(n.clone()), true));
                } else {
                    base_exp_pairs.push((factor, None, false));
                }
            }
            Expr::Number(_) => {
                base_exp_pairs.push((factor, Some(BigRational::one()), false));
            }
            _ => {
                base_exp_pairs.push((factor, Some(BigRational::one()), true));
            }
        }
    }

    let mut combined: Vec<(ExprId, BigRational, usize)> = Vec::new();
    let mut absorbed = vec![false; factors.len()];
    let mut any_combined = false;

    for i in 0..base_exp_pairs.len() {
        if absorbed[i] {
            continue;
        }

        let (base_i, exp_i, is_pow_i) = &base_exp_pairs[i];
        let Some(mut sum_exp) = exp_i.clone() else {
            continue;
        };

        if !is_pow_i {
            continue;
        }

        let mut found_match = false;
        for j in (i + 1)..base_exp_pairs.len() {
            if absorbed[j] {
                continue;
            }

            let (base_j, exp_j, is_pow_j) = &base_exp_pairs[j];
            if !is_pow_j {
                continue;
            }

            let Some(exp_j_val) = exp_j else {
                continue;
            };

            if compare_expr(ctx, *base_i, *base_j) == Ordering::Equal {
                sum_exp += exp_j_val;
                absorbed[j] = true;
                found_match = true;
            }
        }

        if found_match {
            absorbed[i] = true;
            combined.push((*base_i, sum_exp, i));
            any_combined = true;
        }
    }

    if !any_combined {
        return None;
    }

    let mut result_factors: Vec<ExprId> = Vec::new();
    let mut combined_map: HashMap<usize, (ExprId, BigRational)> = HashMap::new();
    for (base, sum_exp, first_idx) in &combined {
        combined_map.insert(*first_idx, (*base, sum_exp.clone()));
    }

    for i in 0..factors.len() {
        if let Some((base, sum_exp)) = combined_map.get(&i) {
            let new_factor = if sum_exp.is_one() {
                *base
            } else if sum_exp.is_zero() {
                ctx.num(1)
            } else {
                let exp_id = ctx.add(Expr::Number(sum_exp.clone()));
                ctx.add(Expr::Pow(*base, exp_id))
            };
            result_factors.push(new_factor);
        } else if !absorbed[i] {
            result_factors.push(factors[i]);
        }
    }

    if result_factors.is_empty() {
        return Some(PowerProductRewrite {
            rewritten: ctx.num(1),
            desc: "All factors cancelled",
        });
    }

    let (&last, rest) = result_factors.split_last()?;
    let mut rewritten = last;
    for &factor in rest.iter().rev() {
        rewritten = mul2_raw(ctx, factor, rewritten);
    }

    if result_factors.len() < factors.len() {
        Some(PowerProductRewrite {
            rewritten,
            desc: "Combine powers with same base (n-ary)",
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{
        try_rewrite_exp_quotient_expr, try_rewrite_mul_nary_combine_powers_expr,
        try_rewrite_power_product_distribution_expr, try_rewrite_power_quotient_expr,
        try_rewrite_product_power_expr, try_rewrite_product_same_exponent_expr,
        try_rewrite_quotient_same_exponent_expr,
    };
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn rewrites_product_power_same_base() {
        let mut ctx = Context::new();
        let expr = parse("x^2 * x^3", &mut ctx).expect("parse");
        assert!(try_rewrite_product_power_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn rewrites_product_same_exponent() {
        let mut ctx = Context::new();
        let expr = parse("2^x * 3^x", &mut ctx).expect("parse");
        assert!(try_rewrite_product_same_exponent_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn rewrites_quotient_same_exponent() {
        let mut ctx = Context::new();
        let expr = parse("2^x / 3^x", &mut ctx).expect("parse");
        assert!(try_rewrite_quotient_same_exponent_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn rewrites_power_quotient_distribution() {
        let mut ctx = Context::new();
        let expr = parse("(a/b)^3", &mut ctx).expect("parse");
        assert!(try_rewrite_power_quotient_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn rewrites_exp_quotient() {
        let mut ctx = Context::new();
        let expr = parse("e^a / e^b", &mut ctx).expect("parse");
        assert!(try_rewrite_exp_quotient_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn rewrites_mul_nary_combine_powers() {
        let mut ctx = Context::new();
        let expr = parse("x^2 * y * x^3", &mut ctx).expect("parse");
        assert!(try_rewrite_mul_nary_combine_powers_expr(&mut ctx, expr).is_some());
    }

    #[test]
    fn rewrites_power_product_distribution() {
        let mut ctx = Context::new();
        let expr = parse("(a*b)^3", &mut ctx).expect("parse");
        assert!(try_rewrite_power_product_distribution_expr(&mut ctx, expr).is_some());
    }
}
