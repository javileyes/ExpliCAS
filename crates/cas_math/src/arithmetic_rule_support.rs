//! Planning helpers for arithmetic rewrite rules.

use crate::build::mul2_raw;
use crate::expr_destructure::{as_add, as_div, as_mul, as_sub};
use crate::expr_rewrite::smart_mul;
use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};

#[derive(Debug, Clone)]
pub struct ArithmeticRewritePlan {
    pub rewritten: ExprId,
    pub description: String,
}

/// `x + 0 = x` and `0 + x = x`.
pub fn try_rewrite_add_zero_expr(ctx: &Context, expr: ExprId) -> Option<ArithmeticRewritePlan> {
    let (lhs, rhs) = as_add(ctx, expr)?;

    if let Expr::Number(n) = ctx.get(rhs) {
        if n.is_zero() {
            return Some(ArithmeticRewritePlan {
                rewritten: lhs,
                description: "x + 0 = x".to_string(),
            });
        }
    }
    if let Expr::Number(n) = ctx.get(lhs) {
        if n.is_zero() {
            return Some(ArithmeticRewritePlan {
                rewritten: rhs,
                description: "0 + x = x".to_string(),
            });
        }
    }
    None
}

/// `x * 1 = x` and `1 * x = x`.
pub fn try_rewrite_mul_one_expr(ctx: &Context, expr: ExprId) -> Option<ArithmeticRewritePlan> {
    let (lhs, rhs) = as_mul(ctx, expr)?;

    if let Expr::Number(n) = ctx.get(rhs) {
        if n.is_one() {
            return Some(ArithmeticRewritePlan {
                rewritten: lhs,
                description: "x * 1 = x".to_string(),
            });
        }
    }
    if let Expr::Number(n) = ctx.get(lhs) {
        if n.is_one() {
            return Some(ArithmeticRewritePlan {
                rewritten: rhs,
                description: "1 * x = x".to_string(),
            });
        }
    }
    None
}

/// Try to combine numeric constants inside Add/Sub/Mul/Div expressions.
pub fn try_rewrite_combine_constants_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ArithmeticRewritePlan> {
    if let Some((lhs, rhs)) = as_add(ctx, expr) {
        if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(lhs), ctx.get(rhs)) {
            let (n1, n2) = (n1.clone(), n2.clone());
            let sum = &n1 + &n2;
            let description = if n2 < BigRational::from_integer(0.into()) {
                let abs_n2 = -&n2;
                format!("{} - {} = {}", n1, abs_n2, sum)
            } else {
                format!("{} + {} = {}", n1, n2, sum)
            };
            return Some(ArithmeticRewritePlan {
                rewritten: ctx.add(Expr::Number(sum)),
                description,
            });
        }

        if let Expr::Number(n1) = ctx.get(lhs) {
            let n1 = n1.clone();
            if let Some((rl, rr)) = as_add(ctx, rhs) {
                if let Expr::Number(n2) = ctx.get(rl) {
                    let n2 = n2.clone();
                    let sum = &n1 + &n2;
                    let sum_expr = ctx.add(Expr::Number(sum));
                    return Some(ArithmeticRewritePlan {
                        rewritten: ctx.add(Expr::Add(sum_expr, rr)),
                        description: format!("Combine nested constants: {} + {}", n1, n2),
                    });
                }
            }
        }
    }

    if let Some((lhs, rhs)) = as_mul(ctx, expr) {
        if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(lhs), ctx.get(rhs)) {
            let (n1, n2) = (n1.clone(), n2.clone());
            let prod = &n1 * &n2;
            return Some(ArithmeticRewritePlan {
                rewritten: ctx.add(Expr::Number(prod.clone())),
                description: format!("{} * {} = {}", n1, n2, prod),
            });
        }

        let mut factors = Vec::new();
        let mut stack = vec![expr];
        while let Some(id) = stack.pop() {
            if let Expr::Mul(l, r) = ctx.get(id) {
                stack.push(*r);
                stack.push(*l);
            } else {
                factors.push(id);
            }
        }

        let mut numeric_factors: Vec<BigRational> = Vec::new();
        let mut non_numeric: Vec<ExprId> = Vec::new();

        for &factor in &factors {
            if let Expr::Number(n) = ctx.get(factor) {
                numeric_factors.push(n.clone());
            } else {
                non_numeric.push(factor);
            }
        }

        if numeric_factors.len() >= 2 {
            let product = numeric_factors
                .iter()
                .fold(BigRational::from_integer(1.into()), |acc, n| acc * n);

            if product.is_zero() {
                return Some(ArithmeticRewritePlan {
                    rewritten: ctx.num(0),
                    description: "0 * x = 0".to_string(),
                });
            }

            let rewritten = if product.is_one() && !non_numeric.is_empty() {
                let mut result = non_numeric[0];
                for &factor in &non_numeric[1..] {
                    result = smart_mul(ctx, result, factor);
                }
                result
            } else if non_numeric.is_empty() {
                ctx.add(Expr::Number(product.clone()))
            } else {
                let prod_expr = ctx.add(Expr::Number(product.clone()));
                let mut result = prod_expr;
                for &factor in &non_numeric {
                    result = smart_mul(ctx, result, factor);
                }
                result
            };

            let nums: Vec<String> = numeric_factors.iter().map(|n| format!("{n}")).collect();
            return Some(ArithmeticRewritePlan {
                rewritten,
                description: format!(
                    "Combine nested constants: {} = {}",
                    nums.join(" * "),
                    product
                ),
            });
        }

        if let Expr::Number(n1) = ctx.get(lhs) {
            let n1 = n1.clone();
            if let Some((rl, rr)) = as_mul(ctx, rhs) {
                if let Expr::Number(n2) = ctx.get(rl) {
                    let n2 = n2.clone();
                    let prod = &n1 * &n2;
                    let prod_expr = ctx.add(Expr::Number(prod));
                    return Some(ArithmeticRewritePlan {
                        rewritten: smart_mul(ctx, prod_expr, rr),
                        description: format!("Combine nested constants: {} * {}", n1, n2),
                    });
                }
            }

            if let Some((num, den)) = as_div(ctx, rhs) {
                if let Expr::Number(n2) = ctx.get(den) {
                    let n2 = n2.clone();
                    if !n2.is_zero() {
                        let ratio = &n1 / &n2;
                        let ratio_expr = ctx.add(Expr::Number(ratio));
                        return Some(ArithmeticRewritePlan {
                            rewritten: smart_mul(ctx, ratio_expr, num),
                            description: format!("{} * (x / {}) -> ({} / {}) * x", n1, n2, n1, n2),
                        });
                    }
                }
            }
        }
    }

    if let Some((lhs, rhs)) = as_sub(ctx, expr) {
        if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(lhs), ctx.get(rhs)) {
            let (n1, n2) = (n1.clone(), n2.clone());
            let diff = &n1 - &n2;
            return Some(ArithmeticRewritePlan {
                rewritten: ctx.add(Expr::Number(diff.clone())),
                description: format!("{} - {} = {}", n1, n2, diff),
            });
        }
    }

    if let Some((lhs, rhs)) = as_div(ctx, expr) {
        if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(lhs), ctx.get(rhs)) {
            let (n1, n2) = (n1.clone(), n2.clone());
            if !n2.is_zero() {
                let quot = &n1 / &n2;
                return Some(ArithmeticRewritePlan {
                    rewritten: ctx.add(Expr::Number(quot.clone())),
                    description: format!("{} / {} = {}", n1, n2, quot),
                });
            }
            return Some(ArithmeticRewritePlan {
                rewritten: ctx.add(Expr::Constant(cas_ast::Constant::Undefined)),
                description: "Division by zero".to_string(),
            });
        }

        if let Expr::Number(d) = ctx.get(rhs) {
            let d = d.clone();
            if !d.is_zero() {
                if let Some((ml, mr)) = as_mul(ctx, lhs) {
                    if let Expr::Number(c) = ctx.get(ml) {
                        let c = c.clone();
                        let ratio = &c / &d;
                        let ratio_expr = ctx.add(Expr::Number(ratio));
                        return Some(ArithmeticRewritePlan {
                            rewritten: smart_mul(ctx, ratio_expr, mr),
                            description: format!("({} * x) / {} -> ({} / {}) * x", c, d, c, d),
                        });
                    }

                    if let Expr::Number(c) = ctx.get(mr) {
                        let c = c.clone();
                        let ratio = &c / &d;
                        let ratio_expr = ctx.add(Expr::Number(ratio));
                        return Some(ArithmeticRewritePlan {
                            rewritten: smart_mul(ctx, ratio_expr, ml),
                            description: format!("(x * {}) / {} -> ({} / {}) * x", c, d, c, d),
                        });
                    }
                }
            }
        }
    }

    None
}

/// Try to simplify numeric sums in exponents:
/// `x^(1/2 + 1/3) -> x^(5/6)`.
pub fn try_rewrite_simplify_numeric_exponents_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ArithmeticRewritePlan> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let base = *base;
    let exp = *exp;

    let mut addends: Vec<BigRational> = Vec::new();
    let mut stack = vec![exp];
    let mut all_numeric = true;

    while let Some(id) = stack.pop() {
        match ctx.get(id) {
            Expr::Add(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Expr::Number(n) => addends.push(n.clone()),
            Expr::Div(num, den) => {
                if let (Expr::Number(n), Expr::Number(d)) = (ctx.get(*num), ctx.get(*den)) {
                    if !d.is_zero() {
                        addends.push(n / d);
                    } else {
                        all_numeric = false;
                    }
                } else {
                    all_numeric = false;
                }
            }
            _ => {
                all_numeric = false;
            }
        }
    }

    if !(all_numeric && addends.len() >= 2) {
        return None;
    }

    let sum: BigRational = addends.iter().sum();
    let new_exp = ctx.add(Expr::Number(sum.clone()));
    let rewritten = ctx.add(Expr::Pow(base, new_exp));

    let addend_strs: Vec<String> = addends
        .iter()
        .map(|r| {
            if r.is_integer() {
                format!("{}", r.numer())
            } else {
                format!("({}/{})", r.numer(), r.denom())
            }
        })
        .collect();
    let sum_str = if sum.is_integer() {
        format!("{}", sum.numer())
    } else {
        format!("{}/{}", sum.numer(), sum.denom())
    };

    Some(ArithmeticRewritePlan {
        rewritten,
        description: format!("{} = {}", addend_strs.join(" + "), sum_str),
    })
}

/// Normalize negation placement inside products.
pub fn try_rewrite_normalize_mul_neg_expr(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ArithmeticRewritePlan> {
    let Expr::Mul(l, r) = ctx.get(expr) else {
        return None;
    };
    let l = *l;
    let r = *r;

    let l_neg = if let Expr::Neg(inner) = ctx.get(l) {
        Some(*inner)
    } else {
        None
    };
    let r_neg = if let Expr::Neg(inner) = ctx.get(r) {
        Some(*inner)
    } else {
        None
    };

    match (l_neg, r_neg) {
        (Some(a), Some(b)) => {
            let new_mul = mul2_raw(ctx, a, b);
            Some(ArithmeticRewritePlan {
                rewritten: new_mul,
                description: "(-a) * (-b) = a * b".to_string(),
            })
        }
        (Some(a), None) => {
            let new_mul = mul2_raw(ctx, a, r);
            Some(ArithmeticRewritePlan {
                rewritten: ctx.add(Expr::Neg(new_mul)),
                description: "(-a) * b = -(a * b)".to_string(),
            })
        }
        (None, Some(b)) => {
            let new_mul = mul2_raw(ctx, l, b);
            Some(ArithmeticRewritePlan {
                rewritten: ctx.add(Expr::Neg(new_mul)),
                description: "a * (-b) = -(a * b)".to_string(),
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        try_rewrite_add_zero_expr, try_rewrite_combine_constants_expr, try_rewrite_mul_one_expr,
        try_rewrite_normalize_mul_neg_expr, try_rewrite_simplify_numeric_exponents_expr,
    };
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: cas_ast::ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn combine_constants_add() {
        let mut ctx = Context::new();
        let expr = parse("2 + 3", &mut ctx).expect("parse");
        let rewrite = try_rewrite_combine_constants_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rendered(&ctx, rewrite.rewritten), "5");
    }

    #[test]
    fn add_zero_identity() {
        let mut ctx = Context::new();
        let expr = parse("x + 0", &mut ctx).expect("parse");
        let rewrite = try_rewrite_add_zero_expr(&ctx, expr).expect("rewrite");
        assert_eq!(rendered(&ctx, rewrite.rewritten), "x");
    }

    #[test]
    fn mul_one_identity() {
        let mut ctx = Context::new();
        let expr = parse("1 * y", &mut ctx).expect("parse");
        let rewrite = try_rewrite_mul_one_expr(&ctx, expr).expect("rewrite");
        assert_eq!(rendered(&ctx, rewrite.rewritten), "y");
    }

    #[test]
    fn combine_constants_mul_chain() {
        let mut ctx = Context::new();
        let expr = parse("(2 * x) * 3", &mut ctx).expect("parse");
        let rewrite = try_rewrite_combine_constants_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rendered(&ctx, rewrite.rewritten), "6 * x");
    }

    #[test]
    fn simplify_numeric_exponents_sum() {
        let mut ctx = Context::new();
        let expr = parse("x^((1/2)+(1/3))", &mut ctx).expect("parse");
        let rewrite = try_rewrite_simplify_numeric_exponents_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rendered(&ctx, rewrite.rewritten), "x^(5/6)");
    }

    #[test]
    fn normalize_mul_neg_right() {
        let mut ctx = Context::new();
        let expr = parse("a * (-b)", &mut ctx).expect("parse");
        let rewrite = try_rewrite_normalize_mul_neg_expr(&mut ctx, expr).expect("rewrite");
        assert_eq!(rendered(&ctx, rewrite.rewritten), "-(a * b)");
    }
}
