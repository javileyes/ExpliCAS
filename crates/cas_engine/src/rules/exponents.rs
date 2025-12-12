use crate::define_rule;
use crate::helpers::is_half;
use crate::ordering::compare_expr;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_integer::Integer;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;

/// Helper: Add two exponents, folding if both are constants
/// This prevents ugly exponents like x^(1+2) and produces x^3 instead
fn add_exp(ctx: &mut Context, e1: ExprId, e2: ExprId) -> ExprId {
    if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(e1), ctx.get(e2)) {
        let sum = n1 + n2;
        ctx.add(Expr::Number(sum))
    } else {
        ctx.add(Expr::Add(e1, e2))
    }
}

/// Helper: Multiply two exponents, folding if both are constants
/// This prevents ugly exponents like x^(2*3) and produces x^6 instead
fn mul_exp(ctx: &mut Context, e1: ExprId, e2: ExprId) -> ExprId {
    if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(e1), ctx.get(e2)) {
        let prod = n1 * n2;
        ctx.add(Expr::Number(prod))
    } else {
        ctx.add(Expr::Mul(e1, e2))
    }
}

define_rule!(ProductPowerRule, "Product of Powers", |ctx, expr| {
    // x^a * x^b -> x^(a+b)
    let should_combine = |ctx: &Context, base: ExprId, e1: ExprId, e2: ExprId| -> bool {
        if let Expr::Number(_) = ctx.get(base) {
            if let (Expr::Number(n1), Expr::Number(n2)) = (ctx.get(e1), ctx.get(e2)) {
                let sum = n1 + n2;
                if sum.is_integer() {
                    return true;
                }
                // Check if proper fraction: |num| < den
                let num = sum.numer().abs();
                let den = sum.denom().abs();
                return num < den;
            }
        }
        true
    };

    let expr_data = ctx.get(expr).clone();
    if let Expr::Mul(lhs, rhs) = expr_data {
        let lhs_data = ctx.get(lhs).clone();
        let rhs_data = ctx.get(rhs).clone();

        // Case 1: Both are powers with same base: x^a * x^b
        if let (Expr::Pow(base1, exp1), Expr::Pow(base2, exp2)) = (&lhs_data, &rhs_data) {
            if compare_expr(ctx, *base1, *base2) == Ordering::Equal
                && should_combine(ctx, *base1, *exp1, *exp2)
            {
                let sum_exp = add_exp(ctx, *exp1, *exp2);
                let new_expr = ctx.add(Expr::Pow(*base1, sum_exp));
                return Some(Rewrite {
                    new_expr,
                    description: "Combine powers with same base".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        }
        // Case 2: One is power, one is base: x^a * x -> x^(a+1)
        // Left is power
        if let Expr::Pow(base1, exp1) = &lhs_data {
            if compare_expr(ctx, *base1, rhs) == Ordering::Equal {
                let one = ctx.num(1);
                if should_combine(ctx, *base1, *exp1, one) {
                    let sum_exp = add_exp(ctx, *exp1, one);
                    let new_expr = ctx.add(Expr::Pow(*base1, sum_exp));
                    return Some(Rewrite {
                        new_expr,
                        description: "Combine power and base".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                }
            }
        }
        // Right is power
        if let Expr::Pow(base2, exp2) = &rhs_data {
            if compare_expr(ctx, *base2, lhs) == Ordering::Equal {
                let one = ctx.num(1);
                if should_combine(ctx, *base2, one, *exp2) {
                    let sum_exp = add_exp(ctx, one, *exp2);
                    let new_expr = ctx.add(Expr::Pow(*base2, sum_exp));
                    return Some(Rewrite {
                        new_expr,
                        description: "Combine base and power".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                }
            }
        }
        // Case 3: Both are same base (implicit power 1): x * x -> x^2
        if compare_expr(ctx, lhs, rhs) == Ordering::Equal {
            let two = ctx.num(2);
            let new_expr = ctx.add(Expr::Pow(lhs, two));
            return Some(Rewrite {
                new_expr,
                description: "Multiply identical terms".to_string(),
                before_local: None,
                after_local: None,
            });
        }

        // Case 4: Nested Multiplication: x * (x * y) -> x^2 * y
        // We rely on CanonicalizeMulRule to have sorted terms, so identical bases are adjacent.
        // Check if rhs is a Mul(rl, rr) and lhs == rl
        if let Expr::Mul(rl, rr) = rhs_data {
            // x * (x * y)
            if compare_expr(ctx, lhs, rl) == Ordering::Equal {
                let two = ctx.num(2);
                let x_squared = ctx.add(Expr::Pow(lhs, two));
                let new_expr = ctx.add(Expr::Mul(x_squared, rr));
                return Some(Rewrite {
                    new_expr,
                    description: "Combine nested identical terms".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }

            // x^a * (x^b * y) -> x^(a+b) * y
            let lhs_pow = if let Expr::Pow(b, e) = &lhs_data {
                Some((*b, *e))
            } else {
                None
            };
            let rhs_pow = if let Expr::Pow(b, e) = ctx.get(rl) {
                Some((*b, *e))
            } else {
                None
            };

            if let (Some((base1, exp1)), Some((base2, exp2))) = (lhs_pow, rhs_pow) {
                if compare_expr(ctx, base1, base2) == Ordering::Equal {
                    let sum_exp = add_exp(ctx, exp1, exp2);
                    let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                    let new_expr = ctx.add(Expr::Mul(new_pow, rr));
                    return Some(Rewrite {
                        new_expr,
                        description: "Combine nested powers".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                }
            }

            // x * (x^a * y) -> x^(a+1) * y
            if let Some((base2, exp2)) = rhs_pow {
                if compare_expr(ctx, lhs, base2) == Ordering::Equal {
                    let one = ctx.num(1);
                    let sum_exp = add_exp(ctx, exp2, one);
                    let new_pow = ctx.add(Expr::Pow(base2, sum_exp));
                    let new_expr = ctx.add(Expr::Mul(new_pow, rr));
                    return Some(Rewrite {
                        new_expr,
                        description: "Combine base and nested power".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                }
            }

            // x^a * (x * y) -> x^(a+1) * y
            if let Some((base1, exp1)) = lhs_pow {
                if compare_expr(ctx, base1, rl) == Ordering::Equal {
                    let one = ctx.num(1);
                    let sum_exp = ctx.add(Expr::Add(exp1, one));
                    let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                    let new_expr = ctx.add(Expr::Mul(new_pow, rr));
                    return Some(Rewrite {
                        new_expr,
                        description: "Combine power and nested base".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                }
            }

            // (c * x^a) * x^b -> c * x^(a+b)
            // Check if lhs is Mul(c, x^a)
            if let Expr::Mul(ll, lr) = lhs_data {
                // Check if ll is number (coefficient)
                if let Expr::Number(_) = ctx.get(ll) {
                    // lr is x^a ?
                    let lr_pow = if let Expr::Pow(b, e) = ctx.get(lr) {
                        Some((*b, *e))
                    } else {
                        None
                    };

                    // Check rhs is x^b
                    let rhs_pow = if let Expr::Pow(b, e) = &rhs_data {
                        Some((*b, *e))
                    } else {
                        None
                    };

                    if let (Some((base1, exp1)), Some((base2, exp2))) = (lr_pow, rhs_pow) {
                        if compare_expr(ctx, base1, base2) == Ordering::Equal {
                            let sum_exp = add_exp(ctx, exp1, exp2);
                            let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                            let new_expr = ctx.add(Expr::Mul(ll, new_pow));
                            return Some(Rewrite {
                                new_expr,
                                description: "Combine coeff-power and power".to_string(),
                                before_local: None,
                                after_local: None,
                            });
                        }
                    }

                    // Check rhs is x (implicit power 1)
                    // (c * x^a) * x -> c * x^(a+1)
                    if let Some((base1, exp1)) = lr_pow {
                        if compare_expr(ctx, base1, rhs) == Ordering::Equal {
                            let one = ctx.num(1);
                            let sum_exp = ctx.add(Expr::Add(exp1, one));
                            let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                            let new_expr = ctx.add(Expr::Mul(ll, new_pow));
                            return Some(Rewrite {
                                new_expr,
                                description: "Combine coeff-power and base".to_string(),
                                before_local: None,
                                after_local: None,
                            });
                        }
                    }
                }
            }

            // (c * x) * x^a -> c * x^(a+1)
            if let Expr::Mul(ll, lr) = lhs_data {
                if let Expr::Number(_) = ctx.get(ll) {
                    // lr is x
                    // rhs is x^a
                    let rhs_pow = if let Expr::Pow(b, e) = &rhs_data {
                        Some((*b, *e))
                    } else {
                        None
                    };

                    if let Some((base2, exp2)) = rhs_pow {
                        if compare_expr(ctx, lr, base2) == Ordering::Equal {
                            let one = ctx.num(1);
                            let sum_exp = add_exp(ctx, exp2, one);
                            let new_pow = ctx.add(Expr::Pow(base2, sum_exp));
                            let new_expr = ctx.add(Expr::Mul(ll, new_pow));
                            return Some(Rewrite {
                                new_expr,
                                description: "Combine coeff-base and power".to_string(),
                                before_local: None,
                                after_local: None,
                            });
                        }
                    }
                }
            }

            // x * (x^b * y) -> x^(1+b) * y
            if let Some((base2, exp2)) = rhs_pow {
                if compare_expr(ctx, lhs, base2) == Ordering::Equal {
                    let one = ctx.num(1);
                    let sum_exp = ctx.add(Expr::Add(one, exp2));
                    let new_pow = ctx.add(Expr::Pow(base2, sum_exp));
                    let new_expr = ctx.add(Expr::Mul(new_pow, rr));
                    return Some(Rewrite {
                        new_expr,
                        description: "Combine nested base and power".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                }
            }
        }
    }
    None
});

// a^n * b^n = (ab)^n - combines products of powers with same exponent
// Only applies when at least one base is a number to avoid infinite loop with PowerProductRule
define_rule!(
    ProductSameExponentRule,
    "Product Same Exponent",
    |ctx, expr| {
        // a^n * b^n -> (a*b)^n
        // Guard: only apply when at least one base is a number (to avoid infinite loop)
        let expr_data = ctx.get(expr).clone();
        if let Expr::Mul(lhs, rhs) = expr_data {
            let lhs_data = ctx.get(lhs).clone();
            let rhs_data = ctx.get(rhs).clone();

            // Case 1: Both are powers with same exponent: a^n * b^n
            if let (Expr::Pow(base1, exp1), Expr::Pow(base2, exp2)) = (&lhs_data, &rhs_data) {
                if compare_expr(ctx, *exp1, *exp2) == Ordering::Equal {
                    // Guard: at least one base must be a number to avoid infinite loop
                    let base1_is_num = matches!(ctx.get(*base1), Expr::Number(_));
                    let base2_is_num = matches!(ctx.get(*base2), Expr::Number(_));
                    if !base1_is_num && !base2_is_num {
                        return None; // Skip if both are non-numeric (would loop with PowerProductRule)
                    }

                    // Same exponent - combine bases
                    let new_base = ctx.add(Expr::Mul(*base1, *base2));
                    let new_expr = ctx.add(Expr::Pow(new_base, *exp1));
                    return Some(Rewrite {
                        new_expr,
                        description: "Combine powers with same exponent".to_string(),
                        before_local: None,
                        after_local: None,
                    });
                }
            }

            // Case 2: Nested: a^n * (b^n * c) -> (a*b)^n * c
            if let Expr::Pow(base1, exp1) = &lhs_data {
                if let Expr::Mul(rl, rr) = &rhs_data {
                    if let Expr::Pow(base2, exp2) = ctx.get(*rl) {
                        if compare_expr(ctx, *exp1, *exp2) == Ordering::Equal {
                            // Guard: at least one base must be a number
                            let base1_is_num = matches!(ctx.get(*base1), Expr::Number(_));
                            let base2_is_num = matches!(ctx.get(*base2), Expr::Number(_));
                            if !base1_is_num && !base2_is_num {
                                return None;
                            }

                            let new_base = ctx.add(Expr::Mul(*base1, *base2));
                            let combined_pow = ctx.add(Expr::Pow(new_base, *exp1));
                            let new_expr = ctx.add(Expr::Mul(combined_pow, *rr));
                            return Some(Rewrite {
                                new_expr,
                                description: "Combine nested powers with same exponent".to_string(),
                                before_local: None,
                                after_local: None,
                            });
                        }
                    }
                }
            }
        }
        None
    }
);

define_rule!(PowerPowerRule, "Power of a Power", |ctx, expr| {
    // (x^a)^b -> x^(a*b)
    let expr_data = ctx.get(expr).clone();
    if let Expr::Pow(base, outer_exp) = expr_data {
        let base_data = ctx.get(base).clone();
        if let Expr::Pow(inner_base, inner_exp) = base_data {
            // Check for even root safety: (x^2)^(1/2) -> |x|
            // If inner_exp is even integer and outer_exp is fractional with even denominator?
            // Or just check specific case (x^2)^(1/2).

            let is_even_int = |e: ExprId| -> bool {
                if let Expr::Number(n) = ctx.get(e) {
                    n.is_integer() && n.to_integer().is_even()
                } else {
                    false
                }
            };

            if is_even_int(inner_exp) && is_half(ctx, outer_exp) {
                // (x^(2k))^(1/2) -> |x|^k
                // If k=1, |x|.
                // new_exp = inner_exp * outer_exp = 2k * 1/2 = k.
                let prod_exp = mul_exp(ctx, inner_exp, outer_exp);
                // We need to wrap base in abs.
                let abs_base = ctx.add(Expr::Function("abs".to_string(), vec![inner_base]));
                let new_expr = ctx.add(Expr::Pow(abs_base, prod_exp));
                return Some(Rewrite {
                    new_expr,
                    description: "Power of power with even root: (x^2k)^(1/2) -> |x|^k".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }

            let prod_exp = mul_exp(ctx, inner_exp, outer_exp);
            let new_expr = ctx.add(Expr::Pow(inner_base, prod_exp));
            return Some(Rewrite {
                new_expr,
                description: "Multiply exponents".to_string(),
                before_local: None,
                after_local: None,
            });
        }
    }
    None
});

define_rule!(EvaluatePowerRule, "Evaluate Numeric Power", |ctx, expr| {
    let expr_data = ctx.get(expr).clone();
    if let Expr::Pow(base, exp) = expr_data {
        let base_data = ctx.get(base).clone();
        let exp_data = ctx.get(exp).clone();

        if let (Expr::Number(b), Expr::Number(e)) = (base_data, exp_data) {
            // Case 1: Integer Exponent
            if e.is_integer() {
                // We need to be careful with large exponents, but BigInt handles arbitrary size.
                // However, BigRational::pow takes i32.
                // If exponent is too large for i32, we might skip or use BigInt pow if base is integer.
                // num_rational 0.4 BigRational doesn't have a generic pow that takes BigInt.
                // It has `pow(i32)`.
                // Let's check if e fits in i32.
                if let Some(e_i32) = e
                    .to_integer()
                    .to_u32()
                    .map(|x| x as i32)
                    .or_else(|| e.to_integer().to_i32())
                {
                    // Check for 0^-n
                    if b.is_zero() && e_i32 < 0 {
                        let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
                        return Some(Rewrite {
                            new_expr: undef,
                            description: "Division by zero".to_string(),
                            before_local: None,
                            after_local: None,
                        });
                    }

                    // b^e_i32
                    // BigRational::pow takes i32.
                    let res = b.pow(e_i32);
                    let new_expr = ctx.add(Expr::Number(res));
                    return Some(Rewrite {
                        new_expr,
                        description: format!("Evaluate power: {}^{}", b, e),
                        before_local: None,
                        after_local: None,
                    });
                }
            }

            // Case 2: Fractional Exponent (Roots)
            // e = num / den.
            let numer = e.numer();
            let denom = e.denom();

            if let Some(n) = denom.to_u32() {
                let b_num = b.numer();
                let b_den = b.denom();

                // println!("Checking root: base={}/{}, n={}", b_num, b_den, n);
                let (out_n, in_n) = extract_root_factor(b_num, n);
                let (out_d, in_d) = extract_root_factor(b_den, n);
                // println!("Extracted: num=({},{}), den=({},{})", out_n, in_n, out_d, in_d);

                // If we extracted anything (outside parts are not 1)
                if !out_n.is_one() || !out_d.is_one() {
                    // b^(num/den) = ( (out_n^n * in_n) / (out_d^n * in_d) ) ^ (num/n)
                    //             = (out_n/out_d)^num * (in_n/in_d)^(num/n)

                    if let Some(pow_num) = numer.to_i32() {
                        let coeff_num = out_n.pow(pow_num as u32); // BigInt pow takes u32
                        let coeff_den = out_d.pow(pow_num as u32);
                        let coeff = BigRational::new(coeff_num, coeff_den);

                        let new_base_val = BigRational::new(in_n, in_d);

                        let coeff_expr = ctx.add(Expr::Number(coeff));

                        if new_base_val.is_one() {
                            // Perfect root
                            return Some(Rewrite {
                                new_expr: coeff_expr,
                                description: format!("Evaluate perfect root: {}^{}", b, e),
                                before_local: None,
                                after_local: None,
                            });
                        } else {
                            // Partial root
                            let new_base = ctx.add(Expr::Number(new_base_val));
                            let new_pow = ctx.add(Expr::Pow(new_base, exp)); // Keep original exponent for the remainder
                            let new_expr = ctx.add(Expr::Mul(coeff_expr, new_pow));
                            return Some(Rewrite {
                                new_expr,
                                description: format!("Simplify root: {}^{}", b, e),
                                before_local: None,
                                after_local: None,
                            });
                        }
                    }
                }
            }
        }
    }
    None
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{Context, DisplayExpr};

    #[test]
    fn test_product_power() {
        let mut ctx = Context::new();
        let rule = ProductPowerRule;

        // x^2 * x^3 -> x^(2+3)
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let x2 = ctx.add(Expr::Pow(x, two));
        let x3 = ctx.add(Expr::Pow(x, three));
        let expr = ctx.add(Expr::Mul(x2, x3));

        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "x^5"
        );

        // x * x -> x^2
        let expr2 = ctx.add(Expr::Mul(x, x));
        let rewrite2 = rule
            .apply(
                &mut ctx,
                expr2,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite2.new_expr
                }
            ),
            "x^2"
        );
    }

    #[test]
    fn test_power_power() {
        let mut ctx = Context::new();
        let rule = PowerPowerRule;

        // (x^2)^3 -> x^(2*3)
        let x = ctx.var("x");
        let two = ctx.num(2);
        let three = ctx.num(3);
        let x2 = ctx.add(Expr::Pow(x, two));
        let expr = ctx.add(Expr::Pow(x2, three));

        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "x^6"
        );
    }

    #[test]
    fn test_zero_one_power() {
        let mut ctx = Context::new();
        let rule = IdentityPowerRule;

        // x^0 -> 1
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let expr = ctx.add(Expr::Pow(x, zero));
        let rewrite = rule
            .apply(
                &mut ctx,
                expr,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite.new_expr
                }
            ),
            "1"
        );

        // x^1 -> x
        let one = ctx.num(1);
        let expr2 = ctx.add(Expr::Pow(x, one));
        let rewrite2 = rule
            .apply(
                &mut ctx,
                expr2,
                &crate::parent_context::ParentContext::root(),
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}",
                DisplayExpr {
                    context: &ctx,
                    id: rewrite2.new_expr
                }
            ),
            "x"
        );
    }
}

define_rule!(IdentityPowerRule, "Identity Power", |ctx, expr| {
    let expr_data = ctx.get(expr).clone();
    if let Expr::Pow(base, exp) = expr_data {
        // x^1 -> x
        if let Expr::Number(n) = ctx.get(exp) {
            if n.is_one() {
                return Some(Rewrite {
                    new_expr: base,
                    description: "x^1 -> x".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
            if n.is_zero() {
                // x^0 -> 1
                return Some(Rewrite {
                    new_expr: ctx.num(1),
                    description: "x^0 -> 1".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        }
        // 1^x -> 1
        if let Expr::Number(n) = ctx.get(base) {
            if n.is_one() {
                return Some(Rewrite {
                    new_expr: ctx.num(1),
                    description: "1^x -> 1".to_string(),
                    before_local: None,
                    after_local: None,
                });
            }
        }
    }
    None
});

define_rule!(PowerProductRule, "Power of a Product", |ctx, expr| {
    // Skip if expression is in canonical (elegant) form
    // e.g., ((x+1)*(x-1))^2 should NOT distribute -> stay as is
    if crate::canonical_forms::is_canonical_form(ctx, expr) {
        return None;
    }

    // (a * b)^n -> a^n * b^n
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        let base = *base;
        let exp = *exp;
        if let Expr::Mul(a, b) = ctx.get(base) {
            let a = *a;
            let b = *b;
            let a_pow = ctx.add(Expr::Pow(a, exp));
            let b_pow = ctx.add(Expr::Pow(b, exp));
            let new_expr = ctx.add(Expr::Mul(a_pow, b_pow));

            return Some(Rewrite {
                new_expr,
                description: "Distribute power over product".to_string(),
                before_local: None,
                after_local: None,
            });
        }
    }
    None
});

define_rule!(PowerQuotientRule, "Power of a Quotient", |ctx, expr| {
    // (a / b)^n -> a^n / b^n
    let expr_data = ctx.get(expr).clone();
    if let Expr::Pow(base, exp) = expr_data {
        let base_data = ctx.get(base).clone();
        if let Expr::Div(num, den) = base_data {
            // Distribute exponent
            let new_num = ctx.add(Expr::Pow(num, exp));
            let new_den = ctx.add(Expr::Pow(den, exp));
            let new_expr = ctx.add(Expr::Div(new_num, new_den));
            return Some(Rewrite {
                new_expr,
                description: "Distribute power over quotient".to_string(),
                before_local: None,
                after_local: None,
            });
        }
    }
    None
});

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(ProductSameExponentRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(EvaluatePowerRule));

    simplifier.add_rule(Box::new(IdentityPowerRule));
    simplifier.add_rule(Box::new(PowerProductRule));
    simplifier.add_rule(Box::new(PowerQuotientRule));
    simplifier.add_rule(Box::new(NegativeBasePowerRule));
}

define_rule!(NegativeBasePowerRule, "Negative Base Power", |ctx, expr| {
    // (-x)^n
    let expr_data = ctx.get(expr).clone();
    if let Expr::Pow(base, exp) = expr_data {
        let base_data = ctx.get(base).clone();
        if let Expr::Neg(inner) = base_data {
            // Check exponent parity
            if let Expr::Number(n) = ctx.get(exp) {
                if n.is_integer() {
                    if n.to_integer().is_even() {
                        // (-x)^even -> x^even
                        let new_expr = ctx.add(Expr::Pow(inner, exp));
                        return Some(Rewrite {
                            new_expr,
                            description: "(-x)^even -> x^even".to_string(),
                            before_local: None,
                            after_local: None,
                        });
                    } else {
                        // (-x)^odd -> -(x^odd)
                        let pow = ctx.add(Expr::Pow(inner, exp));
                        let new_expr = ctx.add(Expr::Neg(pow));
                        return Some(Rewrite {
                            new_expr,
                            description: "(-x)^odd -> -(x^odd)".to_string(),
                            before_local: None,
                            after_local: None,
                        });
                    }
                }
            }
        }
    }
    None
});

fn extract_root_factor(n: &BigInt, k: u32) -> (BigInt, BigInt) {
    if n.is_zero() {
        return (BigInt::zero(), BigInt::one());
    }
    if n.is_one() {
        return (BigInt::one(), BigInt::one());
    }

    let sign = if n.is_negative() { -1 } else { 1 };
    let mut n_abs = n.abs();

    let mut outside = BigInt::one();
    let mut inside = BigInt::one();

    // Trial division
    // Check 2
    let mut count = 0;
    while n_abs.is_even() {
        count += 1;
        n_abs /= 2;
    }
    if count > 0 {
        let out_exp = count / k;
        let in_exp = count % k;
        if out_exp > 0 {
            outside *= BigInt::from(2).pow(out_exp);
        }
        if in_exp > 0 {
            inside *= BigInt::from(2).pow(in_exp);
        }
    }

    let mut d = BigInt::from(3);
    while &d * &d <= n_abs {
        if (&n_abs % &d).is_zero() {
            let mut count = 0;
            while (&n_abs % &d).is_zero() {
                count += 1;
                n_abs /= &d;
            }
            let out_exp = count / k;
            let in_exp = count % k;
            if out_exp > 0 {
                outside *= d.pow(out_exp);
            }
            if in_exp > 0 {
                inside *= d.pow(in_exp);
            }
        }
        d += 2;
    }

    if n_abs > BigInt::one() {
        inside *= n_abs;
    }

    // Handle sign
    if sign == -1 {
        if !k.is_multiple_of(2) {
            outside = -outside;
        } else {
            inside = -inside;
        }
    }

    (outside, inside)
}
