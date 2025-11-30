use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::{Expr, ExprId};
use num_traits::{Zero, One, ToPrimitive};
use num_integer::Integer;
use num_rational::BigRational;
use crate::ordering::compare_expr;
use std::cmp::Ordering;

define_rule!(
    ProductPowerRule,
    "Product of Powers",
    |ctx, expr| {
        // x^a * x^b -> x^(a+b)
        let expr_data = ctx.get(expr).clone();
        if let Expr::Mul(lhs, rhs) = expr_data {

            
            let lhs_data = ctx.get(lhs).clone();
            let rhs_data = ctx.get(rhs).clone();
            

            // Case 1: Both are powers with same base: x^a * x^b
            if let (Expr::Pow(base1, exp1), Expr::Pow(base2, exp2)) = (&lhs_data, &rhs_data) {
                if compare_expr(ctx, *base1, *base2) == Ordering::Equal {
                    let sum_exp = ctx.add(Expr::Add(*exp1, *exp2));
                    let new_expr = ctx.add(Expr::Pow(*base1, sum_exp));
                    return Some(Rewrite {
                        new_expr,
                        description: "Combine powers with same base".to_string(),
                    });
                }
            }
            // Case 2: One is power, one is base: x^a * x -> x^(a+1)
            // Left is power
            if let Expr::Pow(base1, exp1) = &lhs_data {
                if compare_expr(ctx, *base1, rhs) == Ordering::Equal {
                    let one = ctx.num(1);
                    let sum_exp = ctx.add(Expr::Add(*exp1, one));
                    let new_expr = ctx.add(Expr::Pow(*base1, sum_exp));
                    return Some(Rewrite {
                        new_expr,
                        description: "Combine power and base".to_string(),
                    });
                }
            }
            // Right is power
            if let Expr::Pow(base2, exp2) = &rhs_data {
                if compare_expr(ctx, *base2, lhs) == Ordering::Equal {
                    let one = ctx.num(1);
                    let sum_exp = ctx.add(Expr::Add(one, *exp2));
                    let new_expr = ctx.add(Expr::Pow(*base2, sum_exp));
                    return Some(Rewrite {
                        new_expr,
                        description: "Combine base and power".to_string(),
                    });
                }
            }
            // Case 3: Both are same base (implicit power 1): x * x -> x^2
            if compare_expr(ctx, lhs, rhs) == Ordering::Equal {
                let two = ctx.num(2);
                let new_expr = ctx.add(Expr::Pow(lhs, two));
                return Some(Rewrite {
                    new_expr,
                    description: "Multiply identical terms".to_string(),
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
                    });
                }
                
                // x^a * (x^b * y) -> x^(a+b) * y
                let lhs_pow = if let Expr::Pow(b, e) = &lhs_data { Some((*b, *e)) } else { None };
                let rhs_pow = if let Expr::Pow(b, e) = ctx.get(rl) { Some((*b, *e)) } else { None };

                if let (Some((base1, exp1)), Some((base2, exp2))) = (lhs_pow, rhs_pow) {
                    if compare_expr(ctx, base1, base2) == Ordering::Equal {
                        let sum_exp = ctx.add(Expr::Add(exp1, exp2));
                        let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                        let new_expr = ctx.add(Expr::Mul(new_pow, rr));
                        return Some(Rewrite {
                            new_expr,
                            description: "Combine nested powers".to_string(),
                        });
                    }
                }
                
                // x * (x^a * y) -> x^(a+1) * y
                if let Some((base2, exp2)) = rhs_pow {
                    if compare_expr(ctx, lhs, base2) == Ordering::Equal {
                        let one = ctx.num(1);
                        let sum_exp = ctx.add(Expr::Add(exp2, one));
                        let new_pow = ctx.add(Expr::Pow(base2, sum_exp));
                        let new_expr = ctx.add(Expr::Mul(new_pow, rr));
                        return Some(Rewrite {
                            new_expr,
                            description: "Combine base and nested power".to_string(),
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
                        });
                    }
                }
                
                // (c * x^a) * x^b -> c * x^(a+b)
                // Check if lhs is Mul(c, x^a)
                if let Expr::Mul(ll, lr) = lhs_data {
                    // Check if ll is number (coefficient)
                    if let Expr::Number(_) = ctx.get(ll) {
                        // lr is x^a ?
                        let lr_pow = if let Expr::Pow(b, e) = ctx.get(lr) { Some((*b, *e)) } else { None };
                        
                        // Check rhs is x^b
                        let rhs_pow = if let Expr::Pow(b, e) = &rhs_data { Some((*b, *e)) } else { None };
                        
                        if let (Some((base1, exp1)), Some((base2, exp2))) = (lr_pow, rhs_pow) {
                             if compare_expr(ctx, base1, base2) == Ordering::Equal {
                                 let sum_exp = ctx.add(Expr::Add(exp1, exp2));
                                 let new_pow = ctx.add(Expr::Pow(base1, sum_exp));
                                 let new_expr = ctx.add(Expr::Mul(ll, new_pow));
                                 return Some(Rewrite {
                                     new_expr,
                                     description: "Combine coeff-power and power".to_string(),
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
                         let rhs_pow = if let Expr::Pow(b, e) = &rhs_data { Some((*b, *e)) } else { None };
                         
                         if let Some((base2, exp2)) = rhs_pow {
                             if compare_expr(ctx, lr, base2) == Ordering::Equal {
                                 let one = ctx.num(1);
                                 let sum_exp = ctx.add(Expr::Add(exp2, one));
                                 let new_pow = ctx.add(Expr::Pow(base2, sum_exp));
                                 let new_expr = ctx.add(Expr::Mul(ll, new_pow));
                                 return Some(Rewrite {
                                     new_expr,
                                     description: "Combine coeff-base and power".to_string(),
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
                        });
                    }
                }
            }
        }
        None
    }
);

define_rule!(
    PowerPowerRule,
    "Power of a Power",
    |ctx, expr| {
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
                
                let is_half = |e: ExprId| -> bool {
                    if let Expr::Number(n) = ctx.get(e) {
                        *n.numer() == 1.into() && *n.denom() == 2.into()
                    } else {
                        false
                    }
                };

                if is_even_int(inner_exp) && is_half(outer_exp) {
                    // (x^(2k))^(1/2) -> |x|^k
                    // If k=1, |x|.
                    // new_exp = inner_exp * outer_exp = 2k * 1/2 = k.
                    let prod_exp = ctx.add(Expr::Mul(inner_exp, outer_exp));
                    // We need to wrap base in abs.
                    let abs_base = ctx.add(Expr::Function("abs".to_string(), vec![inner_base]));
                    let new_expr = ctx.add(Expr::Pow(abs_base, prod_exp));
                    return Some(Rewrite {
                        new_expr,
                        description: "Power of power with even root: (x^2k)^(1/2) -> |x|^k".to_string(),
                    });
                }

                let prod_exp = ctx.add(Expr::Mul(inner_exp, outer_exp));
                let new_expr = ctx.add(Expr::Pow(inner_base, prod_exp));
                return Some(Rewrite {
                    new_expr,
                    description: "Multiply exponents".to_string(),
                });
            }
        }
        None
    }
);

define_rule!(
    EvaluatePowerRule,
    "Evaluate Numeric Power",
    |ctx, expr| {
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
                    if let Some(e_i32) = e.to_integer().to_u32().map(|x| x as i32).or_else(|| e.to_integer().to_i32()) {
                         // Check for 0^-n
                         if b.is_zero() && e_i32 < 0 {
                             let undef = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
                             return Some(Rewrite {
                                 new_expr: undef,
                                 description: "Division by zero".to_string(),
                             });
                         }

                         // b^e_i32
                         // BigRational::pow takes i32.
                         let res = b.pow(e_i32);
                         let new_expr = ctx.add(Expr::Number(res));
                         return Some(Rewrite {
                             new_expr,
                             description: format!("Evaluate power: {}^{}", b, e),
                         });
                    }
                }
                
                // Case 2: Fractional Exponent (Roots)
                // e = num / den.
                // We want to check if b is a perfect root.
                // b^(num/den) = (b^(1/den))^num
                // Let's try to find nth_root where n = den.
                let numer = e.numer();
                let denom = e.denom();
                
                // Only handle if numerator is 1 for now (standard roots), or maybe simple fractions.
                // If we have 27^(1/3), numer=1, denom=3.
                // We check if b is a perfect 3rd root.
                // BigRational doesn't have nth_root. We need to work with numerator and denominator of base separately.
                // base = b_num / b_den.
                // root = nth_root(b_num) / nth_root(b_den).
                
                // We need to convert denom (BigInt) to u32 for nth_root.
                if let Some(n) = denom.to_u32() {
                    let b_num = b.numer();
                    let b_den = b.denom();
                    
                    let root_num = b_num.nth_root(n);
                    let root_den = b_den.nth_root(n);
                    
                    // Check if perfect root
                    if root_num.pow(n) == *b_num && root_den.pow(n) == *b_den {
                        // It is a perfect root!
                        // So b^(1/n) = root_num / root_den.
                        let root = BigRational::new(root_num, root_den);
                        
                        // Now raise to numerator power: (b^(1/n))^numer
                        if let Some(pow_num) = numer.to_i32() {
                             let res = root.pow(pow_num);
                             let new_expr = ctx.add(Expr::Number(res));
                             return Some(Rewrite {
                                 new_expr,
                                 description: format!("Evaluate perfect root: {}^{}", b, e),
                             });
                        }
                    }
                }
            }
        }
        None
    }
);

define_rule!(
    ZeroOnePowerRule,
    "Zero/One Exponent",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
            // x^0 -> 1
            let exp_data = ctx.get(exp).clone();
            if let Expr::Number(n) = exp_data {
                if n.is_zero() {
                    let one = ctx.num(1);
                    return Some(Rewrite {
                        new_expr: one,
                        description: "Anything to the power of 0 is 1".to_string(),
                    });
                }
                if n.is_one() {
                    return Some(Rewrite {
                        new_expr: base,
                        description: "Exponent 1 is identity".to_string(),
                    });
                }
            }
        }
        None
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::{DisplayExpr, Context};

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

        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "x^(2 + 3)");

        // x * x -> x^2
        let expr2 = ctx.add(Expr::Mul(x, x));
        let rewrite2 = rule.apply(&mut ctx, expr2).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite2.new_expr }), "x^2");
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

        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "x^(2 * 3)");
    }

    #[test]
    fn test_zero_one_power() {
        let mut ctx = Context::new();
        let rule = ZeroOnePowerRule;
        
        // x^0 -> 1
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let expr = ctx.add(Expr::Pow(x, zero));
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "1");

        // x^1 -> x
        let one = ctx.num(1);
        let expr2 = ctx.add(Expr::Pow(x, one));
        let rewrite2 = rule.apply(&mut ctx, expr2).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite2.new_expr }), "x");
    }
}

define_rule!(
    IdentityPowerRule,
    "Identity Power",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
            // x^1 -> x
            if let Expr::Number(n) = ctx.get(exp) {
                if n.is_one() {
                    return Some(Rewrite {
                        new_expr: base,
                        description: "x^1 -> x".to_string(),
                    });
                }
                if n.is_zero() {
                    // x^0 -> 1
                    return Some(Rewrite {
                        new_expr: ctx.num(1),
                        description: "x^0 -> 1".to_string(),
                    });
                }
            }
            // 1^x -> 1
            if let Expr::Number(n) = ctx.get(base) {
                if n.is_one() {
                    return Some(Rewrite {
                        new_expr: ctx.num(1),
                        description: "1^x -> 1".to_string(),
                    });
                }
            }
        }
        None
    }
);

define_rule!(
    PowerProductRule,
    "Power of a Product",
    |ctx, expr| {
        // (a * b)^n -> a^n * b^n
        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
            let base_data = ctx.get(base).clone();
            if let Expr::Mul(lhs, rhs) = base_data {
                // Distribute exponent
                let new_lhs = ctx.add(Expr::Pow(lhs, exp));
                let new_rhs = ctx.add(Expr::Pow(rhs, exp));
                let new_expr = ctx.add(Expr::Mul(new_lhs, new_rhs));
                return Some(Rewrite {
                    new_expr,
                    description: "Distribute power over product".to_string(),
                });
            }
        }
        None
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(ProductPowerRule));
    simplifier.add_rule(Box::new(PowerPowerRule));
    simplifier.add_rule(Box::new(EvaluatePowerRule));
    simplifier.add_rule(Box::new(ZeroOnePowerRule));
    simplifier.add_rule(Box::new(IdentityPowerRule));
    simplifier.add_rule(Box::new(PowerProductRule));
}
