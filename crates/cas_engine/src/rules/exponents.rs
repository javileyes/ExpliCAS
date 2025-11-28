use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::Expr;
use std::rc::Rc;
use num_traits::{Zero, One, ToPrimitive};
use num_rational::BigRational;

define_rule!(
    ProductPowerRule,
    "Product of Powers",
    |expr| {
        // x^a * x^b -> x^(a+b)
        if let Expr::Mul(lhs, rhs) = expr.as_ref() {
            // Case 1: Both are powers with same base: x^a * x^b
            if let (Expr::Pow(base1, exp1), Expr::Pow(base2, exp2)) = (lhs.as_ref(), rhs.as_ref()) {
                if base1 == base2 {
                    return Some(Rewrite {
                        new_expr: Expr::pow(base1.clone(), Expr::add(exp1.clone(), exp2.clone())),
                        description: "Combine powers with same base".to_string(),
                    });
                }
            }
            // Case 2: One is power, one is base: x^a * x -> x^(a+1)
            // Left is power
            if let Expr::Pow(base1, exp1) = lhs.as_ref() {
                if base1.as_ref() == rhs.as_ref() {
                    return Some(Rewrite {
                        new_expr: Expr::pow(base1.clone(), Expr::add(exp1.clone(), Expr::num(1))),
                        description: "Combine power and base".to_string(),
                    });
                }
            }
            // Right is power
            if let Expr::Pow(base2, exp2) = rhs.as_ref() {
                if base2.as_ref() == lhs.as_ref() {
                    return Some(Rewrite {
                        new_expr: Expr::pow(base2.clone(), Expr::add(Expr::num(1), exp2.clone())),
                        description: "Combine base and power".to_string(),
                    });
                }
            }
            // Case 3: Both are same base (implicit power 1): x * x -> x^2
            if lhs == rhs {
                return Some(Rewrite {
                    new_expr: Expr::pow(lhs.clone(), Expr::num(2)),
                    description: "Multiply identical terms".to_string(),
                });
            }
        }
        None
    }
);

define_rule!(
    PowerPowerRule,
    "Power of a Power",
    |expr| {
        // (x^a)^b -> x^(a*b)
        if let Expr::Pow(base, outer_exp) = expr.as_ref() {
            if let Expr::Pow(inner_base, inner_exp) = base.as_ref() {
                return Some(Rewrite {
                    new_expr: Expr::pow(inner_base.clone(), Expr::mul(inner_exp.clone(), outer_exp.clone())),
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
    |expr| {
        if let Expr::Pow(base, exp) = expr.as_ref() {
            if let (Expr::Number(b), Expr::Number(e)) = (base.as_ref(), exp.as_ref()) {
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
                             return Some(Rewrite {
                                 new_expr: Rc::new(Expr::Constant(cas_ast::Constant::Undefined)),
                                 description: "Division by zero".to_string(),
                             });
                         }

                         // b^e_i32
                         // BigRational::pow takes i32.
                         let res = b.pow(e_i32);
                         return Some(Rewrite {
                             new_expr: Rc::new(Expr::Number(res)),
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
                             return Some(Rewrite {
                                 new_expr: Rc::new(Expr::Number(res)),
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
    |expr| {
        if let Expr::Pow(base, exp) = expr.as_ref() {
            // x^0 -> 1
            if let Expr::Number(n) = exp.as_ref() {
                if n.is_zero() {
                    return Some(Rewrite {
                        new_expr: Expr::num(1),
                        description: "Anything to the power of 0 is 1".to_string(),
                    });
                }
                if n.is_one() {
                    return Some(Rewrite {
                        new_expr: base.clone(),
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

    #[test]
    fn test_product_power() {
        let rule = ProductPowerRule;
        
        // x^2 * x^3 -> x^(2+3)
        let expr = Expr::mul(
            Expr::pow(Expr::var("x"), Expr::num(2)),
            Expr::pow(Expr::var("x"), Expr::num(3))
        );
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "x^(2 + 3)");

        // x * x -> x^2
        let expr2 = Expr::mul(Expr::var("x"), Expr::var("x"));
        let rewrite2 = rule.apply(&expr2).unwrap();
        assert_eq!(format!("{}", rewrite2.new_expr), "x^2");
    }

    #[test]
    fn test_power_power() {
        let rule = PowerPowerRule;
        
        // (x^2)^3 -> x^(2*3)
        let expr = Expr::pow(
            Expr::pow(Expr::var("x"), Expr::num(2)),
            Expr::num(3)
        );
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "x^(2 * 3)");
    }

    #[test]
    fn test_zero_one_power() {
        let rule = ZeroOnePowerRule;
        
        // x^0 -> 1
        let expr = Expr::pow(Expr::var("x"), Expr::num(0));
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "1");

        // x^1 -> x
        let expr2 = Expr::pow(Expr::var("x"), Expr::num(1));
        let rewrite2 = rule.apply(&expr2).unwrap();
        assert_eq!(format!("{}", rewrite2.new_expr), "x");
    }
}
