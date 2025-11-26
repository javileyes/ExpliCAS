use crate::rule::{Rule, Rewrite};
use cas_ast::Expr;
use std::rc::Rc;
use num_traits::{Zero, One};

pub struct ProductPowerRule;

impl Rule for ProductPowerRule {
    fn name(&self) -> &str {
        "Product of Powers"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
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
}

pub struct PowerPowerRule;

impl Rule for PowerPowerRule {
    fn name(&self) -> &str {
        "Power of a Power"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
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
}

pub struct ZeroOnePowerRule;

impl Rule for ZeroOnePowerRule {
    fn name(&self) -> &str {
        "Zero/One Exponent"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
