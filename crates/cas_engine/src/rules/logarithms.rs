use crate::rule::{Rule, Rewrite};
use cas_ast::Expr;
use std::rc::Rc;
use num_traits::{Zero, One};

pub struct EvaluateLogRule;

impl Rule for EvaluateLogRule {
    fn name(&self) -> &str {
        "Evaluate Logarithms"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Function(name, args) = expr.as_ref() {
            if name == "log" && args.len() == 2 {
                let base = &args[0];
                let arg = &args[1];

                // 1. log(b, 1) = 0, log(b, 0) = -infinity, log(b, neg) = undefined
                if let Expr::Number(n) = arg.as_ref() {
                    if n.is_one() {
                        return Some(Rewrite {
                            new_expr: Rc::new(Expr::Number(num_rational::BigRational::zero())),
                            description: "log(b, 1) = 0".to_string(),
                        });
                    }
                    if n.is_zero() {
                        return Some(Rewrite {
                            new_expr: Expr::neg(Rc::new(Expr::Constant(cas_ast::Constant::Infinity))),
                            description: "log(b, 0) = -infinity".to_string(),
                        });
                    }
                    if *n < num_rational::BigRational::zero() {
                         return Some(Rewrite {
                            new_expr: Rc::new(Expr::Constant(cas_ast::Constant::Undefined)),
                            description: "log(b, neg) = undefined".to_string(),
                        });
                    }
                    
                    // Check if n is a power of base (if base is a number)
                    if let Expr::Number(b) = base.as_ref() {
                        // Simple check for integer powers for now
                        if b.is_integer() && n.is_integer() {
                            let b_int = b.to_integer();
                            let n_int = n.to_integer();
                            if b_int > num_bigint::BigInt::from(1) {
                                let mut temp = b_int.clone();
                                let mut power = 1;
                                while temp < n_int {
                                    temp = temp * &b_int;
                                    power += 1;
                                }
                                if temp == n_int {
                                    return Some(Rewrite {
                                        new_expr: Expr::num(power),
                                        description: format!("log({}, {}) = {}", b, n, power),
                                    });
                                }
                            }
                        }
                    }
                }

                // 2. log(b, b) = 1
                if base == arg {
                    return Some(Rewrite {
                        new_expr: Rc::new(Expr::Number(num_rational::BigRational::one())),
                        description: "log(b, b) = 1".to_string(),
                    });
                }

                // 3. log(b, b^x) = x
                if let Expr::Pow(p_base, p_exp) = arg.as_ref() {
                    if p_base == base {
                        return Some(Rewrite {
                            new_expr: p_exp.clone(),
                            description: "log(b, b^x) = x".to_string(),
                        });
                    }
                }

                // 4. Expansion: log(b, x^y) = y * log(b, x)
                // Note: This overlaps with rule 3 if x == b. Rule 3 is more specific/simpler, so it should match first.
                // This rule is good for canonicalization.
                if let Expr::Pow(p_base, p_exp) = arg.as_ref() {
                    return Some(Rewrite {
                        new_expr: Expr::mul(p_exp.clone(), Expr::log(base.clone(), p_base.clone())),
                        description: "log(b, x^y) = y * log(b, x)".to_string(),
                    });
                }

                // 5. Product: log(b, x*y) = log(b, x) + log(b, y)
                if let Expr::Mul(lhs, rhs) = arg.as_ref() {
                    return Some(Rewrite {
                        new_expr: Expr::add(
                            Expr::log(base.clone(), lhs.clone()),
                            Expr::log(base.clone(), rhs.clone())
                        ),
                        description: "log(b, x*y) = log(b, x) + log(b, y)".to_string(),
                    });
                }

                // 6. Quotient: log(b, x/y) = log(b, x) - log(b, y)
                if let Expr::Div(num, den) = arg.as_ref() {
                    return Some(Rewrite {
                        new_expr: Expr::sub(
                            Expr::log(base.clone(), num.clone()),
                            Expr::log(base.clone(), den.clone())
                        ),
                        description: "log(b, x/y) = log(b, x) - log(b, y)".to_string(),
                    });
                }
            }
        }
        None
    }
}

pub struct ExponentialLogRule;

impl Rule for ExponentialLogRule {
    fn name(&self) -> &str {
        "Exponential-Log Inverse"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Pow(base, exp) = expr.as_ref() {
            // Check if exponent is log(base, x)
            if let Expr::Function(name, args) = exp.as_ref() {
                if name == "log" && args.len() == 2 {
                    let log_base = &args[0];
                    let log_arg = &args[1];

                    if log_base == base {
                        return Some(Rewrite {
                            new_expr: log_arg.clone(),
                            description: "b^log(b, x) = x".to_string(),
                        });
                    }
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_log_one() {
        let rule = EvaluateLogRule;
        // log(x, 1) -> 0
        let expr = parse("log(x, 1)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "0");
    }

    #[test]
    fn test_log_base_base() {
        let rule = EvaluateLogRule;
        // log(x, x) -> 1
        let expr = parse("log(x, x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "1");
    }

    #[test]
    fn test_log_inverse() {
        let rule = EvaluateLogRule;
        // log(x, x^2) -> 2
        let expr = parse("log(x, x^2)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "2");
    }

    #[test]
    fn test_log_expansion() {
        let rule = EvaluateLogRule;
        // log(b, x^y) -> y * log(b, x)
        let expr = parse("log(2, x^3)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "3 * log(2, x)");
    }

    #[test]
    fn test_log_product() {
        let rule = EvaluateLogRule;
        // log(b, x*y) -> log(b, x) + log(b, y)
        let expr = parse("log(2, x * y)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        let res = format!("{}", rewrite.new_expr);
        assert!(res.contains("log(2, x)"));
        assert!(res.contains("log(2, y)"));
        assert!(res.contains("+"));
    }

    #[test]
    fn test_log_quotient() {
        let rule = EvaluateLogRule;
        // log(b, x/y) -> log(b, x) - log(b, y)
        let expr = parse("log(2, x / y)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        let res = format!("{}", rewrite.new_expr);
        assert!(res.contains("log(2, x)"));
        assert!(res.contains("log(2, y)"));
        assert!(res.contains("-"));
    }

    #[test]
    fn test_ln_e() {
        let rule = EvaluateLogRule;
        // ln(e) -> 1
        // Note: ln(e) parses to log(e, e)
        let expr = parse("ln(e)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "1");
    }
}
