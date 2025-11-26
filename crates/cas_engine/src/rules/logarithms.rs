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

                // 1. log(b, 1) = 0
                if let Expr::Number(n) = arg.as_ref() {
                    if n.is_one() {
                        return Some(Rewrite {
                            new_expr: Rc::new(Expr::Number(num_rational::BigRational::zero())),
                            description: "log(b, 1) = 0".to_string(),
                        });
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
}
