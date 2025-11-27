use crate::rule::{Rule, Rewrite};
use cas_ast::Expr;
use std::rc::Rc;
use num_traits::{Zero, One};

pub struct EvaluateTrigRule;

impl Rule for EvaluateTrigRule {
    fn name(&self) -> &str {
        "Evaluate Trigonometric Functions"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Function(name, args) = expr.as_ref() {
            if args.len() == 1 {
                let arg = &args[0];
                
                // Case 1: Known Values (0)
                if let Expr::Number(n) = arg.as_ref() {
                    if n.is_zero() {
                        match name.as_str() {
                            "sin" | "tan" | "arcsin" | "arctan" => return Some(Rewrite {
                                new_expr: Rc::new(Expr::Number(n.clone())), // 0
                                description: format!("{}(0) = 0", name),
                            }),
                            "cos" => return Some(Rewrite {
                                new_expr: Rc::new(Expr::Number(num_rational::BigRational::one())), // 1
                                description: "cos(0) = 1".to_string(),
                            }),
                            "arccos" => return Some(Rewrite {
                                new_expr: Expr::div(Expr::pi(), Expr::num(2)), // pi/2
                                description: "arccos(0) = pi/2".to_string(),
                            }),
                            _ => {}
                        }
                    } else if n.is_one() {
                        match name.as_str() {
                            "arcsin" => return Some(Rewrite {
                                new_expr: Expr::div(Expr::pi(), Expr::num(2)), // pi/2
                                description: "arcsin(1) = pi/2".to_string(),
                            }),
                            "arccos" => return Some(Rewrite {
                                new_expr: Expr::num(0), // 0
                                description: "arccos(1) = 0".to_string(),
                            }),
                            "arctan" => return Some(Rewrite {
                                new_expr: Expr::div(Expr::pi(), Expr::num(4)), // pi/4
                                description: "arctan(1) = pi/4".to_string(),
                            }),
                            _ => {}
                        }
                    } else if *n == num_rational::BigRational::new(1.into(), 2.into()) { // 1/2
                         match name.as_str() {
                            "arcsin" => return Some(Rewrite {
                                new_expr: Expr::div(Expr::pi(), Expr::num(6)), // pi/6
                                description: "arcsin(1/2) = pi/6".to_string(),
                            }),
                            "arccos" => return Some(Rewrite {
                                new_expr: Expr::div(Expr::pi(), Expr::num(3)), // pi/3
                                description: "arccos(1/2) = pi/3".to_string(),
                            }),
                            _ => {}
                         }
                    }
                }

                // Case 2: Known Values (pi)
                if let Expr::Constant(cas_ast::Constant::Pi) = arg.as_ref() {
                    match name.as_str() {
                        "sin" | "tan" => return Some(Rewrite {
                            new_expr: Rc::new(Expr::Number(num_rational::BigRational::zero())), // 0
                            description: format!("{}(pi) = 0", name),
                        }),
                        "cos" => return Some(Rewrite {
                            new_expr: Rc::new(Expr::Number(-num_rational::BigRational::one())), // -1
                            description: "cos(pi) = -1".to_string(),
                        }),
                        _ => {}
                    }
                }

                // Case 2: Identities for negative arguments
                if let Expr::Neg(inner) = arg.as_ref() {
                    match name.as_str() {
                        "sin" => return Some(Rewrite {
                            new_expr: Expr::neg(Expr::sin(inner.clone())),
                            description: "sin(-x) = -sin(x)".to_string(),
                        }),
                        "cos" => return Some(Rewrite {
                            new_expr: Expr::cos(inner.clone()),
                            description: "cos(-x) = cos(x)".to_string(),
                        }),
                        "tan" => return Some(Rewrite {
                            new_expr: Expr::neg(Expr::tan(inner.clone())),
                            description: "tan(-x) = -tan(x)".to_string(),
                        }),
                        _ => {}
                    }
                }
            }
        }
        None
    }
}

pub struct PythagoreanIdentityRule;

impl Rule for PythagoreanIdentityRule {
    fn name(&self) -> &str {
        "Pythagorean Identity"
    }

    fn apply(&self, expr: &Rc<Expr>) -> Option<Rewrite> {
        if let Expr::Add(lhs, rhs) = expr.as_ref() {
            // Check for sin^2(x) + cos^2(x)
            if let (Some(arg_sin), Some(arg_cos)) = (is_trig_square(lhs, "sin"), is_trig_square(rhs, "cos")) {
                if arg_sin == arg_cos {
                    return Some(Rewrite {
                        new_expr: Rc::new(Expr::Number(num_rational::BigRational::one())),
                        description: "sin^2(x) + cos^2(x) = 1".to_string(),
                    });
                }
            }
            
            // Check for cos^2(x) + sin^2(x)
            if let (Some(arg_cos), Some(arg_sin)) = (is_trig_square(lhs, "cos"), is_trig_square(rhs, "sin")) {
                if arg_sin == arg_cos {
                    return Some(Rewrite {
                        new_expr: Rc::new(Expr::Number(num_rational::BigRational::one())),
                        description: "cos^2(x) + sin^2(x) = 1".to_string(),
                    });
                }
            }
        }
        None
    }
}

// Helper to check if expr is name(arg)^2 and return arg
fn is_trig_square<'a>(expr: &'a Expr, name: &str) -> Option<&'a Rc<Expr>> {
    if let Expr::Pow(base, exp) = expr {
        if let Expr::Number(n) = exp.as_ref() {
            if n.is_integer() && n.to_integer() == 2.into() {
                if let Expr::Function(func_name, args) = base.as_ref() {
                    if func_name == name && args.len() == 1 {
                        return Some(&args[0]);
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_evaluate_trig_zero() {
        let rule = EvaluateTrigRule;

        // sin(0) -> 0
        let expr = parse("sin(0)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "0");

        // cos(0) -> 1
        let expr = parse("cos(0)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "1");

        // tan(0) -> 0
        let expr = parse("tan(0)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "0");
    }

    #[test]
    fn test_evaluate_trig_identities() {
        let rule = EvaluateTrigRule;

        // sin(-x) -> -sin(x)
        let expr = parse("sin(-x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "-sin(x)");

        // cos(-x) -> cos(x)
        let expr = parse("cos(-x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "cos(x)");

        // tan(-x) -> -tan(x)
        let expr = parse("tan(-x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "-tan(x)");
    }

    #[test]
    fn test_evaluate_inverse_trig() {
        let rule = EvaluateTrigRule;

        // arcsin(0) -> 0
        let expr = parse("arcsin(0)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "0");

        // arccos(1) -> 0
        let expr = parse("arccos(1)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "0");

        // arcsin(1) -> pi/2
        // Note: pi/2 might be formatted as "pi / 2" or similar depending on Display impl
        let expr = parse("arcsin(1)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert!(format!("{}", rewrite.new_expr).contains("pi"));
        assert!(format!("{}", rewrite.new_expr).contains("2"));

        // arccos(0) -> pi/2
        let expr = parse("arccos(0)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert!(format!("{}", rewrite.new_expr).contains("pi"));
        assert!(format!("{}", rewrite.new_expr).contains("2"));
    }
}
