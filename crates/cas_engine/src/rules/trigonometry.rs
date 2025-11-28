use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::Expr;
use std::rc::Rc;
use num_traits::{Zero, One};

define_rule!(
    EvaluateTrigRule,
    "Evaluate Trigonometric Functions",
    |expr| {
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
);

use crate::helpers::{extract_double_angle_arg, is_trig_pow};

define_rule!(
    PythagoreanIdentityRule,
    "Pythagorean Identity",
    |expr| {
        if let Expr::Add(lhs, rhs) = expr.as_ref() {
            // Check cos^2 + sin^2
            // Use is_trig_pow instead of is_trig_square
            // We need to extract the arg manually since is_trig_pow returns bool.
            // Or better: use get_trig_arg from helpers if we import it.
            use crate::helpers::get_trig_arg;
            
            if is_trig_pow(lhs, "cos", 2) && is_trig_pow(rhs, "sin", 2) {
                 if let (Some(arg_cos), Some(arg_sin)) = (get_trig_arg(lhs), get_trig_arg(rhs)) {
                     if arg_cos == arg_sin {
                         return Some(Rewrite {
                             new_expr: Rc::new(Expr::Number(num_rational::BigRational::one())),
                             description: "cos^2(x) + sin^2(x) = 1".to_string(),
                         });
                     }
                 }
            }
            
            // Check sin^2 + cos^2
            if is_trig_pow(lhs, "sin", 2) && is_trig_pow(rhs, "cos", 2) {
                 if let (Some(arg_sin), Some(arg_cos)) = (get_trig_arg(lhs), get_trig_arg(rhs)) {
                     if arg_sin == arg_cos {
                         return Some(Rewrite {
                             new_expr: Rc::new(Expr::Number(num_rational::BigRational::one())),
                             description: "sin^2(x) + cos^2(x) = 1".to_string(),
                         });
                     }
                 }
            }
        }
        None
    }
);

define_rule!(
    AngleIdentityRule,
    "Angle Sum/Diff Identity",
    |expr| {
        if let Expr::Function(name, args) = expr.as_ref() {
            if args.len() == 1 {
                let inner = &args[0];
                match name.as_str() {
                    "sin" => {
                        if let Expr::Add(lhs, rhs) = inner.as_ref() {
                            // sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
                            let term1 = Expr::mul(Expr::sin(lhs.clone()), Expr::cos(rhs.clone()));
                            let term2 = Expr::mul(Expr::cos(lhs.clone()), Expr::sin(rhs.clone()));
                            return Some(Rewrite {
                                new_expr: Expr::add(term1, term2),
                                description: "sin(a + b) -> sin(a)cos(b) + cos(a)sin(b)".to_string(),
                            });
                        } else if let Expr::Sub(lhs, rhs) = inner.as_ref() {
                            // sin(a - b) = sin(a)cos(b) - cos(a)sin(b)
                            let term1 = Expr::mul(Expr::sin(lhs.clone()), Expr::cos(rhs.clone()));
                            let term2 = Expr::mul(Expr::cos(lhs.clone()), Expr::sin(rhs.clone()));
                            return Some(Rewrite {
                                new_expr: Expr::sub(term1, term2),
                                description: "sin(a - b) -> sin(a)cos(b) - cos(a)sin(b)".to_string(),
                            });
                        }
                    },
                    "cos" => {
                        if let Expr::Add(lhs, rhs) = inner.as_ref() {
                            // cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
                            let term1 = Expr::mul(Expr::cos(lhs.clone()), Expr::cos(rhs.clone()));
                            let term2 = Expr::mul(Expr::sin(lhs.clone()), Expr::sin(rhs.clone()));
                            return Some(Rewrite {
                                new_expr: Expr::sub(term1, term2),
                                description: "cos(a + b) -> cos(a)cos(b) - sin(a)sin(b)".to_string(),
                            });
                        } else if let Expr::Sub(lhs, rhs) = inner.as_ref() {
                            // cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
                            let term1 = Expr::mul(Expr::cos(lhs.clone()), Expr::cos(rhs.clone()));
                            let term2 = Expr::mul(Expr::sin(lhs.clone()), Expr::sin(rhs.clone()));
                            return Some(Rewrite {
                                new_expr: Expr::add(term1, term2),
                                description: "cos(a - b) -> cos(a)cos(b) + sin(a)sin(b)".to_string(),
                            });
                        }
                    },
                    _ => {}
                }
            }
        }
        None
    }
);

define_rule!(
    TanToSinCosRule,
    "Tan to Sin/Cos",
    |expr| {
        if let Expr::Function(name, args) = expr.as_ref() {
            if name == "tan" && args.len() == 1 {
                // tan(x) -> sin(x) / cos(x)
                return Some(Rewrite {
                    new_expr: Expr::div(Expr::sin(args[0].clone()), Expr::cos(args[0].clone())),
                    description: "tan(x) -> sin(x)/cos(x)".to_string(),
                });
            }
        }
        None
    }
);

define_rule!(
    DoubleAngleRule,
    "Double Angle Identity",
    |expr| {
        if let Expr::Function(name, args) = expr.as_ref() {
            if args.len() == 1 {
                // Check if arg is 2*x or x*2
                // We need to match "2 * x"
                if let Some(inner_var) = extract_double_angle_arg(&args[0]) {
                    match name.as_str() {
                        "sin" => {
                            // sin(2x) -> 2sin(x)cos(x)
                            let new_expr = Expr::mul(
                                Expr::num(2),
                                Expr::mul(Expr::sin(inner_var.clone()), Expr::cos(inner_var.clone()))
                            );
                            return Some(Rewrite {
                                new_expr,
                                description: "sin(2x) -> 2sin(x)cos(x)".to_string(),
                            });
                        },
                        "cos" => {
                            // cos(2x) -> cos^2(x) - sin^2(x)
                            let cos2 = Expr::pow(Expr::cos(inner_var.clone()), Expr::num(2));
                            let sin2 = Expr::pow(Expr::sin(inner_var.clone()), Expr::num(2));
                            return Some(Rewrite {
                                new_expr: Expr::sub(cos2, sin2),
                                description: "cos(2x) -> cos^2(x) - sin^2(x)".to_string(),
                            });
                        },
                        _ => {}
                    }
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
    fn test_trig_identities() {
        let rule = AngleIdentityRule;
        
        // sin(x + y)
        let expr = parse("sin(x + y)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert!(format!("{}", rewrite.new_expr).contains("sin(x)"));
        
        // cos(x + y) -> cos(x)cos(y) - sin(x)sin(y)
        let expr = parse("cos(x + y)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        let res = format!("{}", rewrite.new_expr);
        assert!(res.contains("cos(x)"));
        assert!(res.contains("-"));
        
        // sin(x - y)
        let expr = parse("sin(x - y)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert!(format!("{}", rewrite.new_expr).contains("-"));
    }

    #[test]
    fn test_tan_to_sin_cos() {
        let rule = TanToSinCosRule;
        let expr = parse("tan(x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert_eq!(format!("{}", rewrite.new_expr), "sin(x) / cos(x)");
    }

    #[test]
    fn test_double_angle() {
        let rule = DoubleAngleRule;
        
        // sin(2x)
        let expr = parse("sin(2 * x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert!(format!("{}", rewrite.new_expr).contains("2 * sin(x) * cos(x)")); // Approx check
        
        // cos(2x)
        let expr = parse("cos(2 * x)").unwrap();
        let rewrite = rule.apply(&expr).unwrap();
        assert!(format!("{}", rewrite.new_expr).contains("cos(x)^2 - sin(x)^2"));
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

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(AngleIdentityRule));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    simplifier.add_rule(Box::new(DoubleAngleRule));
}
