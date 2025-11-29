use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::Expr;
use num_traits::{Zero, One};
use crate::helpers::{extract_double_angle_arg, is_trig_pow, get_trig_arg};
use crate::ordering::compare_expr;
use std::cmp::Ordering;

define_rule!(
    EvaluateTrigRule,
    "Evaluate Trigonometric Functions",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Function(name, args) = expr_data {
            if args.len() == 1 {
                let arg = args[0];
                
                // Case 1: Known Values (0)
                if let Expr::Number(n) = ctx.get(arg) {
                    if n.is_zero() {
                        match name.as_str() {
                            "sin" | "tan" | "arcsin" | "arctan" => {
                                let zero = ctx.num(0);
                                return Some(Rewrite {
                                    new_expr: zero,
                                    description: format!("{}(0) = 0", name),
                                });
                            },
                            "cos" => {
                                let one = ctx.num(1);
                                return Some(Rewrite {
                                    new_expr: one,
                                    description: "cos(0) = 1".to_string(),
                                });
                            },
                            "arccos" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let two = ctx.num(2);
                                let new_expr = ctx.add(Expr::Div(pi, two));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "arccos(0) = pi/2".to_string(),
                                });
                            },
                            _ => {}
                        }
                    } else if n.is_one() {
                        match name.as_str() {
                            "arcsin" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let two = ctx.num(2);
                                let new_expr = ctx.add(Expr::Div(pi, two));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "arcsin(1) = pi/2".to_string(),
                                });
                            },
                            "arccos" => {
                                let zero = ctx.num(0);
                                return Some(Rewrite {
                                    new_expr: zero,
                                    description: "arccos(1) = 0".to_string(),
                                });
                            },
                            "arctan" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let four = ctx.num(4);
                                let new_expr = ctx.add(Expr::Div(pi, four));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "arctan(1) = pi/4".to_string(),
                                });
                            },
                            _ => {}
                        }
                    } else if *n == num_rational::BigRational::new(1.into(), 2.into()) { // 1/2
                         match name.as_str() {
                            "arcsin" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let six = ctx.num(6);
                                let new_expr = ctx.add(Expr::Div(pi, six));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "arcsin(1/2) = pi/6".to_string(),
                                });
                            },
                            "arccos" => {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let three = ctx.num(3);
                                let new_expr = ctx.add(Expr::Div(pi, three));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "arccos(1/2) = pi/3".to_string(),
                                });
                            },
                            _ => {}
                         }
                    }
                }

                // Case 2: Known Values (pi)
                let arg_data = ctx.get(arg).clone();
                if let Expr::Constant(cas_ast::Constant::Pi) = arg_data {
                    match name.as_str() {
                        "sin" | "tan" => {
                            let zero = ctx.num(0);
                            return Some(Rewrite {
                                new_expr: zero,
                                description: format!("{}(pi) = 0", name),
                            });
                        },
                        "cos" => {
                            let neg_one = ctx.num(-1);
                            return Some(Rewrite {
                                new_expr: neg_one,
                                description: "cos(pi) = -1".to_string(),
                            });
                        },
                        _ => {}
                    }
                }
                
                // Case 3: Known Values (pi/2)
                let arg_data = ctx.get(arg).clone();
                if let Expr::Div(lhs, rhs) = arg_data {
                    let lhs_data = ctx.get(lhs);
                    let rhs_data = ctx.get(rhs);
                    if let (Expr::Constant(cas_ast::Constant::Pi), Expr::Number(n)) = (lhs_data, rhs_data) {
                        if *n == num_rational::BigRational::from_integer(2.into()) {
                            match name.as_str() {
                                "sin" => {
                                    let one = ctx.num(1);
                                    return Some(Rewrite {
                                        new_expr: one,
                                        description: "sin(pi/2) = 1".to_string(),
                                    });
                                },
                                "cos" => {
                                    let zero = ctx.num(0);
                                    return Some(Rewrite {
                                        new_expr: zero,
                                        description: "cos(pi/2) = 0".to_string(),
                                    });
                                },
                                "tan" => {
                                    let undefined = ctx.add(Expr::Constant(cas_ast::Constant::Undefined));
                                    return Some(Rewrite {
                                        new_expr: undefined,
                                        description: "tan(pi/2) = undefined".to_string(),
                                    });
                                },
                                _ => {}
                            }
                        }
                    }
                }

                // Case 2: Identities for negative arguments
                if let Expr::Neg(inner) = ctx.get(arg) {
                    match name.as_str() {
                        "sin" => {
                            let sin_inner = ctx.add(Expr::Function("sin".to_string(), vec![*inner]));
                            let new_expr = ctx.add(Expr::Neg(sin_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "sin(-x) = -sin(x)".to_string(),
                            });
                        },
                        "cos" => {
                            let new_expr = ctx.add(Expr::Function("cos".to_string(), vec![*inner]));
                            return Some(Rewrite {
                                new_expr,
                                description: "cos(-x) = cos(x)".to_string(),
                            });
                        },
                        "tan" => {
                            let tan_inner = ctx.add(Expr::Function("tan".to_string(), vec![*inner]));
                            let new_expr = ctx.add(Expr::Neg(tan_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "tan(-x) = -tan(x)".to_string(),
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

define_rule!(
    PythagoreanIdentityRule,
    "Pythagorean Identity",
    |ctx, expr| {
        if let Expr::Add(lhs, rhs) = ctx.get(expr) {
            // Check cos^2 + sin^2
            if is_trig_pow(ctx, *lhs, "cos", 2) && is_trig_pow(ctx, *rhs, "sin", 2) {
                 if let (Some(arg_cos), Some(arg_sin)) = (get_trig_arg(ctx, *lhs), get_trig_arg(ctx, *rhs)) {
                     if compare_expr(ctx, arg_cos, arg_sin) == Ordering::Equal {
                         let one = ctx.num(1);
                         return Some(Rewrite {
                             new_expr: one,
                             description: "cos^2(x) + sin^2(x) = 1".to_string(),
                         });
                     }
                 }
            }
            
            // Check sin^2 + cos^2
            if is_trig_pow(ctx, *lhs, "sin", 2) && is_trig_pow(ctx, *rhs, "cos", 2) {
                 if let (Some(arg_sin), Some(arg_cos)) = (get_trig_arg(ctx, *lhs), get_trig_arg(ctx, *rhs)) {
                     if compare_expr(ctx, arg_sin, arg_cos) == Ordering::Equal {
                         let one = ctx.num(1);
                         return Some(Rewrite {
                             new_expr: one,
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
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if args.len() == 1 {
                let inner = args[0];
                match name.as_str() {
                    "sin" => {
                        let inner_data = ctx.get(inner).clone();
                        if let Expr::Add(lhs, rhs) = inner_data {
                            // sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
                            let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![lhs]));
                            let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![rhs]));
                            let term1 = ctx.add(Expr::Mul(sin_a, cos_b));
                            
                            let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![lhs]));
                            let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![rhs]));
                            let term2 = ctx.add(Expr::Mul(cos_a, sin_b));
                            
                            let new_expr = ctx.add(Expr::Add(term1, term2));
                            return Some(Rewrite {
                                new_expr,
                                description: "sin(a + b) -> sin(a)cos(b) + cos(a)sin(b)".to_string(),
                            });
                        } else if let Expr::Sub(lhs, rhs) = inner_data {
                            // sin(a - b) = sin(a)cos(b) - cos(a)sin(b)
                            let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![lhs]));
                            let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![rhs]));
                            let term1 = ctx.add(Expr::Mul(sin_a, cos_b));
                            
                            let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![lhs]));
                            let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![rhs]));
                            let term2 = ctx.add(Expr::Mul(cos_a, sin_b));
                            
                            let new_expr = ctx.add(Expr::Sub(term1, term2));
                            return Some(Rewrite {
                                new_expr,
                                description: "sin(a - b) -> sin(a)cos(b) - cos(a)sin(b)".to_string(),
                            });
                        } else if let Expr::Div(num, den) = inner_data {
                            // sin((a + b) / c) -> sin(a/c + b/c) -> ...
                            let num_data = ctx.get(num).clone();
                            if let Expr::Add(lhs, rhs) = num_data {
                                let a = ctx.add(Expr::Div(lhs, den));
                                let b = ctx.add(Expr::Div(rhs, den));
                                
                                let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![a]));
                                let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![b]));
                                let term1 = ctx.add(Expr::Mul(sin_a, cos_b));
                                
                                let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![a]));
                                let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![b]));
                                let term2 = ctx.add(Expr::Mul(cos_a, sin_b));
                                
                                let new_expr = ctx.add(Expr::Add(term1, term2));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "sin((a + b)/c) -> sin(a/c)cos(b/c) + cos(a/c)sin(b/c)".to_string(),
                                });
                            }
                        }
                    },
                    "cos" => {
                        let inner_data = ctx.get(inner).clone();
                        if let Expr::Add(lhs, rhs) = inner_data {
                            // cos(a + b) = cos(a)cos(b) - sin(a)sin(b)
                            let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![lhs]));
                            let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![rhs]));
                            let term1 = ctx.add(Expr::Mul(cos_a, cos_b));
                            
                            let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![lhs]));
                            let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![rhs]));
                            let term2 = ctx.add(Expr::Mul(sin_a, sin_b));
                            
                            let new_expr = ctx.add(Expr::Sub(term1, term2));
                            return Some(Rewrite {
                                new_expr,
                                description: "cos(a + b) -> cos(a)cos(b) - sin(a)sin(b)".to_string(),
                            });
                        } else if let Expr::Sub(lhs, rhs) = inner_data {
                            // cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
                            let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![lhs]));
                            let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![rhs]));
                            let term1 = ctx.add(Expr::Mul(cos_a, cos_b));
                            
                            let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![lhs]));
                            let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![rhs]));
                            let term2 = ctx.add(Expr::Mul(sin_a, sin_b));
                            
                            let new_expr = ctx.add(Expr::Add(term1, term2));
                            return Some(Rewrite {
                                new_expr,
                                description: "cos(a - b) -> cos(a)cos(b) + sin(a)sin(b)".to_string(),
                            });
                        } else if let Expr::Div(num, den) = inner_data {
                            // cos((a + b) / c) -> cos(a/c + b/c) -> ...
                            let num_data = ctx.get(num).clone();
                            if let Expr::Add(lhs, rhs) = num_data {
                                let a = ctx.add(Expr::Div(lhs, den));
                                let b = ctx.add(Expr::Div(rhs, den));
                                
                                let cos_a = ctx.add(Expr::Function("cos".to_string(), vec![a]));
                                let cos_b = ctx.add(Expr::Function("cos".to_string(), vec![b]));
                                let term1 = ctx.add(Expr::Mul(cos_a, cos_b));
                                
                                let sin_a = ctx.add(Expr::Function("sin".to_string(), vec![a]));
                                let sin_b = ctx.add(Expr::Function("sin".to_string(), vec![b]));
                                let term2 = ctx.add(Expr::Mul(sin_a, sin_b));
                                
                                let new_expr = ctx.add(Expr::Sub(term1, term2));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "cos((a + b)/c) -> cos(a/c)cos(b/c) - sin(a/c)sin(b/c)".to_string(),
                                });
                            }
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
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Function(name, args) = expr_data {
            if name == "tan" && args.len() == 1 {
                // tan(x) -> sin(x) / cos(x)
                let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![args[0]]));
                let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![args[0]]));
                let new_expr = ctx.add(Expr::Div(sin_x, cos_x));
                return Some(Rewrite {
                    new_expr,
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
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if args.len() == 1 {
                // Check if arg is 2*x or x*2
                // We need to match "2 * x"
                if let Some(inner_var) = extract_double_angle_arg(ctx, args[0]) {
                    match name.as_str() {
                        "sin" => {
                            // sin(2x) -> 2sin(x)cos(x)
                            let two = ctx.num(2);
                            let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![inner_var]));
                            let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![inner_var]));
                            let sin_cos = ctx.add(Expr::Mul(sin_x, cos_x));
                            let new_expr = ctx.add(Expr::Mul(two, sin_cos));
                            return Some(Rewrite {
                                new_expr,
                                description: "sin(2x) -> 2sin(x)cos(x)".to_string(),
                            });
                        },
                        "cos" => {
                            // cos(2x) -> cos^2(x) - sin^2(x)
                            let two = ctx.num(2);
                            let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![inner_var]));
                            let cos2 = ctx.add(Expr::Pow(cos_x, two));
                            
                            let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![inner_var]));
                            let sin2 = ctx.add(Expr::Pow(sin_x, two));
                            
                            let new_expr = ctx.add(Expr::Sub(cos2, sin2));
                            return Some(Rewrite {
                                new_expr,
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
    use cas_ast::{DisplayExpr, Context};

    #[test]
    fn test_evaluate_trig_zero() {
        let mut ctx = Context::new();
        let rule = EvaluateTrigRule;

        // sin(0) -> 0
        let expr = parse("sin(0)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "0");

        // cos(0) -> 1
        let expr = parse("cos(0)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "1");

        // tan(0) -> 0
        let expr = parse("tan(0)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "0");
    }

    #[test]
    fn test_evaluate_trig_identities() {
        let mut ctx = Context::new();
        let rule = EvaluateTrigRule;

        // sin(-x) -> -sin(x)
        let expr = parse("sin(-x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "-sin(x)");

        // cos(-x) -> cos(x)
        let expr = parse("cos(-x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "cos(x)");

        // tan(-x) -> -tan(x)
        let expr = parse("tan(-x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "-tan(x)");
    }

    #[test]
    fn test_trig_identities() {
        let mut ctx = Context::new();
        let rule = AngleIdentityRule;
        
        // sin(x + y)
        let expr = parse("sin(x + y)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }).contains("sin(x)"));
        
        // cos(x + y) -> cos(x)cos(y) - sin(x)sin(y)
        let expr = parse("cos(x + y)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        let res = format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr });
        assert!(res.contains("cos(x)"));
        assert!(res.contains("-"));
        
        // sin(x - y)
        let expr = parse("sin(x - y)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }).contains("-"));
    }

    #[test]
    fn test_tan_to_sin_cos() {
        let mut ctx = Context::new();
        let rule = TanToSinCosRule;
        let expr = parse("tan(x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "sin(x) / cos(x)");
    }

    #[test]
    fn test_double_angle() {
        let mut ctx = Context::new();
        let rule = DoubleAngleRule;
        
        // sin(2x)
        let expr = parse("sin(2 * x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }).contains("2 * sin(x) * cos(x)")); // Approx check
        
        // cos(2x)
        let expr = parse("cos(2 * x)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }).contains("cos(x)^2 - sin(x)^2"));
    }

    #[test]
    fn test_evaluate_inverse_trig() {
        let mut ctx = Context::new();
        let rule = EvaluateTrigRule;

        // arcsin(0) -> 0
        let expr = parse("arcsin(0)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "0");

        // arccos(1) -> 0
        let expr = parse("arccos(1)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert_eq!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }), "0");

        // arcsin(1) -> pi/2
        // Note: pi/2 might be formatted as "pi / 2" or similar depending on Display impl
        let expr = parse("arcsin(1)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }).contains("pi"));
        assert!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }).contains("2"));

        // arccos(0) -> pi/2
        let expr = parse("arccos(0)", &mut ctx).unwrap();
        let rewrite = rule.apply(&mut ctx, expr).unwrap();
        assert!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }).contains("pi"));
        assert!(format!("{}", DisplayExpr { context: &ctx, id: rewrite.new_expr }).contains("2"));
    }
}

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(AngleIdentityRule));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    simplifier.add_rule(Box::new(DoubleAngleRule));
}
