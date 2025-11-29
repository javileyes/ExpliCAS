use crate::rule::Rewrite;
use crate::define_rule;
use cas_ast::{Expr, ExprId};
use num_traits::{Zero, One};
use crate::helpers::extract_double_angle_arg;
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
        // Look for sin(x)^2 + cos(x)^2 = 1
        // Or a*sin(x)^2 + a*cos(x)^2 = a
        
        // This requires scanning an Add expression
        let expr_data = ctx.get(expr).clone();
        if let Expr::Add(_, _) = expr_data {
             // Flatten add
             let mut terms = Vec::new();
             crate::helpers::flatten_add(ctx, expr, &mut terms);
             
             // Step 1: Analyze all terms and extract relevant info
             // Store: (TermIndex, CoeffVal, FuncName, Arg)
             struct TrigTerm {
                 index: usize,
                 coeff_val: num_rational::BigRational,
                 func_name: String,
                 arg: ExprId,
             }
             
             let mut trig_terms = Vec::new();
             
             for (i, &term) in terms.iter().enumerate() {
                 let (c_val, v) = crate::helpers::get_parts(ctx, term);
                 // Check v is sin(arg)^2 or cos(arg)^2
                 if let Expr::Pow(base, exp) = ctx.get(v) {
                     if let Expr::Number(n) = ctx.get(*exp) {
                         if *n == num_rational::BigRational::from_integer(2.into()) {
                             if let Expr::Function(name, args) = ctx.get(*base) {
                                 if (name == "sin" || name == "cos") && args.len() == 1 {
                                     trig_terms.push(TrigTerm {
                                         index: i,
                                         coeff_val: c_val,
                                         func_name: name.clone(),
                                         arg: args[0],
                                     });
                                 }
                             }
                         }
                     }
                 }
             }
             
             // Step 2: Find pairs
             for i in 0..trig_terms.len() {
                 let t1 = &trig_terms[i];
                 for j in (i+1)..trig_terms.len() {
                     let t2 = &trig_terms[j];
                     
                     if t1.func_name != t2.func_name && t1.coeff_val == t2.coeff_val {
                         // Check args equality
                         if compare_expr(ctx, t1.arg, t2.arg) == Ordering::Equal {
                             // Found match!
                             // a*sin^2 + a*cos^2 = a
                             
                             // Construct new expression
                             let mut new_terms = Vec::new();
                             for k in 0..terms.len() {
                                 if k != t1.index && k != t2.index {
                                     new_terms.push(terms[k]);
                                 }
                             }
                             
                             // Add 'a' (the coefficient)
                             let a = ctx.add(Expr::Number(t1.coeff_val.clone()));
                             new_terms.push(a);
                             
                             if new_terms.is_empty() {
                                 return Some(Rewrite {
                                     new_expr: ctx.num(0),
                                     description: "Pythagorean Identity (empty)".to_string(),
                                 });
                             }
                             
                             let mut new_expr = new_terms[0];
                             for k in 1..new_terms.len() {
                                 new_expr = ctx.add(Expr::Add(new_expr, new_terms[k]));
                             }
                             
                             return Some(Rewrite {
                                 new_expr,
                                 description: "sin^2 + cos^2 = 1".to_string(),
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

define_rule!(
    RecursiveTrigExpansionRule,
    "Recursive Trig Expansion",
    |ctx, expr| {
        let expr_data = ctx.get(expr).clone();
        if let Expr::Function(name, args) = expr_data {
            if args.len() == 1 && (name == "sin" || name == "cos") {
                // Check for n * x where n is integer > 2
                let inner = args[0];
                let inner_data = ctx.get(inner).clone();
                
                let (n_val, x_val) = if let Expr::Mul(l, r) = inner_data {
                    if let Expr::Number(n) = ctx.get(l) {
                        if n.is_integer() {
                            (n.to_integer(), r)
                        } else {
                            return None;
                        }
                    } else if let Expr::Number(n) = ctx.get(r) {
                        if n.is_integer() {
                            (n.to_integer(), l)
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None;
                };

                if n_val > num_bigint::BigInt::from(2) {
                    // Rewrite sin(nx) -> sin((n-1)x + x)
                    
                    let n_minus_1 = n_val.clone() - 1;
                    let n_minus_1_expr = ctx.add(Expr::Number(num_rational::BigRational::from_integer(n_minus_1)));
                    let term_nm1 = ctx.add(Expr::Mul(n_minus_1_expr, x_val));
                    
                    // sin(nx) = sin((n-1)x)cos(x) + cos((n-1)x)sin(x)
                    // cos(nx) = cos((n-1)x)cos(x) - sin((n-1)x)sin(x)
                    
                    let sin_nm1 = ctx.add(Expr::Function("sin".to_string(), vec![term_nm1]));
                    let cos_nm1 = ctx.add(Expr::Function("cos".to_string(), vec![term_nm1]));
                    let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![x_val]));
                    let cos_x = ctx.add(Expr::Function("cos".to_string(), vec![x_val]));
                    
                    if name == "sin" {
                        let t1 = ctx.add(Expr::Mul(sin_nm1, cos_x));
                        let t2 = ctx.add(Expr::Mul(cos_nm1, sin_x));
                        let new_expr = ctx.add(Expr::Add(t1, t2));
                        return Some(Rewrite {
                            new_expr,
                            description: format!("sin({}x) expansion", n_val),
                        });
                    } else {
                        // cos
                        let t1 = ctx.add(Expr::Mul(cos_nm1, cos_x));
                        let t2 = ctx.add(Expr::Mul(sin_nm1, sin_x));
                        let new_expr = ctx.add(Expr::Sub(t1, t2));
                        return Some(Rewrite {
                            new_expr,
                            description: format!("cos({}x) expansion", n_val),
                        });
                    }
                }
            }
        }
        None
    }
);



define_rule!(
    CanonicalizeTrigSquareRule,
    "Canonicalize Trig Square",
    |ctx, expr| {
        // cos^2(x) -> 1 - sin^2(x)
        let expr_data = ctx.get(expr).clone();
        if let Expr::Pow(base, exp) = expr_data {
            if let Expr::Number(n) = ctx.get(exp) {
                if n.is_integer() && *n == num_rational::BigRational::from_integer(2.into()) {
                    if let Expr::Function(name, args) = ctx.get(base) {
                        if name == "cos" && args.len() == 1 {
                            let arg = args[0];
                            // 1 - sin^2(x)
                            let one = ctx.num(1);
                            let sin_x = ctx.add(Expr::Function("sin".to_string(), vec![arg]));
                            let two = ctx.num(2);
                            let sin_sq = ctx.add(Expr::Pow(sin_x, two));
                            let new_expr = ctx.add(Expr::Sub(one, sin_sq));
                            return Some(Rewrite {
                                new_expr,
                                description: "cos^2(x) -> 1 - sin^2(x)".to_string(),
                            });
                        }
                    }
                }
            }
        }
        None
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateTrigRule));
    simplifier.add_rule(Box::new(PythagoreanIdentityRule));
    simplifier.add_rule(Box::new(AngleIdentityRule));
    simplifier.add_rule(Box::new(TanToSinCosRule));
    simplifier.add_rule(Box::new(DoubleAngleRule));
    simplifier.add_rule(Box::new(RecursiveTrigExpansionRule));
    simplifier.add_rule(Box::new(CanonicalizeTrigSquareRule));
}
