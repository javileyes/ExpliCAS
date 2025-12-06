use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::Expr;
use num_traits::One;
use std::cmp::Ordering;

// Rule 1: Composition Identities - sin(arcsin(x)) = x, etc.
define_rule!(
    InverseTrigCompositionRule,
    "Inverse Trig Composition",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_args.len() == 1 {
                let inner_expr = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner_expr) {
                    if inner_args.len() == 1 {
                        let x = inner_args[0];

                        // sin(arcsin(x)) = x
                        if outer_name == "sin" && inner_name == "arcsin" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "sin(arcsin(x)) = x".to_string(),
                            });
                        }

                        // cos(arccos(x)) = x
                        if outer_name == "cos" && inner_name == "arccos" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "cos(arccos(x)) = x".to_string(),
                            });
                        }

                        // tan(arctan(x)) = x
                        if outer_name == "tan" && inner_name == "arctan" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "tan(arctan(x)) = x".to_string(),
                            });
                        }

                        // arcsin(sin(x)) = x (with domain restrictions, but we simplify anyway)
                        if outer_name == "arcsin" && inner_name == "sin" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "arcsin(sin(x)) = x".to_string(),
                            });
                        }

                        // arccos(cos(x)) = x
                        if outer_name == "arccos" && inner_name == "cos" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "arccos(cos(x)) = x".to_string(),
                            });
                        }

                        // arctan(tan(x)) = x
                        if outer_name == "arctan" && inner_name == "tan" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "arctan(tan(x)) = x".to_string(),
                            });
                        }
                    }
                }
            }
        }
        None
    }
);

// Rule 2: arcsin(x) + arccos(x) = π/2
define_rule!(
    InverseTrigSumRule,
    "Inverse Trig Sum Identity",
    Some(vec!["Add"]),
    |ctx, expr| {
        if let Expr::Add(l, r) = ctx.get(expr) {
            let l_data = ctx.get(*l).clone();
            let r_data = ctx.get(*r).clone();

            // arcsin(x) + arccos(x) = π/2
            if let (Expr::Function(l_name, l_args), Expr::Function(r_name, r_args)) =
                (l_data, r_data)
            {
                if l_args.len() == 1 && r_args.len() == 1 {
                    let l_arg = l_args[0];
                    let r_arg = r_args[0];

                    // Check if arguments are equal
                    let args_equal = l_arg == r_arg
                        || crate::ordering::compare_expr(ctx, l_arg, r_arg) == Ordering::Equal;

                    if args_equal {
                        if (l_name == "arcsin" && r_name == "arccos")
                            || (l_name == "arccos" && r_name == "arcsin")
                        {
                            let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                            let two = ctx.num(2);
                            let new_expr = ctx.add(Expr::Div(pi, two));
                            return Some(Rewrite {
                                new_expr,
                                description: "arcsin(x) + arccos(x) = π/2".to_string(),
                            });
                        }
                    }
                }
            }
        }
        None
    }
);

// Rule 3: arctan(x) + arctan(1/x) = π/2 (for x > 0)
define_rule!(
    InverseTrigAtanRule,
    "Inverse Tan Relations",
    Some(vec!["Add"]),
    |ctx, expr| {
        if let Expr::Add(l, r) = ctx.get(expr) {
            let l_data = ctx.get(*l).clone();
            let r_data = ctx.get(*r).clone();

            // arctan(x) + arctan(1/x) = π/2
            if let (Expr::Function(l_name, l_args), Expr::Function(r_name, r_args)) =
                (l_data, r_data)
            {
                if l_name == "arctan"
                    && r_name == "arctan"
                    && l_args.len() == 1
                    && r_args.len() == 1
                {
                    let x = l_args[0];
                    let y_data = ctx.get(r_args[0]).clone();

                    // Check if y = 1/x
                    if let Expr::Div(num, den) = y_data {
                        if let Expr::Number(n) = ctx.get(num) {
                            if n.is_one() && den == x {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let two = ctx.num(2);
                                let new_expr = ctx.add(Expr::Div(pi, two));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "arctan(x) + arctan(1/x) = π/2".to_string(),
                                });
                            }
                        }
                    }

                    // Also check the reverse: arctan(1/x) + arctan(x) = π/2
                    let x2 = r_args[0];
                    let y2_data = ctx.get(l_args[0]).clone();
                    if let Expr::Div(num, den) = y2_data {
                        if let Expr::Number(n) = ctx.get(num) {
                            if n.is_one() && den == x2 {
                                let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                                let two = ctx.num(2);
                                let new_expr = ctx.add(Expr::Div(pi, two));
                                return Some(Rewrite {
                                    new_expr,
                                    description: "arctan(1/x) + arctan(x) = π/2".to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }
        None
    }
);

// Rule 4: Negative argument handling for inverse trig
define_rule!(
    InverseTrigNegativeRule,
    "Inverse Trig Negative Argument",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if args.len() == 1 {
                let arg = args[0];

                // Check for negative argument: Neg(x) or Mul(-1, x)
                let inner_opt = match ctx.get(arg) {
                    Expr::Neg(inner) => Some(*inner),
                    Expr::Mul(l, r) => {
                        if let Expr::Number(n) = ctx.get(*l) {
                            if *n == num_rational::BigRational::from_integer((-1).into()) {
                                Some(*r)
                            } else {
                                None
                            }
                        } else if let Expr::Number(n) = ctx.get(*r) {
                            if *n == num_rational::BigRational::from_integer((-1).into()) {
                                Some(*l)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                if let Some(inner) = inner_opt {
                    match name.as_str() {
                        "arcsin" => {
                            // arcsin(-x) = -arcsin(x)
                            let arcsin_inner =
                                ctx.add(Expr::Function("arcsin".to_string(), vec![inner]));
                            let new_expr = ctx.add(Expr::Neg(arcsin_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "arcsin(-x) = -arcsin(x)".to_string(),
                            });
                        }
                        "arctan" => {
                            // arctan(-x) = -arctan(x)
                            let arctan_inner =
                                ctx.add(Expr::Function("arctan".to_string(), vec![inner]));
                            let new_expr = ctx.add(Expr::Neg(arctan_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "arctan(-x) = -arctan(x)".to_string(),
                            });
                        }
                        "arccos" => {
                            // arccos(-x) = π - arccos(x)
                            let arccos_inner =
                                ctx.add(Expr::Function("arccos".to_string(), vec![inner]));
                            let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                            let new_expr = ctx.add(Expr::Sub(pi, arccos_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "arccos(-x) = π - arccos(x)".to_string(),
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
        None
    }
);

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(InverseTrigCompositionRule));
    simplifier.add_rule(Box::new(InverseTrigSumRule));
    simplifier.add_rule(Box::new(InverseTrigAtanRule));
    simplifier.add_rule(Box::new(InverseTrigNegativeRule));
}
