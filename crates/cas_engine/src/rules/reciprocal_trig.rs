use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_traits::{One, Zero};

// ==================== Helper Functions ====================

/// Check if expression equals 0
fn is_zero(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        n.is_zero()
    } else {
        false
    }
}

/// Check if expression equals 1
fn is_one(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        n.is_one()
    } else {
        false
    }
}

/// Check if expression equals π/4
fn is_pi_over_four(ctx: &Context, expr: ExprId) -> bool {
    // Handle both forms: pi/4 and 1/4 * pi (canonicalized)
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Constant(c) = ctx.get(*num) {
            if matches!(c, cas_ast::Constant::Pi) {
                if let Expr::Number(n) = ctx.get(*den) {
                    return *n == num_rational::Ratio::from_integer(4.into());
                }
            }
        }
    }

    // Also check for 1/4 * pi form (after simplification, becomes Number(1/4) * Pi)
    if let Expr::Mul(l, r) = ctx.get(expr) {
        // Could be pi * 1/4 or 1/4 * pi
        let (num_part, const_part) = if let Expr::Constant(_) = ctx.get(*l) {
            (*r, *l)
        } else if let Expr::Constant(_) = ctx.get(*r) {
            (*l, *r)
        } else {
            return false;
        };

        // Check if const_part is Pi
        if let Expr::Constant(c) = ctx.get(const_part) {
            if matches!(c, cas_ast::Constant::Pi) {
                // Check if num_part is 1/4 (as a Number)
                if let Expr::Number(n) = ctx.get(num_part) {
                    // Check if it's 1/4
                    return *n == num_rational::Ratio::new(1.into(), 4.into());
                }
            }
        }
    }

    false
}

/// Check if expression equals π/2
fn is_pi_over_two(ctx: &Context, expr: ExprId) -> bool {
    // Handle both forms: pi/2 and 1/2 * pi (canonicalized)
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let Expr::Constant(c) = ctx.get(*num) {
            if matches!(c, cas_ast::Constant::Pi) {
                if let Expr::Number(n) = ctx.get(*den) {
                    return *n == num_rational::Ratio::from_integer(2.into());
                }
            }
        }
    }

    // Also check for 1/2 * pi form (after simplification, becomes Number(1/2) * Pi)
    if let Expr::Mul(l, r) = ctx.get(expr) {
        // Could be pi * 1/2 or 1/2 * pi
        let (num_part, const_part) = if let Expr::Constant(_) = ctx.get(*l) {
            (*r, *l)
        } else if let Expr::Constant(_) = ctx.get(*r) {
            (*l, *r)
        } else {
            return false;
        };

        // Check if const_part is Pi
        if let Expr::Constant(c) = ctx.get(const_part) {
            if matches!(c, cas_ast::Constant::Pi) {
                // Check if num_part is 1/2 (as a Number)
                if let Expr::Number(n) = ctx.get(num_part) {
                    // Check if it's 1/2
                    return *n == num_rational::Ratio::new(1.into(), 2.into());
                }
            }
        }
    }

    false
}

// ==================== Reciprocal Trig Rules ====================

// Rule 1: Evaluate reciprocal trig functions at special values
define_rule!(
    EvaluateReciprocalTrigRule,
    "Evaluate Reciprocal Trig Functions",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if args.len() == 1 {
                let arg = args[0];
                let name = name.clone();

                match name.as_str() {
                    // cot(π/4) = 1
                    "cot" => {
                        if is_pi_over_four(ctx, arg) {
                            return Some(Rewrite {
                                new_expr: ctx.num(1),
                                description: "cot(π/4) = 1".to_string(),
                            });
                        }
                        // cot(π/2) = 0
                        if is_pi_over_two(ctx, arg) {
                            return Some(Rewrite {
                                new_expr: ctx.num(0),
                                description: "cot(π/2) = 0".to_string(),
                            });
                        }
                    }
                    // sec(0) = 1
                    "sec" => {
                        if is_zero(ctx, arg) {
                            return Some(Rewrite {
                                new_expr: ctx.num(1),
                                description: "sec(0) = 1".to_string(),
                            });
                        }
                    }
                    // csc(π/2) = 1
                    "csc" => {
                        if is_pi_over_two(ctx, arg) {
                            return Some(Rewrite {
                                new_expr: ctx.num(1),
                                description: "csc(π/2) = 1".to_string(),
                            });
                        }
                    }
                    // arccot(1) = π/4
                    "arccot" => {
                        if is_one(ctx, arg) {
                            let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                            let four = ctx.num(4);
                            return Some(Rewrite {
                                new_expr: ctx.add(Expr::Div(pi, four)),
                                description: "arccot(1) = π/4".to_string(),
                            });
                        }
                        // arccot(0) = π/2
                        if is_zero(ctx, arg) {
                            let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                            let two = ctx.num(2);
                            return Some(Rewrite {
                                new_expr: ctx.add(Expr::Div(pi, two)),
                                description: "arccot(0) = π/2".to_string(),
                            });
                        }
                    }
                    // arcsec(1) = 0
                    "arcsec" => {
                        if is_one(ctx, arg) {
                            return Some(Rewrite {
                                new_expr: ctx.num(0),
                                description: "arcsec(1) = 0".to_string(),
                            });
                        }
                    }
                    // arccsc(1) = π/2
                    "arccsc" => {
                        if is_one(ctx, arg) {
                            let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                            let two = ctx.num(2);
                            return Some(Rewrite {
                                new_expr: ctx.add(Expr::Div(pi, two)),
                                description: "arccsc(1) = π/2".to_string(),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        None
    }
);

// Rule 2: Composition identities - cot(arccot(x)) = x, etc.
define_rule!(
    ReciprocalTrigCompositionRule,
    "Reciprocal Trig Composition",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_args.len() == 1 {
                let inner_expr = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner_expr) {
                    if inner_args.len() == 1 {
                        let x = inner_args[0];

                        // cot(arccot(x)) = x
                        if outer_name == "cot" && inner_name == "arccot" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "cot(arccot(x)) = x".to_string(),
                            });
                        }

                        // sec(arcsec(x)) = x
                        if outer_name == "sec" && inner_name == "arcsec" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "sec(arcsec(x)) = x".to_string(),
                            });
                        }

                        // csc(arccsc(x)) = x
                        if outer_name == "csc" && inner_name == "arccsc" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "csc(arccsc(x)) = x".to_string(),
                            });
                        }

                        // arccot(cot(x)) = x
                        if outer_name == "arccot" && inner_name == "cot" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "arccot(cot(x)) = x".to_string(),
                            });
                        }

                        // arcsec(sec(x)) = x
                        if outer_name == "arcsec" && inner_name == "sec" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "arcsec(sec(x)) = x".to_string(),
                            });
                        }

                        // arccsc(csc(x)) = x
                        if outer_name == "arccsc" && inner_name == "csc" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "arccsc(csc(x)) = x".to_string(),
                            });
                        }
                    }
                }
            }
        }
        None
    }
);

// Rule 3: Negative argument identities
define_rule!(
    ReciprocalTrigNegativeRule,
    "Reciprocal Trig Negative Argument",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if args.len() == 1 {
                let arg = args[0];
                if let Expr::Neg(inner) = ctx.get(arg) {
                    let name = name.clone();
                    match name.as_str() {
                        // cot(-x) = -cot(x) (odd function)
                        "cot" => {
                            let cot_inner =
                                ctx.add(Expr::Function("cot".to_string(), vec![*inner]));
                            let new_expr = ctx.add(Expr::Neg(cot_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "cot(-x) = -cot(x)".to_string(),
                            });
                        }
                        // sec(-x) = sec(x) (even function)
                        "sec" => {
                            let new_expr = ctx.add(Expr::Function("sec".to_string(), vec![*inner]));
                            return Some(Rewrite {
                                new_expr,
                                description: "sec(-x) = sec(x)".to_string(),
                            });
                        }
                        // csc(-x) = -csc(x) (odd function)
                        "csc" => {
                            let csc_inner =
                                ctx.add(Expr::Function("csc".to_string(), vec![*inner]));
                            let new_expr = ctx.add(Expr::Neg(csc_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "csc(-x) = -csc(x)".to_string(),
                            });
                        }
                        // arccot(-x) = -arccot(x) (using odd function convention)
                        "arccot" => {
                            let arccot_inner =
                                ctx.add(Expr::Function("arccot".to_string(), vec![*inner]));
                            let new_expr = ctx.add(Expr::Neg(arccot_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "arccot(-x) = -arccot(x)".to_string(),
                            });
                        }
                        // arcsec(-x) = π - arcsec(x)
                        "arcsec" => {
                            let arcsec_inner =
                                ctx.add(Expr::Function("arcsec".to_string(), vec![*inner]));
                            let pi = ctx.add(Expr::Constant(cas_ast::Constant::Pi));
                            let new_expr = ctx.add(Expr::Sub(pi, arcsec_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "arcsec(-x) = π - arcsec(x)".to_string(),
                            });
                        }
                        // arccsc(-x) = -arccsc(x) (odd function)
                        "arccsc" => {
                            let arccsc_inner =
                                ctx.add(Expr::Function("arccsc".to_string(), vec![*inner]));
                            let new_expr = ctx.add(Expr::Neg(arccsc_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "arccsc(-x) = -arccsc(x)".to_string(),
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

/// Register all reciprocal trig function rules
pub fn register(simplifier: &mut crate::engine::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateReciprocalTrigRule));
    simplifier.add_rule(Box::new(ReciprocalTrigCompositionRule));
    simplifier.add_rule(Box::new(ReciprocalTrigNegativeRule));
}
