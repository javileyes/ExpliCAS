use crate::define_rule;
use crate::helpers::{is_one, is_zero};
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use std::cmp::Ordering;

// ==================== Helper Functions ====================

// is_zero and is_one are now imported from crate::helpers

/// Check if expression equals 2
fn is_two(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        *n == num_rational::Ratio::from_integer(2.into())
    } else {
        false
    }
}

// ==================== Hyperbolic Function Rules ====================

// Rule 1: Evaluate hyperbolic functions at special values
define_rule!(
    EvaluateHyperbolicRule,
    "Evaluate Hyperbolic Functions",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if args.len() == 1 {
                let arg = args[0];
                let name = name.clone(); // Clone to avoid borrow issues

                match name.as_str() {
                    // sinh(0) = 0, tanh(0) = 0
                    "sinh" | "tanh" => {
                        if is_zero(ctx, arg) {
                            return Some(Rewrite {
                                new_expr: ctx.num(0),
                                description: format!("{}(0) = 0", name),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                            });
                        }
                    }
                    // cosh(0) = 1
                    "cosh" => {
                        if is_zero(ctx, arg) {
                            return Some(Rewrite {
                                new_expr: ctx.num(1),
                                description: "cosh(0) = 1".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                            });
                        }
                    }
                    // asinh(0) = 0, atanh(0) = 0
                    "asinh" | "atanh" => {
                        if is_zero(ctx, arg) {
                            return Some(Rewrite {
                                new_expr: ctx.num(0),
                                description: format!("{}(0) = 0", name),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                            });
                        }
                    }
                    // acosh(1) = 0
                    "acosh" => {
                        if is_one(ctx, arg) {
                            return Some(Rewrite {
                                new_expr: ctx.num(0),
                                description: "acosh(1) = 0".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
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

// Rule 2: Composition identities - sinh(asinh(x)) = x, etc.
define_rule!(
    HyperbolicCompositionRule,
    "Hyperbolic Composition",
    Some(vec!["Function"]),
    solve_safety: crate::solve_safety::SolveSafety::NeedsCondition(
        crate::assumptions::ConditionClass::Analytic
    ),
    |ctx, expr, _parent_ctx| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_args.len() == 1 {
                let inner_expr = outer_args[0];
                if let Expr::Function(inner_name, inner_args) = ctx.get(inner_expr) {
                    if inner_args.len() == 1 {
                        let x = inner_args[0];

                        // sinh(asinh(x)) = x
                        if outer_name == "sinh" && inner_name == "asinh" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "sinh(asinh(x)) = x".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
            required_conditions: vec![],
                            });
                        }

                        // cosh(acosh(x)) = x
                        if outer_name == "cosh" && inner_name == "acosh" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "cosh(acosh(x)) = x".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
            required_conditions: vec![],
                            });
                        }

                        // tanh(atanh(x)) = x
                        if outer_name == "tanh" && inner_name == "atanh" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "tanh(atanh(x)) = x".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
            required_conditions: vec![],
                            });
                        }

                        // asinh(sinh(x)) = x
                        if outer_name == "asinh" && inner_name == "sinh" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "asinh(sinh(x)) = x".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
            required_conditions: vec![],
                            });
                        }

                        // acosh(cosh(x)) = x
                        if outer_name == "acosh" && inner_name == "cosh" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "acosh(cosh(x)) = x".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
            required_conditions: vec![],
                            });
                        }

                        // atanh(tanh(x)) = x
                        if outer_name == "atanh" && inner_name == "tanh" {
                            return Some(Rewrite {
                                new_expr: x,
                                description: "atanh(tanh(x)) = x".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
            required_conditions: vec![],
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
    HyperbolicNegativeRule,
    "Hyperbolic Negative Argument",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if args.len() == 1 {
                let arg = args[0];
                if let Expr::Neg(inner) = ctx.get(arg) {
                    match name.as_str() {
                        // sinh(-x) = -sinh(x) (odd function)
                        "sinh" => {
                            let sinh_inner =
                                ctx.add(Expr::Function("sinh".to_string(), vec![*inner]));
                            let new_expr = ctx.add(Expr::Neg(sinh_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "sinh(-x) = -sinh(x)".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                            });
                        }
                        // cosh(-x) = cosh(x) (even function)
                        "cosh" => {
                            let new_expr =
                                ctx.add(Expr::Function("cosh".to_string(), vec![*inner]));
                            return Some(Rewrite {
                                new_expr,
                                description: "cosh(-x) = cosh(x)".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                            });
                        }
                        // tanh(-x) = -tanh(x) (odd function)
                        "tanh" => {
                            let tanh_inner =
                                ctx.add(Expr::Function("tanh".to_string(), vec![*inner]));
                            let new_expr = ctx.add(Expr::Neg(tanh_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "tanh(-x) = -tanh(x)".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                            });
                        }
                        // asinh(-x) = -asinh(x) (odd function)
                        "asinh" => {
                            let asinh_inner =
                                ctx.add(Expr::Function("asinh".to_string(), vec![*inner]));
                            let new_expr = ctx.add(Expr::Neg(asinh_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "asinh(-x) = -asinh(x)".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
                            });
                        }
                        // atanh(-x) = -atanh(x) (odd function)
                        "atanh" => {
                            let atanh_inner =
                                ctx.add(Expr::Function("atanh".to_string(), vec![*inner]));
                            let new_expr = ctx.add(Expr::Neg(atanh_inner));
                            return Some(Rewrite {
                                new_expr,
                                description: "atanh(-x) = -atanh(x)".to_string(),
                                before_local: None,
                                after_local: None,
                                assumption_events: Default::default(),
                                required_conditions: vec![],
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

// Rule 4: Hyperbolic Pythagorean identity: cosh²(x) - sinh²(x) = 1
define_rule!(
    HyperbolicPythagoreanRule,
    "Hyperbolic Pythagorean Identity",
    Some(vec!["Sub"]),
    |ctx, expr| {
        if let Expr::Sub(l, r) = ctx.get(expr) {
            let l_data = ctx.get(*l).clone();
            let r_data = ctx.get(*r).clone();

            // Check pattern: cosh(x)^2 - sinh(x)^2
            if let (Expr::Pow(l_base, l_exp), Expr::Pow(r_base, r_exp)) = (&l_data, &r_data) {
                // Both should be squared
                if is_two(ctx, *l_exp) && is_two(ctx, *r_exp) {
                    if let (Expr::Function(l_fn, l_args), Expr::Function(r_fn, r_args)) =
                        (ctx.get(*l_base), ctx.get(*r_base))
                    {
                        // Case 1: cosh(x)^2 - sinh(x)^2 = 1
                        if l_fn == "cosh"
                            && r_fn == "sinh"
                            && l_args.len() == 1
                            && r_args.len() == 1
                        {
                            // Check if arguments are the same (semantic comparison)
                            if crate::ordering::compare_expr(ctx, l_args[0], r_args[0])
                                == Ordering::Equal
                            {
                                return Some(Rewrite {
                                    new_expr: ctx.num(1),
                                    description: "cosh²(x) - sinh²(x) = 1".to_string(),
                                    before_local: None,
                                    after_local: None,
                                    assumption_events: Default::default(),
                                    required_conditions: vec![],
                                });
                            }
                        }

                        // Case 2: sinh(x)^2 - cosh(x)^2 = -1
                        if l_fn == "sinh"
                            && r_fn == "cosh"
                            && l_args.len() == 1
                            && r_args.len() == 1
                        {
                            // Check if arguments are the same (semantic comparison)
                            if crate::ordering::compare_expr(ctx, l_args[0], r_args[0])
                                == Ordering::Equal
                            {
                                return Some(Rewrite {
                                    new_expr: ctx.num(-1),
                                    description: "sinh²(x) - cosh²(x) = -1".to_string(),
                                    before_local: None,
                                    after_local: None,
                                    assumption_events: Default::default(),
                                    required_conditions: vec![],
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

/// Register all hyperbolic function rules
pub fn register(simplifier: &mut crate::engine::Simplifier) {
    simplifier.add_rule(Box::new(EvaluateHyperbolicRule));
    simplifier.add_rule(Box::new(HyperbolicCompositionRule));
    simplifier.add_rule(Box::new(HyperbolicNegativeRule));
    simplifier.add_rule(Box::new(HyperbolicPythagoreanRule));
}
