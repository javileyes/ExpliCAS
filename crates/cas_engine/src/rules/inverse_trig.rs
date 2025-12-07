use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_traits::One;
use std::cmp::Ordering;

// ==================== Helper Functions for Pattern Matching ====================

/// Check if expression equals 1
fn is_one(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(expr) {
        n.is_one()
    } else {
        false
    }
}

/// Check if two expressions are reciprocals: a = 1/b or b = 1/a
fn are_reciprocals(ctx: &Context, expr1: ExprId, expr2: ExprId) -> bool {
    // Get clones to avoid borrow issues
    let data1 = ctx.get(expr1).clone();
    let data2 = ctx.get(expr2).clone();

    // Case 1: expr2 = 1 / expr1
    if let Expr::Div(num, den) = &data2 {
        if is_one(ctx, *num) {
            // Compare semantically, not just ExprId
            if crate::ordering::compare_expr(ctx, *den, expr1) == Ordering::Equal {
                return true;
            }
        }
    }

    // Case 2: expr1 = 1 / expr2
    if let Expr::Div(num, den) = &data1 {
        if is_one(ctx, *num) {
            // Compare semantically, not just ExprId
            if crate::ordering::compare_expr(ctx, *den, expr2) == Ordering::Equal {
                return true;
            }
        }
    }

    false
}

/// Check if expression is sin(x)/cos(x) pattern (expanded tan)
/// Returns Some(x) if pattern matches, None otherwise
fn is_expanded_tan(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Div(num, den) = ctx.get(expr) {
        if let (Expr::Function(num_fn, num_args), Expr::Function(den_fn, den_args)) =
            (ctx.get(*num), ctx.get(*den))
        {
            if num_fn == "sin"
                && den_fn == "cos"
                && num_args.len() == 1
                && den_args.len() == 1
                && num_args[0] == den_args[0]
            {
                return Some(num_args[0]); // Return the argument x
            }
        }
    }
    None
}

// ==================== Inverse Trig Identity Rules ====================

// Rule 1: Composition Identities - sin(arcsin(x)) = x, etc.
define_rule!(
    InverseTrigCompositionRule,
    "Inverse Trig Composition",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(outer_name, outer_args) = ctx.get(expr) {
            if outer_args.len() == 1 {
                let inner_expr = outer_args[0];

                // Check for literal function composition first
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

                // ✨ NEW: Check for expanded tan in arctan: arctan(sin(x)/cos(x)) = x
                if outer_name == "arctan" {
                    if let Some(arg) = is_expanded_tan(ctx, inner_expr) {
                        return Some(Rewrite {
                            new_expr: arg,
                            description: "arctan(sin(x)/cos(x)) = arctan(tan(x)) = x".to_string(),
                        });
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
                    // ✨ ENHANCED: Use are_reciprocals helper for better pattern matching
                    if are_reciprocals(ctx, l_args[0], r_args[0]) {
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

// ==================== Phase 5: Inverse Function Relations ====================
// Unify inverse trig functions by converting arcsec/arccsc/arccot to arccos/arcsin/arctan

/// arcsec(x) → arccos(1/x)
define_rule!(
    ArcsecToArccosRule,
    "arcsec(x) → arccos(1/x)",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if (name == "arcsec" || name == "asec") && args.len() == 1 {
                let arg = args[0];

                // Build 1/arg
                let one = ctx.num(1);
                let reciprocal = ctx.add(Expr::Div(one, arg));

                // Build arccos(1/arg)
                let result = ctx.add(Expr::Function("arccos".to_string(), vec![reciprocal]));

                return Some(Rewrite {
                    new_expr: result,
                    description: "arcsec(x) → arccos(1/x)".to_string(),
                });
            }
        }
        None
    }
);

/// arccsc(x) → arcsin(1/x)
define_rule!(
    ArccscToArcsinRule,
    "arccsc(x) → arcsin(1/x)",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if (name == "arccsc" || name == "acsc") && args.len() == 1 {
                let arg = args[0];

                // Build 1/arg
                let one = ctx.num(1);
                let reciprocal = ctx.add(Expr::Div(one, arg));

                // Build arcsin(1/arg)
                let result = ctx.add(Expr::Function("arcsin".to_string(), vec![reciprocal]));

                return Some(Rewrite {
                    new_expr: result,
                    description: "arccsc(x) → arcsin(1/x)".to_string(),
                });
            }
        }
        None
    }
);

/// arccot(x) → arctan(1/x)
/// Simplified version - works for all x ≠ 0 on principal branch
define_rule!(
    ArccotToArctanRule,
    "arccot(x) → arctan(1/x)",
    Some(vec!["Function"]),
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            if (name == "arccot" || name == "acot") && args.len() == 1 {
                let arg = args[0];

                // Build 1/arg
                let one = ctx.num(1);
                let reciprocal = ctx.add(Expr::Div(one, arg));

                // Build arctan(1/arg)
                let result = ctx.add(Expr::Function("arctan".to_string(), vec![reciprocal]));

                return Some(Rewrite {
                    new_expr: result,
                    description: "arccot(x) → arctan(1/x)".to_string(),
                });
            }
        }
        None
    }
);

// ==================== Registration ====================

pub fn register(simplifier: &mut crate::Simplifier) {
    simplifier.add_rule(Box::new(InverseTrigCompositionRule));
    simplifier.add_rule(Box::new(InverseTrigSumRule));
    simplifier.add_rule(Box::new(InverseTrigAtanRule));
    simplifier.add_rule(Box::new(InverseTrigNegativeRule));
    simplifier.add_rule(Box::new(ArcsecToArccosRule));
    simplifier.add_rule(Box::new(ArccscToArcsinRule));
    simplifier.add_rule(Box::new(ArccotToArctanRule));
}
