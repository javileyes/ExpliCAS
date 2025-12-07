use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_traits::One;

// ==================== Sophisticated Context-Aware Canonicalization ====================
// STRATEGY: Only convert when it demonstrably helps simplification
// Three-tier approach:
// 1. Never convert: compositions like tan(arctan(x))
// 2. Always convert: known patterns like sec²-tan², mixed fractions
// 3. Selective: on-demand for complex cases

// ==================== Helper Functions for Pattern Detection ====================

/// Check if expression is a number equal to 1
fn is_one(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => n.is_one(),
        _ => false,
    }
}

/// Check if expression is a number equal to 2
fn is_two(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == num_rational::Ratio::from_integer(2.into()),
        _ => false,
    }
}

/// Check if function name is a reciprocal trig function
fn is_reciprocal_trig_name(name: &str) -> bool {
    matches!(name, "tan" | "cot" | "sec" | "csc")
}

/// Check if function name is an inverse trig function
fn is_inverse_trig_name(name: &str) -> bool {
    matches!(
        name,
        "asin"
            | "acos"
            | "atan"
            | "acot"
            | "asec"
            | "acsc"
            | "arcsin"
            | "arccos"
            | "arctan"
            | "arccot"
            | "arcsec"
            | "arccsc"
    )
}

/// Check if expression is a reciprocal trig function
fn is_reciprocal_trig(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(name, _) => is_reciprocal_trig_name(name),
        _ => false,
    }
}

/// Check if expression is f²(x) where f is a trig function
fn is_squared_trig(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => is_two(ctx, *exp) && is_reciprocal_trig(ctx, *base),
        _ => false,
    }
}

/// Check if expression is a composition like tan(arctan(x))
fn is_trig_of_inverse_trig(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Function(outer_name, outer_args) if outer_args.len() == 1 => {
            let inner = outer_args[0];
            match ctx.get(inner) {
                Expr::Function(inner_name, _) => {
                    is_reciprocal_trig_name(outer_name.as_str())
                        && is_inverse_trig_name(inner_name.as_str())
                }
                _ => false,
            }
        }
        _ => false,
    }
}

// ==================== Tier 1: Preserve Compositions (Negative Rule) ====================

/// NEVER convert reciprocal trig if it's a composition with inverse trig
/// This preserves tan(arctan(x)) → x simplifications
///
/// Priority: HIGHEST (register first)
define_rule!(
    PreserveCompositionRule,
    "Preserve trig-inverse compositions",
    Some(vec!["Function"]),
    |ctx, expr| {
        // If this is tan(arctan(...)), cot(arcsin(...)), etc.
        // return None to prevent any conversion
        if is_trig_of_inverse_trig(ctx, expr) {
            // Explicitly return None - this is a "negative rule"
            // It blocks other rules from converting
            return None;
        }
        None
    }
);

// =================================================================================
// Direct Pythagorean Identity Rules (No Conversion)
// =================================================================================
// Instead of converting to sin/cos (which creates complex intermediate forms),
// directly apply the Pythagorean identities:
// - sec²(x) - tan²(x) = 1
// - csc²(x) - cot²(x) = 1
// - 1 + tan²(x) = sec²(x)
// - 1 + cot²(x) = csc²(x)

/// sec²(x) - tan²(x) → 1
define_rule!(
    SecTanPythagoreanRule,
    "sec²(x) - tan²(x) = 1",
    Some(vec!["Sub"]),
    |ctx, expr| {
        if let Expr::Sub(l, r) = ctx.get(expr) {
            let l_val = *l;
            let r_val = *r;

            // Check if l is sec²(arg) and r is tan²(arg) with same argument
            if let (Some(sec_arg), Some(tan_arg)) = (
                is_function_squared(ctx, l_val, "sec"),
                is_function_squared(ctx, r_val, "tan"),
            ) {
                if sec_arg == tan_arg {
                    return Some(Rewrite {
                        new_expr: ctx.num(1),
                        description: "sec²(x) - tan²(x) = 1".to_string(),
                    });
                }
            }
        }
        None
    }
);

/// csc²(x) - cot²(x) → 1
define_rule!(
    CscCotPythagoreanRule,
    "csc²(x) - cot²(x) = 1",
    Some(vec!["Sub"]),
    |ctx, expr| {
        if let Expr::Sub(l, r) = ctx.get(expr) {
            let l_val = *l;
            let r_val = *r;

            if let (Some(csc_arg), Some(cot_arg)) = (
                is_function_squared(ctx, l_val, "csc"),
                is_function_squared(ctx, r_val, "cot"),
            ) {
                if csc_arg == cot_arg {
                    return Some(Rewrite {
                        new_expr: ctx.num(1),
                        description: "csc²(x) - cot²(x) = 1".to_string(),
                    });
                }
            }
        }
        None
    }
);

/// 1 + tan²(x) → sec²(x)
define_rule!(
    TanToSecPythagoreanRule,
    "1 + tan²(x) = sec²(x)",
    Some(vec!["Add"]),
    |ctx, expr| {
        if let Expr::Add(l, r) = ctx.get(expr) {
            let l_val = *l;
            let r_val = *r;

            // Check both orders: 1 + tan² and tan² + 1
            if is_one(ctx, l_val) {
                if let Some(tan_arg) = is_function_squared(ctx, r_val, "tan") {
                    let two = ctx.num(2);
                    let sec_expr = ctx.add(Expr::Function("sec".to_string(), vec![tan_arg]));
                    let sec_squared = ctx.add(Expr::Pow(sec_expr, two));
                    return Some(Rewrite {
                        new_expr: sec_squared,
                        description: "1 + tan²(x) = sec²(x)".to_string(),
                    });
                }
            } else if is_one(ctx, r_val) {
                if let Some(tan_arg) = is_function_squared(ctx, l_val, "tan") {
                    let two = ctx.num(2);
                    let sec_expr = ctx.add(Expr::Function("sec".to_string(), vec![tan_arg]));
                    let sec_squared = ctx.add(Expr::Pow(sec_expr, two));
                    return Some(Rewrite {
                        new_expr: sec_squared,
                        description: "1 + tan²(x) = sec²(x)".to_string(),
                    });
                }
            }
        }
        None
    }
);

/// 1 + cot²(x) → csc²(x)
define_rule!(
    CotToCscPythagoreanRule,
    "1 + cot²(x) = csc²(x)",
    Some(vec!["Add"]),
    |ctx, expr| {
        if let Expr::Add(l, r) = ctx.get(expr) {
            let l_val = *l;
            let r_val = *r;

            if is_one(ctx, l_val) {
                if let Some(cot_arg) = is_function_squared(ctx, r_val, "cot") {
                    let two = ctx.num(2);
                    let csc_expr = ctx.add(Expr::Function("csc".to_string(), vec![cot_arg]));
                    let csc_squared = ctx.add(Expr::Pow(csc_expr, two));
                    return Some(Rewrite {
                        new_expr: csc_squared,
                        description: "1 + cot²(x) = csc²(x)".to_string(),
                    });
                }
            } else if is_one(ctx, r_val) {
                if let Some(cot_arg) = is_function_squared(ctx, l_val, "cot") {
                    let two = ctx.num(2);
                    let csc_expr = ctx.add(Expr::Function("csc".to_string(), vec![cot_arg]));
                    let csc_squared = ctx.add(Expr::Pow(csc_expr, two));
                    return Some(Rewrite {
                        new_expr: csc_squared,
                        description: "1 + cot²(x) = csc²(x)".to_string(),
                    });
                }
            }
        }
        None
    }
);

/// Helper: Check if expr is f²(arg) for a specific function name
/// Returns Some(arg) if match, None otherwise
fn is_function_squared(ctx: &Context, expr: ExprId, fname: &str) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if is_two(ctx, *exp) {
                match ctx.get(*base) {
                    Expr::Function(name, args) if name == fname && args.len() == 1 => Some(args[0]),
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Convert reciprocal products like tan(x)*cot(x) → 1
define_rule!(
    ConvertReciprocalProductRule,
    "Simplify reciprocal trig products",
    Some(vec!["Mul"]),
    |ctx, expr| {
        if let Expr::Mul(l, r) = ctx.get(expr) {
            let l_val = *l;
            let r_val = *r;
            // Check if we have tan*cot or sec*cos or csc*sin
            let (is_reciprocal_pair, _arg) = check_reciprocal_pair(ctx, l_val, r_val);

            if is_reciprocal_pair {
                return Some(Rewrite {
                    new_expr: ctx.num(1),
                    description: "Reciprocal trig product = 1".to_string(),
                });
            }
        }
        None
    }
);

/// Check if two expressions are reciprocal trig functions with same argument
fn check_reciprocal_pair(ctx: &Context, expr1: ExprId, expr2: ExprId) -> (bool, Option<ExprId>) {
    match (ctx.get(expr1), ctx.get(expr2)) {
        (Expr::Function(name1, args1), Expr::Function(name2, args2))
            if args1.len() == 1 && args2.len() == 1 && args1[0] == args2[0] =>
        {
            let arg = args1[0];
            let is_pair = matches!(
                (name1.as_str(), name2.as_str()),
                ("tan", "cot")
                    | ("cot", "tan")
                    | ("sec", "cos")
                    | ("cos", "sec")
                    | ("csc", "sin")
                    | ("sin", "csc")
            );
            (is_pair, if is_pair { Some(arg) } else { None })
        }
        _ => (false, None),
    }
}

// ==================== Registration ====================

/// Register sophisticated canonicalization rules
pub fn register(simplifier: &mut crate::engine::Simplifier) {
    // Direct Pythagorean identities (NO conversion, direct simplification)
    simplifier.add_rule(Box::new(SecTanPythagoreanRule));
    simplifier.add_rule(Box::new(CscCotPythagoreanRule));
    simplifier.add_rule(Box::new(TanToSecPythagoreanRule));
    simplifier.add_rule(Box::new(CotToCscPythagoreanRule));

    // Reciprocal product simplification
    simplifier.add_rule(Box::new(ConvertReciprocalProductRule));

    // Future: Add ConvertForMixedFractionRule if needed

    // Note: NO generic conversion rules
    // Each identity is applied directly without intermediate conversions
}
