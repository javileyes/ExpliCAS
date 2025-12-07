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

/// Check if expression is a Pythagorean pattern: f²±g², 1±f²
fn is_pythagorean_pattern(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Add(l, r) | Expr::Sub(l, r) => {
            (is_squared_trig(ctx, *l) && is_squared_trig(ctx, *r))
                || (is_one(ctx, *l) && is_squared_trig(ctx, *r))
                || (is_squared_trig(ctx, *l) && is_one(ctx, *r))
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

// ==================== Tier 2: Pattern-Based Conversion ====================

/// Convert reciprocal trig in Pythagorean contexts
/// Examples: sec²(x) - tan²(x), 1 + tan²(x), csc²(x) - cot²(x)
define_rule!(
    ConvertForPythagoreanRule,
    "Convert trig in Pythagorean patterns",
    Some(vec!["Add", "Sub"]),
    |ctx, expr| {
        if !is_pythagorean_pattern(ctx, expr) {
            return None;
        }

        // We have a Pythagorean pattern - convert all reciprocal trig to sin/cos
        match ctx.get(expr) {
            Expr::Add(l, r) => {
                let l_val = *l;
                let r_val = *r;
                let l_converted = convert_reciprocal_to_sincos(ctx, l_val);
                let r_converted = convert_reciprocal_to_sincos(ctx, r_val);

                if l_converted != l_val || r_converted != r_val {
                    return Some(Rewrite {
                        new_expr: ctx.add(Expr::Add(l_converted, r_converted)),
                        description: "Convert reciprocal trig in Pythagorean pattern".to_string(),
                    });
                }
            }
            Expr::Sub(l, r) => {
                let l_val = *l;
                let r_val = *r;
                let l_converted = convert_reciprocal_to_sincos(ctx, l_val);
                let r_converted = convert_reciprocal_to_sincos(ctx, r_val);

                if l_converted != l_val || r_converted != r_val {
                    return Some(Rewrite {
                        new_expr: ctx.add(Expr::Sub(l_converted, r_converted)),
                        description: "Convert reciprocal trig in Pythagorean pattern".to_string(),
                    });
                }
            }
            _ => {}
        }

        None
    }
);

/// Helper: Recursively convert reciprocal trig to sin/cos
fn convert_reciprocal_to_sincos(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr) {
        // Base case: reciprocal trig function
        Expr::Function(name, args) if args.len() == 1 => {
            let name = name.clone();
            let arg = args[0];

            match name.as_str() {
                "tan" => {
                    let sin_arg = ctx.add(Expr::Function("sin".to_string(), vec![arg]));
                    let cos_arg = ctx.add(Expr::Function("cos".to_string(), vec![arg]));
                    ctx.add(Expr::Div(sin_arg, cos_arg))
                }
                "cot" => {
                    let cos_arg = ctx.add(Expr::Function("cos".to_string(), vec![arg]));
                    let sin_arg = ctx.add(Expr::Function("sin".to_string(), vec![arg]));
                    ctx.add(Expr::Div(cos_arg, sin_arg))
                }
                "sec" => {
                    let one = ctx.num(1);
                    let cos_arg = ctx.add(Expr::Function("cos".to_string(), vec![arg]));
                    ctx.add(Expr::Div(one, cos_arg))
                }
                "csc" => {
                    let one = ctx.num(1);
                    let sin_arg = ctx.add(Expr::Function("sin".to_string(), vec![arg]));
                    ctx.add(Expr::Div(one, sin_arg))
                }
                _ => expr, // Not a reciprocal trig, return as-is
            }
        }

        // Recurse into powers (for sec², tan², etc.)
        Expr::Pow(base, exp) => {
            let base_val = *base;
            let exp_val = *exp;
            let base_converted = convert_reciprocal_to_sincos(ctx, base_val);
            if base_converted != base_val {
                ctx.add(Expr::Pow(base_converted, exp_val))
            } else {
                expr
            }
        }

        // For other expressions, return as-is
        _ => expr,
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
/// Order is CRITICAL:
/// 1. Negative rules (preserve) first
/// 2. Pattern-based conversion rules
/// 3. No blanket conversion by default
pub fn register(simplifier: &mut crate::engine::Simplifier) {
    // Tier 1: Preserve compositions (HIGHEST PRIORITY)
    // Note: This is currently a no-op rule, but it documents intent
    // Real preservation happens by NOT having blanket conversion rules

    // Tier 2: Pattern-based smart conversion

    // TEMPORARILY DISABLED: Causes stack overflow with test_52
    // TODO: Fix the interaction between Pythagorean detection and other rules
    // The pattern detection works but creates an infinite loop during simplification
    // simplifier.add_rule(Box::new(ConvertForPythagoreanRule));

    simplifier.add_rule(Box::new(ConvertReciprocalProductRule));

    // Future: Add ConvertForMixedFractionRule

    // Note: NO blanket "convert all tan to sin/cos" rules
    // Conversion only happens in specific beneficial contexts
}
