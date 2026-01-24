use crate::build::mul2_raw;
use crate::define_rule;
use crate::helpers::is_one;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};

// ==================== Sophisticated Context-Aware Canonicalization ====================
// STRATEGY: Only convert when it demonstrably helps simplification
// Three-tier approach:
// 1. Never convert: compositions like tan(arctan(x))
// 2. Always convert: known patterns like sec²-tan², mixed fractions
// 3. Selective: on-demand for complex cases

// ==================== Helper Functions for Pattern Detection ====================

// is_one is now imported from crate::helpers

// Check if expression is a number equal to 2
fn is_two(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(n) => *n == num_rational::Ratio::from_integer(2.into()),
        _ => false,
    }
}

// Check if function name is a reciprocal trig function
fn is_reciprocal_trig_name(name: &str) -> bool {
    matches!(name, "tan" | "cot" | "sec" | "csc")
}

// Check if function name is an inverse trig function
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

// Check if expression is a composition like tan(arctan(x))
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

// ============================== Function Name Canonicalization ==============================

// Canonicalize trig function names: asin→arcsin, acos→arccos, atan→arctan
// This prevents bugs from mixed naming like "arccos(x) - acos(x)" not simplifying
define_rule!(
    TrigFunctionNameCanonicalizationRule,
    "Canonicalize Trig Function Names",
    Some(vec!["Function"]),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        if let Expr::Function(name, args) = ctx.get(expr) {
            // Clone name and args to avoid borrow issues
            let name_clone = name.clone();
            let args_clone = args.clone();

            let canonical_name = match name_clone.as_str() {
                // Short forms → Canonical long forms
                "asin" => Some("arcsin"),
                "acos" => Some("arccos"),
                "atan" => Some("arctan"),
                "asec" => Some("arcsec"),
                "acsc" => Some("arccsc"),
                "acot" => Some("arccot"),

                // Already canonical - no change needed
                "arcsin" | "arccos" | "arctan" | "arcsec" | "arccsc" | "arccot" => None,

                // Not an inverse trig function - skip
                _ => None,
            };

            if let Some(canonical) = canonical_name {
                let new_fn = ctx.add(Expr::Function(canonical.to_string(), args_clone));
                return Some(Rewrite::new(new_fn).desc(format!("{} → {}", name_clone, canonical)));
            }
        }
        None
    }
);

// ==================== Phase 4: Mixed Fraction Helpers ====================

use std::collections::HashSet;

// Recursively collect all trig function names in an expression
fn collect_trig_recursive(ctx: &Context, expr: ExprId, funcs: &mut HashSet<String>) {
    match ctx.get(expr) {
        Expr::Function(name, args) => {
            // Add if it's a trig function
            if matches!(name.as_str(), "sin" | "cos" | "tan" | "cot" | "sec" | "csc") {
                funcs.insert(name.clone());
            }
            // Recurse into arguments
            for &arg in args {
                collect_trig_recursive(ctx, arg, funcs);
            }
        }
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) => {
            collect_trig_recursive(ctx, *l, funcs);
            collect_trig_recursive(ctx, *r, funcs);
        }
        Expr::Pow(base, exp) => {
            collect_trig_recursive(ctx, *base, funcs);
            collect_trig_recursive(ctx, *exp, funcs);
        }
        Expr::Neg(inner) => {
            collect_trig_recursive(ctx, *inner, funcs);
        }
        _ => {}
    }
}

// Collect all trig function names in expression
fn collect_trig_functions(ctx: &Context, expr: ExprId) -> HashSet<String> {
    let mut funcs = HashSet::new();
    collect_trig_recursive(ctx, expr, &mut funcs);
    funcs
}

// Check if has multiple different trig function types
fn has_multiple_trig_types(funcs: &HashSet<String>) -> bool {
    funcs.len() >= 2
}

// Check if expression is f²(x) where f is any trig function
// Returns the argument of the trig function if it matches, otherwise None.
fn is_any_trig_function_squared(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) => {
            if is_two(ctx, *exp) {
                match ctx.get(*base) {
                    Expr::Function(name, args) if args.len() == 1 => {
                        // Check if it's a trig function (sin, cos, tan, cot, sec, csc)
                        if matches!(name.as_str(), "sin" | "cos" | "tan" | "cot" | "sec" | "csc") {
                            return Some(args[0]);
                        }
                    }
                    _ => {}
                }
            }
            None
        }
        _ => None,
    }
}

// Check if expression is a Pythagorean-style pattern (f² ± g² or 1 ± f²)
// These should be handled by direct Pythagorean rules, not converted
fn is_pythagorean_style(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Add(l, r) | Expr::Sub(l, r) => {
            let l_is_squared = is_any_trig_function_squared(ctx, *l).is_some();
            let r_is_squared = is_any_trig_function_squared(ctx, *r).is_some();
            let l_is_one = is_one(ctx, *l);
            let r_is_one = is_one(ctx, *r);

            // Patterns: f²±g², 1±f², f²±1
            // Simplified: r_is_squared && (l_is_one || l_is_squared), or l_is_squared && r_is_one
            (r_is_squared && (l_is_one || l_is_squared)) || (l_is_squared && r_is_one)
        }
        _ => false,
    }
}

// Check if should trigger mixed fraction conversion
fn is_mixed_trig_fraction(ctx: &Context, num: ExprId, den: ExprId) -> bool {
    // Don't convert if numerator or denominator is a Pythagorean pattern
    // Those are handled better by direct Pythagorean identity rules
    if is_pythagorean_style(ctx, num) || is_pythagorean_style(ctx, den) {
        return false;
    }

    let num_funcs = collect_trig_functions(ctx, num);
    let den_funcs = collect_trig_functions(ctx, den);

    // Must have some trig functions
    if num_funcs.is_empty() && den_funcs.is_empty() {
        return false;
    }

    // Check for mixed types in numerator OR denominator
    let num_has_mixed = has_multiple_trig_types(&num_funcs);
    let den_has_mixed = has_multiple_trig_types(&den_funcs);

    // Check if there's any reciprocal trig
    let has_reciprocal = num_funcs.iter().any(|n| is_reciprocal_trig_name(n))
        || den_funcs.iter().any(|n| is_reciprocal_trig_name(n));

    (num_has_mixed || den_has_mixed) && has_reciprocal
}

// Recursively convert trig functions to sin/cos
fn convert_trig_to_sincos(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr_data = ctx.get(expr).clone();
    match expr_data {
        Expr::Function(name, args) if args.len() == 1 => {
            let arg = args[0];
            let converted_arg = convert_trig_to_sincos(ctx, arg);

            match name.as_str() {
                "tan" => {
                    // tan(x) → sin(x)/cos(x)
                    let sin_x = ctx.call("sin", vec![converted_arg]);
                    let cos_x = ctx.call("cos", vec![converted_arg]);
                    ctx.add(Expr::Div(sin_x, cos_x))
                }
                "cot" => {
                    // cot(x) → cos(x)/sin(x)
                    let sin_x = ctx.call("sin", vec![converted_arg]);
                    let cos_x = ctx.call("cos", vec![converted_arg]);
                    ctx.add(Expr::Div(cos_x, sin_x))
                }
                "sec" => {
                    // sec(x) → 1/cos(x)
                    let one = ctx.num(1);
                    let cos_x = ctx.call("cos", vec![converted_arg]);
                    ctx.add(Expr::Div(one, cos_x))
                }
                "csc" => {
                    // csc(x) → 1/sin(x)
                    let one = ctx.num(1);
                    let sin_x = ctx.call("sin", vec![converted_arg]);
                    ctx.add(Expr::Div(one, sin_x))
                }
                _ => {
                    // Keep sin/cos as-is, but with converted arg
                    ctx.add(Expr::Function(name, vec![converted_arg]))
                }
            }
        }
        Expr::Add(l, r) => {
            let new_l = convert_trig_to_sincos(ctx, l);
            let new_r = convert_trig_to_sincos(ctx, r);
            ctx.add(Expr::Add(new_l, new_r))
        }
        Expr::Sub(l, r) => {
            let new_l = convert_trig_to_sincos(ctx, l);
            let new_r = convert_trig_to_sincos(ctx, r);
            ctx.add(Expr::Sub(new_l, new_r))
        }
        Expr::Mul(l, r) => {
            let new_l = convert_trig_to_sincos(ctx, l);
            let new_r = convert_trig_to_sincos(ctx, r);
            mul2_raw(ctx, new_l, new_r)
        }
        Expr::Div(l, r) => {
            let new_l = convert_trig_to_sincos(ctx, l);
            let new_r = convert_trig_to_sincos(ctx, r);
            ctx.add(Expr::Div(new_l, new_r))
        }
        Expr::Pow(base, exp) => {
            let new_base = convert_trig_to_sincos(ctx, base);
            // Don't recurse into exponent
            ctx.add(Expr::Pow(new_base, exp))
        }
        Expr::Neg(inner) => {
            let new_inner = convert_trig_to_sincos(ctx, inner);
            ctx.add(Expr::Neg(new_inner))
        }
        _ => expr, // Return as-is for other types
    }
}

// ==================== End Phase 4 Helpers ====================

// ==================== Tier 1: Preserve Compositions (Negative Rule) ====================

// NEVER convert reciprocal trig if it's a composition with inverse trig
// This preserves tan(arctan(x)) → x simplifications
// Priority: HIGHEST (register first)
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

// sec²(x) - tan²(x) → 1
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
                    return Some(Rewrite::new(ctx.num(1)).desc("sec²(x) - tan²(x) = 1"));
                }
            }
        }
        None
    }
);

// csc²(x) - cot²(x) → 1
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
                    return Some(Rewrite::new(ctx.num(1)).desc("csc²(x) - cot²(x) = 1"));
                }
            }
        }
        None
    }
);

// 1 + tan²(x) → sec²(x)
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
                    let sec_expr = ctx.call("sec", vec![tan_arg]);
                    let sec_squared = ctx.add(Expr::Pow(sec_expr, two));
                    return Some(Rewrite::new(sec_squared).desc("1 + tan²(x) = sec²(x)"));
                }
            } else if is_one(ctx, r_val) {
                if let Some(tan_arg) = is_function_squared(ctx, l_val, "tan") {
                    let two = ctx.num(2);
                    let sec_expr = ctx.call("sec", vec![tan_arg]);
                    let sec_squared = ctx.add(Expr::Pow(sec_expr, two));
                    return Some(Rewrite::new(sec_squared).desc("1 + tan²(x) = sec²(x)"));
                }
            }
        }
        None
    }
);

// 1 + cot²(x) → csc²(x)
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
                    let csc_expr = ctx.call("csc", vec![cot_arg]);
                    let csc_squared = ctx.add(Expr::Pow(csc_expr, two));
                    return Some(Rewrite::new(csc_squared).desc("1 + cot²(x) = csc²(x)"));
                }
            } else if is_one(ctx, r_val) {
                if let Some(cot_arg) = is_function_squared(ctx, l_val, "cot") {
                    let two = ctx.num(2);
                    let csc_expr = ctx.call("csc", vec![cot_arg]);
                    let csc_squared = ctx.add(Expr::Pow(csc_expr, two));
                    return Some(Rewrite::new(csc_squared).desc("1 + cot²(x) = csc²(x)"));
                }
            }
        }
        None
    }
);

// ==================== Pythagorean Identity Variants with Constants ====================

// sec²(x) - tan²(x) - 1 → 0
// This handles the variant where we have the full identity minus 1
define_rule!(
    SecTanMinusOneIdentityRule,
    "sec²(x) - tan²(x) - 1 = 0",
    Some(vec!["Sub"]),
    |ctx, expr| {
        // Pattern: (sec² - tan²) - 1
        // We know sec² - tan² = 1, so (sec² - tan²) - 1 = 0
        if let Expr::Sub(left, right) = ctx.get(expr) {
            let left_val = *left;
            let right_val = *right;

            // Check if right is 1
            if is_one(ctx, right_val) {
                // Check if left is  sec² - tan² pattern
                if let Expr::Sub(ll, lr) = ctx.get(left_val) {
                    let ll_val = *ll;
                    let lr_val = *lr;

                    if let (Some(sec_arg), Some(tan_arg)) = (
                        is_function_squared(ctx, ll_val, "sec"),
                        is_function_squared(ctx, lr_val, "tan"),
                    ) {
                        if sec_arg == tan_arg {
                            return Some(
                                Rewrite::new(ctx.num(0)).desc("sec²(x) - tan²(x) - 1 = 0"),
                            );
                        }
                    }
                }
            }
        }
        None
    }
);

// csc²(x) - cot²(x) - 1 → 0
define_rule!(
    CscCotMinusOneIdentityRule,
    "csc²(x) - cot²(x) - 1 = 0",
    Some(vec!["Sub"]),
    |ctx, expr| {
        if let Expr::Sub(left, right) = ctx.get(expr) {
            let left_val = *left;
            let right_val = *right;

            if is_one(ctx, right_val) {
                if let Expr::Sub(ll, lr) = ctx.get(left_val) {
                    let ll_val = *ll;
                    let lr_val = *lr;

                    if let (Some(csc_arg), Some(cot_arg)) = (
                        is_function_squared(ctx, ll_val, "csc"),
                        is_function_squared(ctx, lr_val, "cot"),
                    ) {
                        if csc_arg == cot_arg {
                            return Some(
                                Rewrite::new(ctx.num(0)).desc("csc²(x) - cot²(x) - 1 = 0"),
                            );
                        }
                    }
                }
            }
        }
        None
    }
);

// Helper: Check if expr is f²(arg) for a specific function name
// Returns Some(arg) if match, None otherwise
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

// Convert reciprocal products like tan(x)*cot(x) → 1
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
                return Some(Rewrite::new(ctx.num(1)).desc("Reciprocal trig product = 1"));
            }
        }
        None
    }
);

// Check if two expressions are reciprocal trig functions with same argument
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

// ==================== Phase 4: Mixed Fraction Conversion ====================

// Convert mixed trig fractions to sin/cos for better algebraic simplification
define_rule!(
    ConvertForMixedFractionRule,
    "Convert Mixed Trig Fraction to sin/cos",
    Some(vec!["Div"]),
    |ctx, expr| {
        if let Expr::Div(num, den) = ctx.get(expr) {
            let num = *num;
            let den = *den;

            // Check if this is a mixed trig fraction
            if is_mixed_trig_fraction(ctx, num, den) {
                // Convert both numerator and denominator to sin/cos
                let new_num = convert_trig_to_sincos(ctx, num);
                let new_den = convert_trig_to_sincos(ctx, den);

                let result = ctx.add(Expr::Div(new_num, new_den));

                return Some(Rewrite::new(result).desc("Convert mixed trig fraction to sin/cos"));
            }
        }
        None
    }
);

// ==================== Registration ====================

// Register ONLY direct Pythagorean identity rules
// CRITICAL: Must be called BEFORE any conversion rules to preserve patterns
// sec²-tan²-1 must match BEFORE tan² becomes sin²/cos²
pub fn register_pythagorean_identities(simplifier: &mut crate::engine::Simplifier) {
    // These are the HIGHEST PRIORITY rules that must fire first
    simplifier.add_rule(Box::new(SecTanPythagoreanRule));
    simplifier.add_rule(Box::new(CscCotPythagoreanRule));
    simplifier.add_rule(Box::new(TanToSecPythagoreanRule));
    simplifier.add_rule(Box::new(CotToCscPythagoreanRule));

    // Pythagorean variants with constants
    simplifier.add_rule(Box::new(SecTanMinusOneIdentityRule));
    simplifier.add_rule(Box::new(CscCotMinusOneIdentityRule));
}

// Register sophisticated canonicalization rules
// CRITICAL: These rules are applied AFTER compositions resolve
// so that tan(arctan(x)) → x happens before any conversion attempts
pub fn register(simplifier: &mut crate::engine::Simplifier) {
    // Function name canonicalization - MUST run first
    simplifier.add_rule(Box::new(TrigFunctionNameCanonicalizationRule));

    // Reciprocal product simplification
    simplifier.add_rule(Box::new(ConvertReciprocalProductRule));

    // Mixed fraction conversion
    simplifier.add_rule(Box::new(ConvertForMixedFractionRule));
}
