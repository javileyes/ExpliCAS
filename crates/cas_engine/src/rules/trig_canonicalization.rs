use crate::define_rule;
use crate::helpers::is_one;
use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr};
use cas_math::trig_canonicalization_support::{
    check_reciprocal_pair, convert_trig_to_sincos, is_function_squared, is_mixed_trig_fraction,
    is_trig_of_inverse_trig,
};

// ==================== Sophisticated Context-Aware Canonicalization ====================
// STRATEGY: Only convert when it demonstrably helps simplification
// Three-tier approach:
// 1. Never convert: compositions like tan(arctan(x))
// 2. Always convert: known patterns like sec²-tan², mixed fractions
// 3. Selective: on-demand for complex cases

// ============================== Function Name Canonicalization ==============================

// Canonicalize trig function names: asin→arcsin, acos→arccos, atan→arctan
// This prevents bugs from mixed naming like "arccos(x) - acos(x)" not simplifying
define_rule!(
    TrigFunctionNameCanonicalizationRule,
    "Canonicalize Trig Function Names",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    crate::phase::PhaseMask::CORE | crate::phase::PhaseMask::POST,
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let args_clone = args.clone();

            let canonical_name = match ctx.builtin_of(*fn_id) {
                // Short forms → Canonical long forms
                Some(BuiltinFn::Asin) => Some("arcsin"),
                Some(BuiltinFn::Acos) => Some("arccos"),
                Some(BuiltinFn::Atan) => Some("arctan"),
                Some(BuiltinFn::Asec) => Some("arcsec"),
                Some(BuiltinFn::Acsc) => Some("arccsc"),
                Some(BuiltinFn::Acot) => Some("arccot"),

                // Already canonical or not an inverse trig function - no change
                _ => None,
            };

            if let Some(canonical) = canonical_name {
                let old_name = ctx.builtin_of(*fn_id).unwrap().name();
                let new_fn = ctx.call(canonical, args_clone);
                return Some(Rewrite::new(new_fn).desc_lazy(|| format!("{} → {}", old_name, canonical)));
            }
        }
        None
    }
);

// ==================== Tier 1: Preserve Compositions (Negative Rule) ====================

// NEVER convert reciprocal trig if it's a composition with inverse trig
// This preserves tan(arctan(x)) → x simplifications
// Priority: HIGHEST (register first)
define_rule!(
    PreserveCompositionRule,
    "Preserve trig-inverse compositions",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
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
    Some(crate::target_kind::TargetKindSet::SUB),
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
    Some(crate::target_kind::TargetKindSet::SUB),
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
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| {
        if let Expr::Add(l, r) = ctx.get(expr) {
            let l_val = *l;
            let r_val = *r;

            // Check both orders: 1 + tan² and tan² + 1
            if is_one(ctx, l_val) {
                if let Some(tan_arg) = is_function_squared(ctx, r_val, "tan") {
                    let two = ctx.num(2);
                    let sec_expr = ctx.call_builtin(cas_ast::BuiltinFn::Sec, vec![tan_arg]);
                    let sec_squared = ctx.add(Expr::Pow(sec_expr, two));
                    return Some(Rewrite::new(sec_squared).desc("1 + tan²(x) = sec²(x)"));
                }
            } else if is_one(ctx, r_val) {
                if let Some(tan_arg) = is_function_squared(ctx, l_val, "tan") {
                    let two = ctx.num(2);
                    let sec_expr = ctx.call_builtin(cas_ast::BuiltinFn::Sec, vec![tan_arg]);
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
    Some(crate::target_kind::TargetKindSet::ADD),
    |ctx, expr| {
        if let Expr::Add(l, r) = ctx.get(expr) {
            let l_val = *l;
            let r_val = *r;

            if is_one(ctx, l_val) {
                if let Some(cot_arg) = is_function_squared(ctx, r_val, "cot") {
                    let two = ctx.num(2);
                    let csc_expr = ctx.call_builtin(cas_ast::BuiltinFn::Csc, vec![cot_arg]);
                    let csc_squared = ctx.add(Expr::Pow(csc_expr, two));
                    return Some(Rewrite::new(csc_squared).desc("1 + cot²(x) = csc²(x)"));
                }
            } else if is_one(ctx, r_val) {
                if let Some(cot_arg) = is_function_squared(ctx, l_val, "cot") {
                    let two = ctx.num(2);
                    let csc_expr = ctx.call_builtin(cas_ast::BuiltinFn::Csc, vec![cot_arg]);
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
    Some(crate::target_kind::TargetKindSet::SUB),
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
    Some(crate::target_kind::TargetKindSet::SUB),
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

// Convert reciprocal products like tan(x)*cot(x) → 1
define_rule!(
    ConvertReciprocalProductRule,
    "Simplify reciprocal trig products",
    Some(crate::target_kind::TargetKindSet::MUL),
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

// ==================== Phase 4: Mixed Fraction Conversion ====================

// Convert mixed trig fractions to sin/cos for better algebraic simplification
define_rule!(
    ConvertForMixedFractionRule,
    "Convert Mixed Trig Fraction to sin/cos",
    Some(crate::target_kind::TargetKindSet::DIV),
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
