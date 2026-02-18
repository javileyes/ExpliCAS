//! poly_gcd_modp and poly_eq_modp REPL functions.
//!
//! Exposes Zippel mod-p GCD to REPL for fast polynomial verification.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use cas_formatter::DisplayExpr;
use cas_math::gcd_zippel_modp::ZippelPreset;
use cas_math::poly_modp_conv::{
    check_poly_equal_modp_expr, compute_gcd_modp_expr_with_options, PolyConvError,
    DEFAULT_PRIME as INTERNAL_DEFAULT_PRIME,
};

const DEFAULT_PRIME: u64 = INTERNAL_DEFAULT_PRIME;

// =============================================================================
// Eager Eval Infrastructure: Strip expand + Common Factor Extraction
// =============================================================================

/// Strip expand() wrappers iteratively (no recursion depth risk).
/// Also strips __hold wrappers.
fn strip_expand_wrapper(ctx: &Context, mut expr: ExprId) -> ExprId {
    loop {
        if let Expr::Function(fn_id, args) = ctx.get(expr) {
            let builtin = ctx.builtin_of(*fn_id);
            if matches!(
                builtin,
                Some(cas_ast::BuiltinFn::Expand | cas_ast::BuiltinFn::Hold)
            ) && args.len() == 1
            {
                expr = args[0];
                continue;
            }
        }
        return expr;
    }
}

/// Collect multiplicative factors with integer exponents from an expression.
/// - Mul(...) is flattened
/// - Pow(base, k) with integer k becomes (base, k)
/// - Neg(x) is unwrapped (handled by returning negative sign separately)
///
/// Returns (is_negative, factors)
fn collect_mul_factors(ctx: &Context, expr: ExprId) -> (bool, Vec<(ExprId, i64)>) {
    let mut factors = Vec::new();

    // Check for top-level Neg
    let (is_neg, actual_expr) = match ctx.get(expr) {
        Expr::Neg(inner) => (true, *inner),
        _ => (false, expr),
    };

    collect_mul_factors_recursive(ctx, actual_expr, 1, &mut factors);
    (is_neg, factors)
}

fn collect_mul_factors_recursive(
    ctx: &Context,
    expr: ExprId,
    mult: i64,
    factors: &mut Vec<(ExprId, i64)>,
) {
    match ctx.get(expr) {
        Expr::Mul(left, right) => {
            collect_mul_factors_recursive(ctx, *left, mult, factors);
            collect_mul_factors_recursive(ctx, *right, mult, factors);
        }
        Expr::Pow(base, exp) => {
            if let Some(k) = get_integer_exponent(ctx, *exp) {
                factors.push((*base, mult * k));
            } else {
                factors.push((expr, mult));
            }
        }
        _ => {
            factors.push((expr, mult));
        }
    }
}

fn get_integer_exponent(ctx: &Context, exp: ExprId) -> Option<i64> {
    match ctx.get(exp) {
        Expr::Number(n) => {
            if n.is_integer() {
                n.to_integer().try_into().ok()
            } else {
                None
            }
        }
        Expr::Neg(inner) => get_integer_exponent(ctx, *inner).map(|k| -k),
        _ => None,
    }
}

/// Extract common multiplicative factors between two expressions.
/// Returns (common_factors, reduced_a, reduced_b) where:
/// - common_factors: Vec of (base, min_exp) pairs
/// - reduced_a, reduced_b: original expressions with common factors removed
///
/// Invariant: a = common * reduced_a, b = common * reduced_b
#[allow(clippy::type_complexity)]
fn extract_common_mul_factors(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
) -> Option<(Vec<(ExprId, i64)>, ExprId, ExprId)> {
    let (_neg_a, fa) = collect_mul_factors(ctx, a);
    let (_neg_b, fb) = collect_mul_factors(ctx, b);

    // Build factor maps for intersection
    use std::collections::HashMap;

    // Use DisplayExpr as key (not ideal but works for structural comparison)
    let mut map_a: HashMap<String, (ExprId, i64)> = HashMap::new();
    for (base, exp) in &fa {
        let key = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: *base
            }
        );
        map_a
            .entry(key)
            .and_modify(|e| e.1 += exp)
            .or_insert((*base, *exp));
    }

    let mut map_b: HashMap<String, (ExprId, i64)> = HashMap::new();
    for (base, exp) in &fb {
        let key = format!(
            "{}",
            DisplayExpr {
                context: ctx,
                id: *base
            }
        );
        map_b
            .entry(key)
            .and_modify(|e| e.1 += exp)
            .or_insert((*base, *exp));
    }

    // Find intersection: common factors with min exponent
    let mut common: Vec<(ExprId, i64)> = Vec::new();
    for (key, (base_a, exp_a)) in &map_a {
        if let Some((_, exp_b)) = map_b.get(key) {
            // Both have this factor - take minimum positive exponent
            let min_exp = (*exp_a).min(*exp_b);
            if min_exp > 0 {
                common.push((*base_a, min_exp));
            }
        }
    }

    // If no common factors found, return None to skip optimization
    if common.is_empty() {
        return None;
    }

    // Build reduced expressions by dividing out common factors
    let common_expr = build_product_from_factors(ctx, &common);

    // For reduced expressions, we divide by common
    // reduced_a = a / common, reduced_b = b / common
    let reduced_a = ctx.add(Expr::Div(a, common_expr));
    let reduced_b = ctx.add(Expr::Div(b, common_expr));

    Some((common, reduced_a, reduced_b))
}

/// Build product expression from factors Vec<(base, exp)>
fn build_product_from_factors(ctx: &mut Context, factors: &[(ExprId, i64)]) -> ExprId {
    if factors.is_empty() {
        return ctx.num(1);
    }

    let mut result: Option<ExprId> = None;

    for &(base, exp) in factors {
        let term = if exp == 1 {
            base
        } else {
            let exp_expr = ctx.num(exp);
            ctx.add(Expr::Pow(base, exp_expr))
        };

        result = Some(match result {
            None => term,
            Some(acc) => ctx.add(Expr::Mul(acc, term)),
        });
    }

    result.unwrap_or_else(|| ctx.num(1))
}

/// Eager evaluation of poly_gcd_modp with factor extraction optimization.
///
/// This function:
/// 1. Strips expand() wrappers from arguments
/// 2. Extracts common multiplicative factors (gcd(a*g, b*g) = g * gcd(a,b))
/// 3. Computes GCD on reduced polynomials (much smaller)
/// 4. Returns common * gcd_result
pub fn compute_gcd_modp_with_factor_extraction(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
) -> Option<ExprId> {
    // Step 1: Strip expand() and __hold wrappers (NO symbolic expansion!)
    let a0 = strip_expand_wrapper(ctx, a);
    let b0 = strip_expand_wrapper(ctx, b);

    // Step 2: Try to extract common factors
    if let Some((common, ra, rb)) = extract_common_mul_factors(ctx, a0, b0) {
        // We have common factors! Compute gcd on reduced expressions
        // IMPORTANT: Do NOT expand - compute_gcd_modp uses MultiPoly which handles Pow directly
        // Just strip wrappers from reduced expressions too
        let ra_stripped = strip_expand_wrapper(ctx, ra);
        let rb_stripped = strip_expand_wrapper(ctx, rb);

        // Compute gcd on reduced polynomials (MultiPoly handles Pow natively)
        match compute_gcd_modp_with_options(
            ctx,
            ra_stripped,
            rb_stripped,
            DEFAULT_PRIME,
            None,
            None,
        ) {
            Ok(gcd_rest) => {
                // Reconstruct: common * gcd_rest
                let common_expr = build_product_from_factors(ctx, &common);

                // If gcd_rest is 1, just return common
                if let Expr::Number(n) = ctx.get(gcd_rest) {
                    use num_traits::One;
                    if n.is_one() {
                        // Wrap in __hold
                        return Some(cas_ast::hold::wrap_hold(ctx, common_expr));
                    }
                }

                // Otherwise return common * gcd_rest
                let result = ctx.add(Expr::Mul(common_expr, gcd_rest));
                return Some(cas_ast::hold::wrap_hold(ctx, result));
            }
            Err(_) => {
                // Fall through to direct computation
            }
        }
    }

    // Step 3: No common factors or extraction failed - try direct GCD (no expand!)
    // The MultiPoly converter handles Pow(base, n) natively via pow() method
    match compute_gcd_modp_with_options(ctx, a0, b0, DEFAULT_PRIME, None, None) {
        Ok(gcd_expr) => Some(cas_ast::hold::wrap_hold(ctx, gcd_expr)),
        Err(_) => None,
    }
}

/// Eager evaluation pass for poly_gcd_modp calls.
///
/// This function traverses the expression tree TOP-DOWN and evaluates
/// poly_gcd_modp calls BEFORE the normal simplification pipeline.
///
/// CRITICAL: When we find poly_gcd_modp, we do NOT descend into its children.
/// This prevents the expensive symbolic expansion of huge arguments.
pub fn eager_eval_poly_gcd_calls(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> (ExprId, Vec<crate::Step>) {
    let mut steps = Vec::new();
    let result = eager_eval_recursive(ctx, expr, &mut steps, collect_steps);
    (result, steps)
}

fn eager_eval_recursive(
    ctx: &mut Context,
    expr: ExprId,
    steps: &mut Vec<crate::Step>,
    collect_steps: bool,
) -> ExprId {
    // Check if this is poly_gcd_modp - if so, evaluate and STOP descent
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let (fn_id, args) = (*fn_id, args.clone());
        let name = ctx.sym_name(fn_id);
        if (name == "poly_gcd_modp" || name == "pgcdp") && args.len() >= 2 {
            if let Some(result) = compute_gcd_modp_with_factor_extraction(ctx, args[0], args[1]) {
                // Create step for the evaluation
                if collect_steps {
                    steps.push(crate::Step::new(
                        "Eager eval poly_gcd_modp (bypass simplifier)",
                        "Polynomial GCD mod p",
                        expr,
                        result,
                        Vec::new(),
                        Some(ctx),
                    ));
                }
                return result;
            }
        }

        // For other functions, recurse into children
        let new_args: Vec<ExprId> = args
            .iter()
            .map(|&arg| eager_eval_recursive(ctx, arg, steps, collect_steps))
            .collect();

        // Check if any arg changed
        if new_args
            .iter()
            .zip(args.iter())
            .any(|(new, old)| new != old)
        {
            return ctx.add(Expr::Function(fn_id, new_args));
        }
        return expr;
    }

    // Recurse into children for other expression types
    enum Recurse {
        Binary(ExprId, ExprId, u8), // 0=Add, 1=Sub, 2=Mul, 3=Div, 4=Pow
        Unary(ExprId, u8),          // 0=Neg, 1=Hold
        Leaf,
    }
    let recurse = match ctx.get(expr) {
        Expr::Add(l, r) => Recurse::Binary(*l, *r, 0),
        Expr::Sub(l, r) => Recurse::Binary(*l, *r, 1),
        Expr::Mul(l, r) => Recurse::Binary(*l, *r, 2),
        Expr::Div(l, r) => Recurse::Binary(*l, *r, 3),
        Expr::Pow(b, e) => Recurse::Binary(*b, *e, 4),
        Expr::Neg(e) => Recurse::Unary(*e, 0),
        Expr::Hold(inner) => Recurse::Unary(*inner, 1),
        _ => Recurse::Leaf,
    };
    match recurse {
        Recurse::Binary(l, r, op) => {
            let nl = eager_eval_recursive(ctx, l, steps, collect_steps);
            let nr = eager_eval_recursive(ctx, r, steps, collect_steps);
            if nl != l || nr != r {
                match op {
                    0 => ctx.add(Expr::Add(nl, nr)),
                    1 => ctx.add(Expr::Sub(nl, nr)),
                    2 => ctx.add(Expr::Mul(nl, nr)),
                    3 => ctx.add(Expr::Div(nl, nr)),
                    _ => ctx.add(Expr::Pow(nl, nr)),
                }
            } else {
                expr
            }
        }
        Recurse::Unary(inner, op) => {
            let ni = eager_eval_recursive(ctx, inner, steps, collect_steps);
            if ni != inner {
                match op {
                    0 => ctx.add(Expr::Neg(ni)),
                    _ => ctx.add(Expr::Hold(ni)),
                }
            } else {
                expr
            }
        }
        Recurse::Leaf => expr,
    }
}

// Rule for poly_gcd_modp(a, b [, p]) function.
// Computes Zippel GCD of two polynomial expressions mod p.
define_rule!(
    PolyGcdModpRule,
    "Polynomial GCD mod p",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    priority: 200, // High priority to evaluate early
    |ctx, expr| {
        let (fn_id, args) = if let Expr::Function(fn_id, args) = ctx.get(expr) {
            (*fn_id, args.clone())
        } else {
            return None;
        };
        {
            let name = ctx.sym_name(fn_id).to_string();
            let is_gcd_modp = name == "poly_gcd_modp" || name == "pgcdp";

            if is_gcd_modp && args.len() >= 2 && args.len() <= 4 {
                // Use eager eval with factor extraction
                if let Some(result) = compute_gcd_modp_with_factor_extraction(ctx, args[0], args[1])
                {
                    return Some(Rewrite::simple(
                        result,
                        format!(
                            "poly_gcd_modp({}, {}) [eager eval + factor extraction]",
                            DisplayExpr {
                                context: ctx,
                                id: args[0]
                            },
                            DisplayExpr {
                                context: ctx,
                                id: args[1]
                            }
                        ),
                    ));
                }

                // Fallback: try with explicit args (for extra options)
                let a = strip_expand_wrapper(ctx, args[0]);
                let b = strip_expand_wrapper(ctx, args[1]);

                // Parse remaining args: main_var (usize) and/or preset (string)
                let mut main_var: Option<usize> = None;
                let mut preset_str: Option<String> = None;

                for &arg in args.iter().skip(2) {
                    // Try to extract as usize (small number = main_var)
                    if let Some(v) = extract_usize(ctx, arg) {
                        if v <= 64 {
                            main_var = Some(v);
                            continue;
                        }
                    }
                    // Try to extract as string (preset name)
                    if let Some(s) = extract_string(ctx, arg) {
                        preset_str = Some(s);
                    }
                }

                // Parse preset from string if provided
                let preset = preset_str.as_deref().and_then(ZippelPreset::parse);
                match compute_gcd_modp_with_options(ctx, a, b, DEFAULT_PRIME, main_var, preset) {
                    Ok(gcd_expr) => {
                        // Wrap in __hold to prevent further simplification
                        let held = cas_ast::hold::wrap_hold(ctx, gcd_expr);

                        return Some(Rewrite::simple(
                            held,
                            format!(
                                "poly_gcd_modp({}, {})",
                                DisplayExpr {
                                    context: ctx,
                                    id: a
                                },
                                DisplayExpr {
                                    context: ctx,
                                    id: b
                                }
                            ),
                        ));
                    }
                    Err(e) => {
                        // Return error as message (don't crash)
                        eprintln!("poly_gcd_modp error: {}", e);
                        return None;
                    }
                }
            }
        }

        None
    }
);

// Rule for poly_eq_modp(a, b [, p]) function.
// Returns 1 if polynomials are equal mod p, 0 otherwise.
define_rule!(
    PolyEqModpRule,
    "Polynomial equality mod p",
    Some(crate::target_kind::TargetKindSet::FUNCTION),
    PhaseMask::CORE | PhaseMask::TRANSFORM,
    priority: 200,
    |ctx, expr| {
        let (fn_id, args) = if let Expr::Function(fn_id, args) = ctx.get(expr) {
            (*fn_id, args.clone())
        } else {
            return None;
        };
        {
            let name = ctx.sym_name(fn_id).to_string();
            let is_eq_modp = name == "poly_eq_modp" || name == "peqp";

            if is_eq_modp && (args.len() == 2 || args.len() == 3) {
                let a = args[0];
                let b = args[1];
                let p = if args.len() == 3 {
                    extract_prime(ctx, args[2]).unwrap_or(DEFAULT_PRIME)
                } else {
                    DEFAULT_PRIME
                };

                match check_poly_equal_modp(ctx, a, b, p) {
                    Ok(equal) => {
                        let result = if equal { ctx.num(1) } else { ctx.num(0) };

                        return Some(Rewrite::simple(
                            result,
                            format!(
                                "poly_eq_modp({}, {}) = {}",
                                DisplayExpr {
                                    context: ctx,
                                    id: a
                                },
                                DisplayExpr {
                                    context: ctx,
                                    id: b
                                },
                                if equal { "true" } else { "false" }
                            ),
                        ));
                    }
                    Err(e) => {
                        eprintln!("poly_eq_modp error: {}", e);
                        return None;
                    }
                }
            }
        }

        None
    }
);

/// Extract prime from expression (must be integer)
fn extract_prime(ctx: &Context, expr: ExprId) -> Option<u64> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            use num_traits::ToPrimitive;
            return n.to_integer().to_u64();
        }
    }
    None
}

/// Extract usize from expression (must be non-negative integer)
fn extract_usize(ctx: &Context, expr: ExprId) -> Option<usize> {
    if let Expr::Number(n) = ctx.get(expr) {
        if n.is_integer() {
            use num_traits::ToPrimitive;
            return n.to_integer().to_usize();
        }
    }
    None
}

/// Extract string from expression (Variable as string literal)
fn extract_string(ctx: &Context, expr: ExprId) -> Option<String> {
    // Check for Variable (used as string literal, e.g., "mm_gcd")
    if let Expr::Variable(sym_id) = ctx.get(expr) {
        return Some(ctx.sym_name(*sym_id).to_string());
    }
    None
}

/// Compute GCD mod p and return as Expr (public for unified API)
pub fn compute_gcd_modp_with_options(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
    p: u64,
    main_var: Option<usize>,
    preset: Option<ZippelPreset>,
) -> Result<ExprId, PolyConvError> {
    compute_gcd_modp_expr_with_options(ctx, a, b, p, main_var, preset)
}

/// Check if two polynomials are equal mod p
fn check_poly_equal_modp(
    ctx: &Context,
    a: ExprId,
    b: ExprId,
    p: u64,
) -> Result<bool, PolyConvError> {
    check_poly_equal_modp_expr(ctx, a, b, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_poly_eq_modp_same() {
        let mut ctx = Context::new();
        let a = parse("x + 1", &mut ctx).unwrap();
        let b = parse("1 + x", &mut ctx).unwrap();

        let result = check_poly_equal_modp(&ctx, a, b, DEFAULT_PRIME).unwrap();
        assert!(result);
    }

    #[test]
    fn test_poly_eq_modp_different() {
        let mut ctx = Context::new();
        let a = parse("x + 1", &mut ctx).unwrap();
        let b = parse("x + 2", &mut ctx).unwrap();

        let result = check_poly_equal_modp(&ctx, a, b, DEFAULT_PRIME).unwrap();
        assert!(!result);
    }
}
