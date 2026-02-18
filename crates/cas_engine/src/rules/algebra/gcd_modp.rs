//! poly_gcd_modp and poly_eq_modp REPL functions.
//!
//! Exposes Zippel mod-p GCD to REPL for fast polynomial verification.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use cas_formatter::DisplayExpr;
use cas_math::gcd_zippel_modp::ZippelPreset;
use cas_math::poly_gcd_structural::poly_gcd_structural;
use cas_math::poly_modp_conv::{
    check_poly_equal_modp_expr, compute_gcd_modp_expr_with_options,
    DEFAULT_PRIME as INTERNAL_DEFAULT_PRIME,
};
use num_traits::One;

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

/// Eager evaluation of poly_gcd_modp with factor extraction optimization.
///
/// This function:
/// 1. Strips expand() wrappers from arguments
/// 2. Extracts common multiplicative factors (gcd(a*g, b*g) = g * gcd(a,b))
/// 3. Computes GCD on reduced polynomials (much smaller)
/// 4. Returns common * gcd_result
fn compute_gcd_modp_with_factor_extraction(
    ctx: &mut Context,
    a: ExprId,
    b: ExprId,
) -> Option<ExprId> {
    // Step 1: Strip expand() and __hold wrappers (NO symbolic expansion!)
    let a0 = strip_expand_wrapper(ctx, a);
    let b0 = strip_expand_wrapper(ctx, b);

    // Step 2: Try to extract common factors structurally
    let common_expr = poly_gcd_structural(ctx, a0, b0);
    let has_common_factor = !matches!(ctx.get(common_expr), Expr::Number(n) if n.is_one());

    if has_common_factor {
        let ra = ctx.add(Expr::Div(a0, common_expr));
        let rb = ctx.add(Expr::Div(b0, common_expr));

        // IMPORTANT: Do NOT expand - compute_gcd_modp uses MultiPoly which handles Pow directly
        let ra_stripped = strip_expand_wrapper(ctx, ra);
        let rb_stripped = strip_expand_wrapper(ctx, rb);

        // Compute gcd on reduced polynomials (MultiPoly handles Pow natively).
        match compute_gcd_modp_expr_with_options(
            ctx,
            ra_stripped,
            rb_stripped,
            DEFAULT_PRIME,
            None,
            None,
        ) {
            Ok(gcd_rest) => {
                // If gcd_rest is 1, just return common
                if matches!(ctx.get(gcd_rest), Expr::Number(n) if n.is_one()) {
                    return Some(cas_ast::hold::wrap_hold(ctx, common_expr));
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
    match compute_gcd_modp_expr_with_options(ctx, a0, b0, DEFAULT_PRIME, None, None) {
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
pub(crate) fn eager_eval_poly_gcd_calls(
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
                match compute_gcd_modp_expr_with_options(
                    ctx,
                    a,
                    b,
                    DEFAULT_PRIME,
                    main_var,
                    preset,
                ) {
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

                match check_poly_equal_modp_expr(ctx, a, b, p) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_poly_eq_modp_same() {
        let mut ctx = Context::new();
        let a = parse("x + 1", &mut ctx).unwrap();
        let b = parse("1 + x", &mut ctx).unwrap();

        let result = check_poly_equal_modp_expr(&ctx, a, b, DEFAULT_PRIME).unwrap();
        assert!(result);
    }

    #[test]
    fn test_poly_eq_modp_different() {
        let mut ctx = Context::new();
        let a = parse("x + 1", &mut ctx).unwrap();
        let b = parse("x + 2", &mut ctx).unwrap();

        let result = check_poly_equal_modp_expr(&ctx, a, b, DEFAULT_PRIME).unwrap();
        assert!(!result);
    }
}
