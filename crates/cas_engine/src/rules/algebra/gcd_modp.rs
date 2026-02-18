//! poly_gcd_modp and poly_eq_modp REPL functions.
//!
//! Exposes Zippel mod-p GCD to REPL for fast polynomial verification.

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use cas_formatter::DisplayExpr;
use cas_math::expr_extract::extract_u64_integer;
use cas_math::poly_gcd_mode::parse_modp_options;
use cas_math::poly_modp_conv::{
    check_poly_equal_modp_expr, compute_gcd_modp_expr_with_factor_extraction,
    DEFAULT_PRIME as INTERNAL_DEFAULT_PRIME,
};

const DEFAULT_PRIME: u64 = INTERNAL_DEFAULT_PRIME;

// =============================================================================
// Eager Eval Infrastructure: Strip expand + Common Factor Extraction
// =============================================================================

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
    let gcd =
        compute_gcd_modp_expr_with_factor_extraction(ctx, a, b, DEFAULT_PRIME, None, None).ok()?;
    Some(cas_ast::hold::wrap_hold(ctx, gcd))
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
                // Parse remaining args: preset and/or main_var.
                let (preset, main_var) = parse_modp_options(ctx, &args[2..]);
                match compute_gcd_modp_expr_with_factor_extraction(
                    ctx,
                    args[0],
                    args[1],
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
                                    id: args[0]
                                },
                                DisplayExpr {
                                    context: ctx,
                                    id: args[1]
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
                    extract_u64_integer(ctx, args[2]).unwrap_or(DEFAULT_PRIME)
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
