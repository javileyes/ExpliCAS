//! Safe pre-simplification for limits.
//!
//! This module implements an allowlist-only simplification pipeline
//! that is applied before limit rules. It is designed to be:
//! - Conservative: only structural transforms, no domain assumptions
//! - Auditable: explicit allowlist, no external simplifier dependencies
//! - Depth-limited to prevent stack overflow
//!
//! # Allowed Transforms (Allowlist)
//! - `a + 0 → a`, `0 + a → a`
//! - `a * 1 → a`, `1 * a → a`
//! - `0 * a → 0`, `a * 0 → 0`
//! - `a - 0 → a`
//! - `-(-a) → a`
//! - `a - a → 0` (structural equality only)
//! - `a + (-a) → 0` (structural equality only)
//!
//! # Excluded (even if they seem "safe")
//! - `a/a → 1` (requires `a ≠ 0` assumption)
//! - `a^0 → 1` (0^0 is undefined)
//! - Rationalization
//! - Aggressive expansion

use crate::helpers::{is_one, is_zero};
use crate::Budget;
use crate::CasError;
use cas_ast::{Context, Expr, ExprId};

/// Apply safe pre-simplification to an expression before limit evaluation.
///
/// This is an allowlist-only pipeline that:
/// - Does NOT introduce domain assumptions
/// - Does NOT rationalize
/// - Does NOT expand aggressively
/// - Uses depth limit (500) to prevent stack overflow
///
/// Returns the (possibly simplified) expression, or error on depth exceeded.
pub fn presimplify_safe(
    ctx: &mut Context,
    expr: ExprId,
    _budget: &mut Budget, // Reserved for future budget integration
) -> Result<ExprId, CasError> {
    // Single-pass bottom-up transform with allowlist rules
    presimplify_recursive(ctx, expr, 0)
}

const MAX_DEPTH: usize = 500;

fn presimplify_recursive(
    ctx: &mut Context,
    expr: ExprId,
    depth: usize,
) -> Result<ExprId, CasError> {
    // Depth check to prevent stack overflow
    if depth > MAX_DEPTH {
        // Bail out gracefully, return expression as-is
        return Ok(expr);
    }

    match ctx.get(expr).clone() {
        Expr::Add(a, b) => {
            // Recurse first
            let a2 = presimplify_recursive(ctx, a, depth + 1)?;
            let b2 = presimplify_recursive(ctx, b, depth + 1)?;

            // Apply safe add rules
            Ok(apply_safe_add_rules(ctx, a2, b2))
        }
        Expr::Sub(a, b) => {
            let a2 = presimplify_recursive(ctx, a, depth + 1)?;
            let b2 = presimplify_recursive(ctx, b, depth + 1)?;

            // Apply safe sub rules
            Ok(apply_safe_sub_rules(ctx, a2, b2))
        }
        Expr::Mul(a, b) => {
            let a2 = presimplify_recursive(ctx, a, depth + 1)?;
            let b2 = presimplify_recursive(ctx, b, depth + 1)?;

            // Apply safe mul rules
            Ok(apply_safe_mul_rules(ctx, a2, b2))
        }
        Expr::Neg(a) => {
            let a2 = presimplify_recursive(ctx, a, depth + 1)?;

            // -(-x) → x
            if let Expr::Neg(inner) = ctx.get(a2) {
                return Ok(*inner);
            }

            Ok(ctx.add(Expr::Neg(a2)))
        }
        Expr::Div(num, den) => {
            // Recurse into num/den but do NOT apply a/a → 1
            let num2 = presimplify_recursive(ctx, num, depth + 1)?;
            let den2 = presimplify_recursive(ctx, den, depth + 1)?;
            Ok(ctx.add(Expr::Div(num2, den2)))
        }
        Expr::Pow(base, exp) => {
            // Recurse but do NOT apply a^0 → 1
            let base2 = presimplify_recursive(ctx, base, depth + 1)?;
            let exp2 = presimplify_recursive(ctx, exp, depth + 1)?;
            Ok(ctx.add(Expr::Pow(base2, exp2)))
        }
        Expr::Function(ref name, ref args) => {
            // Recurse into function arguments
            let mut new_args = Vec::with_capacity(args.len());
            for arg in args {
                new_args.push(presimplify_recursive(ctx, *arg, depth + 1)?);
            }
            Ok(ctx.add(Expr::Function(*name, new_args)))
        }
        // Terminal nodes: return as-is
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) => Ok(expr),
        // Other nodes: return as-is (Matrix, SessionRef)
        _ => Ok(expr),
    }
}

/// Safe Add rules: a+0→a, 0+a→a, a+(-a)→0
fn apply_safe_add_rules(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    // a + 0 → a
    if is_zero(ctx, b) {
        return a;
    }
    // 0 + a → a
    if is_zero(ctx, a) {
        return b;
    }

    // a + (-a) → 0 (structural equality)
    if let Expr::Neg(neg_inner) = ctx.get(b) {
        if *neg_inner == a {
            return ctx.num(0);
        }
    }
    if let Expr::Neg(neg_inner) = ctx.get(a) {
        if *neg_inner == b {
            return ctx.num(0);
        }
    }

    ctx.add(Expr::Add(a, b))
}

/// Safe Sub rules: a-0→a, a-a→0
fn apply_safe_sub_rules(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    // a - 0 → a
    if is_zero(ctx, b) {
        return a;
    }

    // a - a → 0 (structural equality)
    if a == b {
        return ctx.num(0);
    }

    ctx.add(Expr::Sub(a, b))
}

/// Safe Mul rules: a*0→0, 0*a→0, a*1→a, 1*a→a
fn apply_safe_mul_rules(ctx: &mut Context, a: ExprId, b: ExprId) -> ExprId {
    // a * 0 → 0, 0 * a → 0
    if is_zero(ctx, a) || is_zero(ctx, b) {
        return ctx.num(0);
    }

    // a * 1 → a
    if is_one(ctx, b) {
        return a;
    }
    // 1 * a → a
    if is_one(ctx, a) {
        return b;
    }

    ctx.add(Expr::Mul(a, b))
}

// Using canonical is_zero and is_one from crate::helpers

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_presimplify_add_zero() {
        let mut ctx = Context::new();
        let mut budget = Budget::new();

        // x + 0 → x
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let expr = ctx.add(Expr::Add(x, zero));

        let result = presimplify_safe(&mut ctx, expr, &mut budget).unwrap();
        assert_eq!(result, x);
    }

    #[test]
    fn test_presimplify_sub_self() {
        let mut ctx = Context::new();
        let mut budget = Budget::new();

        // x - x → 0
        let x = ctx.var("x");
        let expr = ctx.add(Expr::Sub(x, x));

        let result = presimplify_safe(&mut ctx, expr, &mut budget).unwrap();
        let zero = ctx.num(0);
        assert_eq!(result, zero);
    }

    #[test]
    fn test_presimplify_mul_zero() {
        let mut ctx = Context::new();
        let mut budget = Budget::new();

        // x * 0 → 0
        let x = ctx.var("x");
        let zero = ctx.num(0);
        let expr = ctx.add(Expr::Mul(x, zero));

        let result = presimplify_safe(&mut ctx, expr, &mut budget).unwrap();
        assert_eq!(result, zero);
    }

    #[test]
    fn test_presimplify_no_div_cancel() {
        let mut ctx = Context::new();
        let mut budget = Budget::new();

        // x / x should NOT become 1 (domain assumption)
        let x = ctx.var("x");
        let expr = ctx.add(Expr::Div(x, x));

        let result = presimplify_safe(&mut ctx, expr, &mut budget).unwrap();
        // Should remain x/x, not 1
        assert!(matches!(ctx.get(result), Expr::Div(_, _)));
    }

    #[test]
    fn test_presimplify_double_neg() {
        let mut ctx = Context::new();
        let mut budget = Budget::new();

        // -(-x) → x
        let x = ctx.var("x");
        let neg_x = ctx.add(Expr::Neg(x));
        let double_neg = ctx.add(Expr::Neg(neg_x));

        let result = presimplify_safe(&mut ctx, double_neg, &mut budget).unwrap();
        assert_eq!(result, x);
    }

    #[test]
    fn test_presimplify_nested_sub_zero() {
        let mut ctx = Context::new();
        let mut budget = Budget::new();

        // (x - x) / y → 0 / y
        let x = ctx.var("x");
        let y = ctx.var("y");
        let x_minus_x = ctx.add(Expr::Sub(x, x));
        let expr = ctx.add(Expr::Div(x_minus_x, y));

        let result = presimplify_safe(&mut ctx, expr, &mut budget).unwrap();
        // Should be 0/y
        if let Expr::Div(num, _) = ctx.get(result) {
            assert!(is_zero(&ctx, *num), "Numerator should be 0");
        } else {
            panic!("Expected Div");
        }
    }
}
