//! Difference of Cubes Simplification for Cube Root Expressions
//!
//! This module provides a specialized pre-order rule for simplifying quotients
//! that match the "difference of cubes" factorization pattern with cube roots.
//!
//! ## Pattern:
//! ```text
//! (x - b³) / (x^(2/3) + b·x^(1/3) + b²) → x^(1/3) - b
//! ```

use crate::define_rule;
use crate::ordering::compare_expr;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed};
use std::cmp::Ordering;

/// Check if `exp` is the rational 1/3
fn is_one_third(ctx: &Context, exp: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(exp) {
        n.numer() == &BigInt::from(1) && n.denom() == &BigInt::from(3)
    } else {
        false
    }
}

/// Check if `exp` is the rational 2/3
fn is_two_thirds(ctx: &Context, exp: ExprId) -> bool {
    if let Expr::Number(n) = ctx.get(exp) {
        n.numer() == &BigInt::from(2) && n.denom() == &BigInt::from(3)
    } else {
        false
    }
}

/// Extract base from x^(1/3) expression
fn extract_cbrt_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if is_one_third(ctx, *exp) {
            return Some(*base);
        }
    }
    None
}

/// Extract base from x^(2/3) expression
fn extract_cbrt_squared_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    if let Expr::Pow(base, exp) = ctx.get(expr) {
        if is_two_thirds(ctx, *exp) {
            return Some(*base);
        }
    }
    None
}

/// Get rational value from Number expression
fn get_rational(ctx: &Context, id: ExprId) -> Option<BigRational> {
    if let Expr::Number(n) = ctx.get(id) {
        Some(n.clone())
    } else {
        None
    }
}

/// Try to match (x - c) pattern and return (base_x, c_value)
/// Handles canonicalized Add expressions where terms may be reordered
fn match_x_minus_const(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    // Helper to collect additive terms
    fn collect_add_terms(ctx: &Context, e: ExprId) -> Vec<ExprId> {
        match ctx.get(e) {
            Expr::Add(a, b) => {
                let mut left = collect_add_terms(ctx, *a);
                let mut right = collect_add_terms(ctx, *b);
                left.append(&mut right);
                left
            }
            _ => vec![e],
        }
    }

    let terms = collect_add_terms(ctx, expr);

    // We need exactly 2 terms: x and Neg(c) (or just -c number)
    if terms.len() != 2 {
        return None;
    }

    let mut var_term: Option<ExprId> = None;
    let mut neg_const: Option<BigRational> = None;

    for term in &terms {
        match ctx.get(*term) {
            // Variable x
            Expr::Variable(_) => {
                var_term = Some(*term);
            }
            // Neg(c) where c is a number
            Expr::Neg(inner) => {
                if let Expr::Number(n) = ctx.get(*inner) {
                    neg_const = Some(n.clone());
                }
            }
            // Negative number directly (not common but handle it)
            Expr::Number(n) if n.is_negative() => {
                neg_const = Some(-n.clone());
            }
            _ => {}
        }
    }

    // Both must be found
    let x = var_term?;
    let c = neg_const?;

    Some((x, c))
}

/// Check if a rational is a perfect cube and return its cube root
fn perfect_cube_root(n: &BigRational) -> Option<BigRational> {
    // Only handle integers for now
    if !n.is_integer() {
        return None;
    }
    let int_val = n.to_integer();

    // Check small perfect cubes
    for b in 1..=100i32 {
        let cube = BigInt::from(b).pow(3);
        if cube == int_val {
            return Some(BigRational::from_integer(BigInt::from(b)));
        }
        if cube == -int_val.clone() {
            return Some(BigRational::from_integer(BigInt::from(-b)));
        }
    }
    None
}

// CancelCubeRootDifferenceRule: Simplifies (x - b³) / (x^(2/3) + b·x^(1/3) + b²) → x^(1/3) - b
//
// This is a pre-order rule that catches the specific algebraic pattern before
// the general fraction simplification machinery can cause oscillation.
define_rule!(
    CancelCubeRootDifferenceRule,
    "Cancel Cube Root Difference",
    None,
    PhaseMask::CORE, // Run early in Core phase
    |ctx, expr| {
        // Match: Div(num, den)
        let div_fields = match ctx.get(expr) {
            Expr::Div(n, d) => Some((*n, *d)),
            _ => None,
        };
        if let Some((num, den)) = div_fields {
            // Try to match numerator: x - b³
            let (base_x, cube_val) = match_x_minus_const(ctx, num)?;

            // cube_val should be b³, find b
            let b_val = perfect_cube_root(&cube_val)?;

            // Now verify denominator is: x^(2/3) + b·x^(1/3) + b²
            // We'll check the structure more flexibly

            // Expected values
            let b_squared = &b_val * &b_val;

            // Check denominator structure - can be Add of 3 terms
            // For now, check the specific structure of nested Adds

            // Helper to collect additive terms
            fn collect_add_terms(ctx: &Context, e: ExprId) -> Vec<ExprId> {
                match ctx.get(e) {
                    Expr::Add(a, b) => {
                        let mut left = collect_add_terms(ctx, *a);
                        let mut right = collect_add_terms(ctx, *b);
                        left.append(&mut right);
                        left
                    }
                    _ => vec![e],
                }
            }

            let den_terms = collect_add_terms(ctx, den);

            if den_terms.len() != 3 {
                return None;
            }

            // Find terms:
            // - x^(2/3)
            // - b·x^(1/3) or just x^(1/3) if b=1
            // - b² (constant)

            let mut found_x_2_3 = false;
            let mut found_cbrt_term = false;
            let mut found_const = false;

            for term in &den_terms {
                // Check for x^(2/3)
                if let Some(base) = extract_cbrt_squared_base(ctx, *term) {
                    if compare_expr(ctx, base, base_x) == Ordering::Equal {
                        found_x_2_3 = true;
                        continue;
                    }
                }

                // Check for x^(1/3) or b·x^(1/3)
                if let Some(base) = extract_cbrt_base(ctx, *term) {
                    if compare_expr(ctx, base, base_x) == Ordering::Equal {
                        // Just x^(1/3), b should be 1
                        if b_val == BigRational::one() {
                            found_cbrt_term = true;
                            continue;
                        }
                    }
                }

                // Check for b·x^(1/3) as Mul(b, x^(1/3))
                if let Expr::Mul(l, r) = ctx.get(*term) {
                    // Try both orders
                    if let Some(coeff) = get_rational(ctx, *l) {
                        if let Some(base) = extract_cbrt_base(ctx, *r) {
                            if compare_expr(ctx, base, base_x) == Ordering::Equal && coeff == b_val
                            {
                                found_cbrt_term = true;
                                continue;
                            }
                        }
                    }
                    if let Some(coeff) = get_rational(ctx, *r) {
                        if let Some(base) = extract_cbrt_base(ctx, *l) {
                            if compare_expr(ctx, base, base_x) == Ordering::Equal && coeff == b_val
                            {
                                found_cbrt_term = true;
                                continue;
                            }
                        }
                    }
                }

                // Check for constant b²
                if let Some(val) = get_rational(ctx, *term) {
                    if val == b_squared {
                        found_const = true;
                        continue;
                    }
                }
            }

            if !found_x_2_3 || !found_cbrt_term || !found_const {
                return None;
            }

            // Build result: x^(1/3) - b
            let one_third = ctx.add(Expr::Number(BigRational::new(
                BigInt::from(1),
                BigInt::from(3),
            )));
            let cbrt_x = ctx.add(Expr::Pow(base_x, one_third));
            let b_expr = ctx.add(Expr::Number(b_val.clone()));
            let neg_b = ctx.add(Expr::Neg(b_expr));
            let factor = ctx.add(Expr::Add(cbrt_x, neg_b)); // x^(1/3) - b

            // === ChainedRewrite Pattern: Factor → Cancel ===
            // Step 1 (main): Build intermediate form ((x^(1/3) - b)·den) / den
            let factored_num = ctx.add(Expr::Mul(factor, den));
            let intermediate = ctx.add(Expr::Div(factored_num, den));

            use crate::implicit_domain::ImplicitCondition;
            use crate::rule::ChainedRewrite;

            // V2.14.32: .local() now works correctly thanks to scoped search in timeline.rs
            // The engine limits highlight search to before_local subtree, avoiding
            // incorrect highlights in denominator from shared ExprIds.
            let factor_rw = Rewrite::new(intermediate)
                .desc_lazy(|| format!(
                    "Factor difference of cubes: x - {} = (x^(1/3) - {})·(x^(2/3) + {}·x^(1/3) + {})",
                    cube_val, b_val, b_val, &b_val * &b_val
                ))
                .local(num, factored_num)
                .requires(ImplicitCondition::NonZero(den));

            // Step 2 (chained): Cancel common factor - reduce to final result
            let cancel = ChainedRewrite::new(factor)
                .desc("Cancel common factor")
                .local(intermediate, factor);

            return Some(factor_rw.chain(cancel));
        }
        None
    }
);

/// Register the difference of cubes rules
pub fn register(simplifier: &mut crate::Simplifier) {
    // Register BEFORE general fraction simplification for pre-order behavior
    simplifier.add_rule(Box::new(CancelCubeRootDifferenceRule));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::DisplayExpr;

    #[test]
    fn test_cancel_cube_root_difference_basic() {
        let mut ctx = Context::new();

        // Build: (x - 27) / (x^(2/3) + 3*x^(1/3) + 9)
        let x = ctx.var("x");
        let c27 = ctx.num(27);
        let c3 = ctx.num(3);
        let c9 = ctx.num(9);

        // Numerator: x - 27 as Add(x, Neg(27))
        let neg_27 = ctx.add(Expr::Neg(c27));
        let num = ctx.add(Expr::Add(x, neg_27));

        // Exponents
        let one_third = ctx.add(Expr::Number(BigRational::new(
            BigInt::from(1),
            BigInt::from(3),
        )));
        let two_thirds = ctx.add(Expr::Number(BigRational::new(
            BigInt::from(2),
            BigInt::from(3),
        )));

        // Denominator terms
        let x_2_3 = ctx.add(Expr::Pow(x, two_thirds));
        let x_1_3 = ctx.add(Expr::Pow(x, one_third));
        let term_mid = ctx.add(Expr::Mul(c3, x_1_3));

        // Den: x^(2/3) + 3*x^(1/3) + 9
        let den_partial = ctx.add(Expr::Add(x_2_3, term_mid));
        let den = ctx.add(Expr::Add(den_partial, c9));

        // Full expression
        let expr = ctx.add(Expr::Div(num, den));

        let rule = CancelCubeRootDifferenceRule;
        let result = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(result.is_some(), "Rule should match this pattern");

        let rewrite = result.unwrap();
        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );

        // Should be x^(1/3) - 3 or equivalent
        println!("Result: {}", result_str);
        assert!(
            result_str.contains("x") && result_str.contains("1/3"),
            "Result should contain cube root of x: got {}",
            result_str
        );
    }
}
