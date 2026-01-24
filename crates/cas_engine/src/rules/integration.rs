//! Integration preparation rules.
//!
//! These rules transform expressions into forms more amenable to integration:
//! - Product-to-sum identities (Werner formulas)
//! - Telescoping product series (Morrie's law)
//!
//! Only active when `ContextMode::IntegratePrep` is set.

use crate::engine::Simplifier;
use crate::parent_context::ParentContext;
use crate::rule::{Rewrite, Rule};
use cas_ast::{Context, Expr, ExprId};

/// Product-to-sum identity for trigonometric products (Werner formulas).
///
/// `2 * sin(A) * cos(B) → sin(A+B) + sin(A-B)`
/// `2 * cos(A) * cos(B) → cos(A+B) + cos(A-B)`
/// `2 * sin(A) * sin(B) → cos(A-B) - cos(A+B)`
pub struct ProductToSumRule;

impl Rule for ProductToSumRule {
    fn name(&self) -> &str {
        "ProductToSum"
    }

    fn priority(&self) -> i32 {
        50 // Medium priority for Werner formulas
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        _parent_ctx: &ParentContext,
    ) -> Option<Rewrite> {
        // Look for: 2 * sin(A) * cos(B) or similar patterns
        // First, flatten the multiplication and check for coefficient 2

        let factors = crate::helpers::flatten_mul_chain(ctx, expr);

        // Check if we have at least 3 factors (coeff, sin, cos)
        if factors.len() < 2 {
            return None;
        }

        // Find numeric coefficient
        let mut coeff_idx = None;
        let mut sin_idx = None;
        let mut cos_idx = None;
        let mut sin_arg = None;
        let mut cos_arg = None;

        for (i, &f) in factors.iter().enumerate() {
            match ctx.get(f) {
                Expr::Number(n) => {
                    if n.to_integer() == 2.into() && coeff_idx.is_none() {
                        coeff_idx = Some(i);
                    }
                }
                Expr::Function(name, args) if args.len() == 1 => {
                    if name == "sin" && sin_idx.is_none() {
                        sin_idx = Some(i);
                        sin_arg = Some(args[0]);
                    } else if name == "cos" && cos_idx.is_none() {
                        cos_idx = Some(i);
                        cos_arg = Some(args[0]);
                    }
                }
                _ => {}
            }
        }

        // Pattern: 2 * sin(A) * cos(B) → sin(A+B) + sin(A-B)
        if let (Some(_), Some(_), Some(_), Some(a), Some(b)) =
            (coeff_idx, sin_idx, cos_idx, sin_arg, cos_arg)
        {
            // Build sin(A+B) + sin(A-B)
            let a_plus_b = ctx.add(Expr::Add(a, b));
            let a_minus_b = ctx.add(Expr::Sub(a, b));
            let sin_sum = ctx.call("sin", vec![a_plus_b]);
            let sin_diff = ctx.call("sin", vec![a_minus_b]);
            let result = ctx.add(Expr::Add(sin_sum, sin_diff));

            return Some(
                Rewrite::new(result).desc("2·sin(A)·cos(B) → sin(A+B) + sin(A-B) (Werner)"),
            );
        }

        None
    }
}

/// Morrie's law: telescoping product of cosines.
///
/// `cos(u) * cos(2u) * cos(4u) * ... * cos(2^(n-1) u) → sin(2^n u) / (2^n sin(u))`
///
/// Example: `cos(x) * cos(2x) * cos(4x) → sin(8x) / (8 sin(x))`
///
/// **Warning**: This introduces a division by sin(u), so it's only valid where sin(u) ≠ 0.
/// The rule emits a domain_assumption warning.
pub struct CosProductTelescopingRule;

impl Rule for CosProductTelescopingRule {
    fn name(&self) -> &str {
        "CosProductTelescoping"
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Mul"])
    }

    fn priority(&self) -> i32 {
        100 // High priority - must match before general identity rules
    }

    fn apply(
        &self,
        ctx: &mut Context,
        expr: ExprId,
        _parent_ctx: &ParentContext,
    ) -> Option<Rewrite> {
        use num_bigint::BigInt;
        use num_rational::BigRational;

        // Flatten the product
        let factors = crate::helpers::flatten_mul_chain(ctx, expr);

        // We need at least 2 cosine factors
        if factors.len() < 2 {
            return None;
        }

        // Extract cosine arguments: look for cos(k*u) pattern
        // Collect (factor_idx, multiplier_k, base_arg_u)
        let mut cos_info: Vec<(usize, i64, ExprId)> = Vec::new();

        for (i, &f) in factors.iter().enumerate() {
            if let Expr::Function(name, args) = ctx.get(f) {
                if name == "cos" && args.len() == 1 {
                    let arg = args[0];
                    // Try to extract k and u from: k*u or just u (k=1)
                    let (k, u) = extract_multiplier_and_base(ctx, arg);
                    cos_info.push((i, k, u));
                }
            }
        }

        // Need at least 2 cosines
        if cos_info.len() < 2 {
            return None;
        }

        // Check if the multipliers form a geometric sequence with ratio 2
        // i.e., {1, 2, 4, ...} or {a, 2a, 4a, ...}
        // First, group by base expression
        // For simplicity, check if all have the same base u
        let base_u = cos_info[0].2;
        let mut multipliers: Vec<i64> = Vec::new();

        for (_, k, u) in &cos_info {
            // Check same base (structural equality)
            if *u != base_u {
                return None; // Different bases
            }
            multipliers.push(*k);
        }

        // Sort multipliers
        multipliers.sort();

        // Check for {1, 2, 4, ...} pattern (powers of 2 starting from 1)
        // or more generally {a, 2a, 4a, ...} (powers of 2 times a)
        let base_mult = multipliers[0];
        if base_mult <= 0 {
            return None;
        }

        let n = multipliers.len();
        for (i, &m) in multipliers.iter().enumerate() {
            let expected = base_mult * (1i64 << i); // base * 2^i
            if m != expected {
                return None; // Not a power-of-2 sequence
            }
        }

        // Pattern matches! Apply Morrie's law
        // Result: sin(2^n * base_mult * u) / (2^n * sin(base_mult * u))
        let power_of_2 = 1i64 << n; // 2^n
        let final_mult = base_mult * power_of_2; // 2^n * base_mult

        // Build final_mult * u
        let final_mult_num = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(
            final_mult,
        ))));
        let final_arg = ctx.add(Expr::Mul(final_mult_num, base_u));

        // Build base_mult * u (argument for sin in denominator)
        let base_mult_num = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(
            base_mult,
        ))));
        let base_arg = if base_mult == 1 {
            base_u
        } else {
            ctx.add(Expr::Mul(base_mult_num, base_u))
        };

        // Build sin(final_arg) and sin(base_arg)
        let sin_num = ctx.call("sin", vec![final_arg]);
        let sin_den = ctx.call("sin", vec![base_arg]);

        // Build 2^n * sin(base_arg)
        let power_of_2_num = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(
            power_of_2,
        ))));
        let denominator = ctx.add(Expr::Mul(power_of_2_num, sin_den));

        // Build sin(final_arg) / (2^n * sin(base_arg))
        let result = ctx.add(Expr::Div(sin_num, denominator));

        Some(Rewrite::new(result).desc(format!(
            "cos telescoping (Morrie's law): cos(u)·cos(2u)·...·cos(2^{}u) → sin(2^{}u)/(2^{}·sin(u))",
            n - 1, n, n
        )).assume(crate::assumptions::AssumptionEvent::nonzero(ctx, sin_den)))
    }
}

/// Extract multiplier k and base expression u from k*u or just u (k=1).
fn extract_multiplier_and_base(ctx: &Context, expr: ExprId) -> (i64, ExprId) {
    if let Expr::Mul(l, r) = ctx.get(expr) {
        // Check if left is a number
        if let Expr::Number(n) = ctx.get(*l) {
            if n.is_integer() {
                if let Ok(k) = n.to_integer().try_into() {
                    return (k, *r);
                }
            }
        }
        // Check if right is a number (canonical form puts numbers first, but be safe)
        if let Expr::Number(n) = ctx.get(*r) {
            if n.is_integer() {
                if let Ok(k) = n.to_integer().try_into() {
                    return (k, *l);
                }
            }
        }
    }
    // No multiplier: k=1, u=expr
    (1, expr)
}

/// Register integration preparation rules.
pub fn register_integration_prep(simplifier: &mut Simplifier) {
    simplifier.add_rule(Box::new(ProductToSumRule));
    simplifier.add_rule(Box::new(CosProductTelescopingRule));
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_parser::parse;

    #[test]
    fn test_product_to_sum_sin_cos() {
        let mut ctx = Context::new();
        let expr = parse("2*sin(x)*cos(y)", &mut ctx).unwrap();

        let rule = ProductToSumRule;
        let result = rule.apply(&mut ctx, expr, &ParentContext::root());

        assert!(
            result.is_some(),
            "ProductToSum should match 2*sin(x)*cos(y)"
        );
    }

    #[test]
    fn test_cos_product_telescoping() {
        let mut ctx = Context::new();
        // cos(x) * cos(2*x) * cos(4*x) -> sin(8*x) / (8*sin(x))
        let expr = parse("cos(x)*cos(2*x)*cos(4*x)", &mut ctx).unwrap();

        let rule = CosProductTelescopingRule;
        let result = rule.apply(&mut ctx, expr, &ParentContext::root());

        assert!(
            result.is_some(),
            "CosProductTelescoping should match cos(x)*cos(2x)*cos(4x)"
        );

        let rewrite = result.unwrap();
        // CosProductTelescopingRule now uses structured assumption_events
        assert!(
            !rewrite.assumption_events.is_empty(),
            "Should have assumption_events for sin(u) ≠ 0 warning"
        );
    }
}
