//! Miscellaneous trig identity rules.
//!
//! This module contains:
//! - TrigSumToProductContractionRule: sin(a)+sin(b) → product form

use crate::rule::Rewrite;
use cas_ast::{BuiltinFn, Expr, ExprId};

// =============================================================================
// TrigSumToProductContractionRule: sin(a)+sin(b) → 2*sin((a+b)/2)*cos((a-b)/2)
// =============================================================================
// Contracts sum/difference of sines or cosines into product form.
// IMPORTANT: Only contraction direction (sum→product), no inverse.
//
// Guard: Only applies when a and b are linear multiples of the same base variable.
// This prevents explosion and ensures (a+b)/2 and (a-b)/2 simplify nicely.
//
// Identities:
//   sin(a) + sin(b) → 2*sin((a+b)/2)*cos((a-b)/2)
//   sin(a) - sin(b) → 2*cos((a+b)/2)*sin((a-b)/2)
//   cos(a) + cos(b) → 2*cos((a+b)/2)*cos((a-b)/2)
//   cos(a) - cos(b) → -2*sin((a+b)/2)*sin((a-b)/2)
//
// Example: sin(u) + sin(3u) → 2*sin(2u)*cos(u)
// =============================================================================
pub struct TrigSumToProductContractionRule;

impl crate::rule::Rule for TrigSumToProductContractionRule {
    fn name(&self) -> &str {
        "Trig Sum to Product"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        // Match Add or Sub of two trig functions
        let (left, right, is_add) = match ctx.get(expr).clone() {
            Expr::Add(l, r) => (l, r, true),
            Expr::Sub(l, r) => (l, r, false),
            _ => return None,
        };

        // Both must be sin or cos functions
        let (l_name, l_arg) = extract_trig_fn(ctx, left)?;
        let (r_name, r_arg) = extract_trig_fn(ctx, right)?;

        // Must be same function type (sin+sin or cos+cos)
        if l_name != r_name {
            return None;
        }

        // Guard: Check if both args are linear multiples of the same base
        // Returns (base, coef_a, coef_b) if a = coef_a * base, b = coef_b * base
        let (base, coef_a, coef_b) = extract_linear_coefficients(ctx, l_arg, r_arg)?;

        // Compute (a+b)/2 and (a-b)/2 using coefficients
        // (coef_a + coef_b) / 2 and (coef_a - coef_b) / 2
        let sum_coef = &coef_a + &coef_b;
        let diff_coef = &coef_a - &coef_b;

        let two = num_rational::BigRational::from_integer(2.into());
        let half_sum = sum_coef / (&two);
        let half_diff = diff_coef / &two;

        // Build half_sum * base and half_diff * base
        let half_sum_arg = build_coef_times_base(ctx, &half_sum, base);
        let half_diff_arg = build_coef_times_base(ctx, &half_diff, base);

        // Build the product form based on function name and operation
        let two_id = ctx.num(2);
        let result = match (l_name.as_str(), is_add) {
            ("sin", true) => {
                // sin(a) + sin(b) → 2*sin((a+b)/2)*cos((a-b)/2)
                let sin_half_sum = ctx.call("sin", vec![half_sum_arg]);
                let cos_half_diff = ctx.call("cos", vec![half_diff_arg]);
                let product = ctx.add(Expr::Mul(sin_half_sum, cos_half_diff));
                ctx.add(Expr::Mul(two_id, product))
            }
            ("sin", false) => {
                // sin(a) - sin(b) → 2*cos((a+b)/2)*sin((a-b)/2)
                let cos_half_sum = ctx.call("cos", vec![half_sum_arg]);
                let sin_half_diff = ctx.call("sin", vec![half_diff_arg]);
                let product = ctx.add(Expr::Mul(cos_half_sum, sin_half_diff));
                ctx.add(Expr::Mul(two_id, product))
            }
            ("cos", true) => {
                // cos(a) + cos(b) → 2*cos((a+b)/2)*cos((a-b)/2)
                let cos_half_sum = ctx.call("cos", vec![half_sum_arg]);
                let cos_half_diff = ctx.call("cos", vec![half_diff_arg]);
                let product = ctx.add(Expr::Mul(cos_half_sum, cos_half_diff));
                ctx.add(Expr::Mul(two_id, product))
            }
            ("cos", false) => {
                // cos(a) - cos(b) → -2*sin((a+b)/2)*sin((a-b)/2)
                let sin_half_sum = ctx.call("sin", vec![half_sum_arg]);
                let sin_half_diff = ctx.call("sin", vec![half_diff_arg]);
                let product = ctx.add(Expr::Mul(sin_half_sum, sin_half_diff));
                let two_times = ctx.add(Expr::Mul(two_id, product));
                ctx.add(Expr::Neg(two_times))
            }
            _ => return None,
        };

        Some(Rewrite::new(result).desc("Sum-to-product: sin/cos sum → product form"))
    }

    fn target_types(&self) -> Option<Vec<&str>> {
        Some(vec!["Add", "Sub"])
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }
}

/// Extract trig function name and argument
fn extract_trig_fn(ctx: &cas_ast::Context, expr: ExprId) -> Option<(String, ExprId)> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        let builtin = ctx.builtin_of(*fn_id);
        if args.len() == 1 {
            match builtin {
                Some(BuiltinFn::Sin) => return Some(("sin".to_string(), args[0])),
                Some(BuiltinFn::Cos) => return Some(("cos".to_string(), args[0])),
                _ => {}
            }
        }
    }
    None
}

/// Extract linear coefficients if both args are linear multiples of a common base.
/// Returns (base, coef_a, coef_b) where a = coef_a * base, b = coef_b * base
fn extract_linear_coefficients(
    ctx: &cas_ast::Context,
    a: ExprId,
    b: ExprId,
) -> Option<(ExprId, num_rational::BigRational, num_rational::BigRational)> {
    let (coef_a, base_a) = extract_coef_and_base(ctx, a);
    let (coef_b, base_b) = extract_coef_and_base(ctx, b);

    // Check if bases are structurally equal
    if crate::ordering::compare_expr(ctx, base_a, base_b) == std::cmp::Ordering::Equal {
        Some((base_a, coef_a, coef_b))
    } else {
        None
    }
}

/// Extract coefficient and base from an expression.
/// n*t → (n, t), t → (1, t), 3 (literal) → (3, 1)
fn extract_coef_and_base(
    ctx: &cas_ast::Context,
    expr: ExprId,
) -> (num_rational::BigRational, ExprId) {
    match ctx.get(expr).clone() {
        Expr::Mul(l, r) => {
            // Check if left is a number
            if let Expr::Number(n) = ctx.get(l) {
                return (n.clone(), r);
            }
            // Check if right is a number
            if let Expr::Number(n) = ctx.get(r) {
                return (n.clone(), l);
            }
            // No numeric coefficient - treat as 1 * expr
            (num_rational::BigRational::from_integer(1.into()), expr)
        }
        Expr::Number(n) => {
            // Pure number - but we need a "dummy" base that matches
            // This case shouldn't happen in typical trig args, return as-is
            (n.clone(), expr)
        }
        Expr::Neg(inner) => {
            // -t → (-1, t)
            let (inner_coef, inner_base) = extract_coef_and_base(ctx, inner);
            (-inner_coef, inner_base)
        }
        _ => {
            // Variable or function: treat as 1 * expr
            (num_rational::BigRational::from_integer(1.into()), expr)
        }
    }
}

fn build_coef_times_base(
    ctx: &mut cas_ast::Context,
    coef: &num_rational::BigRational,
    base: ExprId,
) -> ExprId {
    use num_traits::{One, Zero};

    if coef.is_zero() {
        ctx.num(0)
    } else if coef.is_one() {
        base
    } else if *coef == -num_rational::BigRational::one() {
        ctx.add(Expr::Neg(base))
    } else {
        let coef_id = ctx.add(Expr::Number(coef.clone()));
        ctx.add(Expr::Mul(coef_id, base))
    }
}

#[cfg(test)]
mod cot_half_angle_tests {
    use crate::rule::Rule;
    use crate::rules::trigonometry::identities::{
        CotHalfAngleDifferenceRule, TrigHiddenCubicIdentityRule,
    };
    use cas_ast::{Context, DisplayExpr};
    use cas_parser::parse;

    #[test]
    fn test_cot_half_angle_basic() {
        let mut ctx = Context::new();
        let rule = CotHalfAngleDifferenceRule;

        // cot(x/2) - cot(x) → 1/sin(x)
        let expr = parse("cot(x/2) - cot(x)", &mut ctx).unwrap();
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_some(), "Should match cot(x/2) - cot(x)");

        let result = rewrite.unwrap();
        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: result.new_expr
            }
        );
        assert!(
            result_str.contains("sin"),
            "Result should contain sin, got: {}",
            result_str
        );
    }

    #[test]
    fn test_cot_half_angle_no_match_different_args() {
        let mut ctx = Context::new();
        let rule = CotHalfAngleDifferenceRule;

        // cot(x/2) - cot(y) → no change (different args)
        let expr = parse("cot(x/2) - cot(y)", &mut ctx).unwrap();
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match cot(x/2) - cot(y)");
    }

    #[test]
    fn test_cot_half_angle_no_match_third() {
        let mut ctx = Context::new();
        let rule = CotHalfAngleDifferenceRule;

        // cot(x/3) - cot(x) → no change (not half-angle)
        let expr = parse("cot(x/3) - cot(x)", &mut ctx).unwrap();
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match cot(x/3) - cot(x)");
    }

    // =========================================================================
    // TrigHiddenCubicIdentityRule tests
    // =========================================================================

    #[test]
    fn test_hidden_cubic_basic() {
        let mut ctx = Context::new();
        let expr = parse("sin(x)^6 + cos(x)^6 + 3*sin(x)^2*cos(x)^2", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_hidden_cubic_permutation_cos_first() {
        let mut ctx = Context::new();
        // Different order: cos^6 first
        let expr = parse("cos(x)^6 + 3*cos(x)^2*sin(x)^2 + sin(x)^6", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_hidden_cubic_coeff_product_first() {
        let mut ctx = Context::new();
        // Coefficient product first
        let expr = parse("3*sin(x)^2*cos(x)^2 + sin(x)^6 + cos(x)^6", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_hidden_cubic_equivalent_coeff() {
        let mut ctx = Context::new();
        // Coefficient 6/2 = 3
        let expr = parse("sin(x)^6 + cos(x)^6 + (6/2)*sin(x)^2*cos(x)^2", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_hidden_cubic_no_match_wrong_coeff() {
        let mut ctx = Context::new();
        // Wrong coefficient: 2 instead of 3
        let expr = parse("sin(x)^6 + cos(x)^6 + 2*sin(x)^2*cos(x)^2", &mut ctx).unwrap();

        let rule = TrigHiddenCubicIdentityRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match with coeff=2");
    }

    #[test]
    fn test_hidden_cubic_no_match_different_args() {
        let mut ctx = Context::new();
        // Different arguments: x vs y
        let expr = parse("sin(x)^6 + cos(y)^6 + 3*sin(x)^2*cos(y)^2", &mut ctx).unwrap();

        let rule = TrigHiddenCubicIdentityRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match with different args");
    }

    #[test]
    fn test_hidden_cubic_no_match_extra_terms() {
        let mut ctx = Context::new();
        // Extra term: should not match partially
        let expr = parse("sin(x)^6 + cos(x)^6 + 3*sin(x)^2*cos(x)^2 + 1", &mut ctx).unwrap();

        let rule = TrigHiddenCubicIdentityRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        // flatten_add will produce 4 terms, so rule should not match (requires exactly 3)
        assert!(rewrite.is_none(), "Should not match with extra terms");
    }
}
