//! Miscellaneous trig identity rules.
//!
//! This module contains:
//! - TrigSumToProductContractionRule: sin(a)+sin(b) → product form

use crate::rule::Rewrite;
use cas_ast::ExprId;
use cas_math::trig_sum_product_support::try_rewrite_sum_to_product_contraction_expr;

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
        let rewrite = try_rewrite_sum_to_product_contraction_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD_SUB)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::Medium
    }
}

#[cfg(test)]
mod cot_half_angle_tests {
    use crate::rule::Rule;
    use crate::rules::trigonometry::identities::{
        CotHalfAngleDifferenceRule, TrigHiddenCubicIdentityRule,
    };
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
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
