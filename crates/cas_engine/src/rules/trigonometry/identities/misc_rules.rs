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
