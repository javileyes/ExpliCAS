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
use cas_ast::{Context, ExprId};
use cas_math::integration_prep_support::try_rewrite_cos_product_telescoping_expr;
use cas_math::trig_sum_product_support::try_rewrite_product_to_sum_werner_expr;

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
        let rewrite = try_rewrite_product_to_sum_werner_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
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

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL)
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
        let rewrite = try_rewrite_cos_product_telescoping_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc).assume(
            crate::assumptions::AssumptionEvent::nonzero(ctx, rewrite.assume_nonzero_expr),
        ))
    }
}

/// Register integration preparation rules.
pub fn register_integration_prep(simplifier: &mut Simplifier) {
    simplifier.add_rule(Box::new(ProductToSumRule));
    simplifier.add_rule(Box::new(CosProductTelescopingRule));
}
