//! Contraction rules for trigonometric expressions.
//!
//! These are the INVERSE of expansion rules — they contract expanded forms back
//! to compact representations (half-angle tangent, double angle contraction).

use crate::rule::Rewrite;
use cas_ast::ExprId;
use cas_math::trig_contraction_support::{
    should_block_double_angle_contraction_for_marks, try_rewrite_angle_sum_fraction_to_tan_expr,
    try_rewrite_cos2x_additive_contraction_expr, try_rewrite_double_angle_contraction_expr,
    try_rewrite_half_angle_tangent_div_expr,
};

// =============================================================================
// HALF-ANGLE TANGENT RULE
// (1 - cos(2x)) / sin(2x) → tan(x)
// sin(2x) / (1 + cos(2x)) → tan(x)
// =============================================================================
// These are half-angle tangent identities derived from:
//   1 - cos(2x) = 2·sin²(x)
//   1 + cos(2x) = 2·cos²(x)
//   sin(2x) = 2·sin(x)·cos(x)
//
// DOMAIN WARNING: This transformation can extend the domain:
// - Pattern 1: Original requires sin(2x) ≠ 0, but tan(x) only requires cos(x) ≠ 0
// - Pattern 2: Original requires 1+cos(2x) ≠ 0, but tan(x) only requires cos(x) ≠ 0
//
// To preserve soundness, we introduce requires for cos(x) ≠ 0 (for tan(x) to be defined)
// and inherit the original denominator ≠ 0 condition.
//
// Uses SoundnessLabel::EquivalenceUnderIntroducedRequires
pub struct HalfAngleTangentRule;

impl crate::rule::Rule for HalfAngleTangentRule {
    fn name(&self) -> &str {
        "Half-Angle Tangent Identity"
    }

    fn priority(&self) -> i32 {
        50 // Normal priority
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        use crate::ImplicitCondition;
        let rewrite = try_rewrite_half_angle_tangent_div_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(rewrite.desc)
                .requires(ImplicitCondition::NonZero(rewrite.inherited_nonzero))
                .requires(ImplicitCondition::NonZero(rewrite.required_nonzero)),
        )
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn soundness(&self) -> crate::rule::SoundnessLabel {
        crate::rule::SoundnessLabel::EquivalenceUnderIntroducedRequires
    }
}

// =============================================================================
// DOUBLE ANGLE CONTRACTION RULE
// 2·sin(t)·cos(t) → sin(2t), cos²(t) - sin²(t) → cos(2t)
// =============================================================================
// This is the INVERSE of DoubleAngleRule - contracts expanded forms back to double angle.
// Essential for recognizing Weierstrass substitution identities.
pub struct DoubleAngleContractionRule;

impl crate::rule::Rule for DoubleAngleContractionRule {
    fn name(&self) -> &str {
        "Double Angle Contraction"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        if should_block_double_angle_contraction_for_marks(_parent_ctx.pattern_marks()) {
            return None;
        }
        let rewrite = try_rewrite_double_angle_contraction_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::MUL.union(crate::target_kind::TargetKindSet::SUB))
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        200 // Run before expansion rules to prevent ping-pong
    }
}

// =============================================================================
// Cos2xAdditiveContractionRule: 1 - 2·sin²(t) → cos(2t), 2·cos²(t) - 1 → cos(2t)
// =============================================================================
// These are alternate forms of the double-angle cosine identity that the
// existing DoubleAngleContractionRule does not handle (it only handles
// cos²(t) - sin²(t) → cos(2t)).
//
// Mathematical identities:
//   cos(2t) = 1 - 2·sin²(t)
//   cos(2t) = 2·cos²(t) - 1
//
// We scan additive leaves for a pair: constant ±1 and ∓2·trig²(t).

pub struct Cos2xAdditiveContractionRule;

impl crate::rule::Rule for Cos2xAdditiveContractionRule {
    fn name(&self) -> &str {
        "Cos 2x Additive Contraction"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let rewrite = try_rewrite_cos2x_additive_contraction_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // POST only: prevents oscillation with HalfAngleExpansion and other trig
        // rules during TRANSFORM. In POST no expansion rules fire, so contraction
        // is stable and normalises 2cos²(t)-1 / 1-2sin²(t) to cos(2t).
        crate::phase::PhaseMask::POST
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::ADD.union(crate::target_kind::TargetKindSet::SUB))
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }

    fn priority(&self) -> i32 {
        200
    }
}

// =============================================================================
// AngleSumFractionToTanRule
// (sin(a)cos(b) + cos(a)sin(b)) / (cos(a)cos(b) - sin(a)sin(b)) → tan(a+b)
// (sin(a)cos(b) - cos(a)sin(b)) / (cos(a)cos(b) + sin(a)sin(b)) → tan(a-b)
// =============================================================================
// This rule contracts expanded angle-addition fractions back to tan.
// It targets Div nodes only, so it cannot loop with AngleIdentityRule
// (which targets Function nodes with sin/cos of Add/Sub arguments).
//
// Also handles the case where the numerator/denominator has extra common
// factors (e.g., multiplied through by cos²), by first trying the
// bare 2-term pattern.

pub struct AngleSumFractionToTanRule;

impl crate::rule::Rule for AngleSumFractionToTanRule {
    fn name(&self) -> &str {
        "Angle Sum Fraction to Tan"
    }

    fn priority(&self) -> i32 {
        190 // Below contraction rules at 200, above normal
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        let rewrite = try_rewrite_angle_sum_fraction_to_tan_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}
