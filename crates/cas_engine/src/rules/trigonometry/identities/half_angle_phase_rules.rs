//! Half-angle, cotangent half-angle, Weierstrass, and identity zero rules.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::ExprId;
use cas_math::hyperbolic_identity_support::try_rewrite_tanh_pythagorean_add_chain;
use cas_math::trig_half_angle_support::{
    try_rewrite_cot_half_angle_difference_expr, CotHalfAngleDifferenceRewriteKind,
};
use cas_math::trig_sum_product_support::try_rewrite_tan_difference_expr;
use cas_math::trig_weierstrass_support::{
    try_rewrite_weierstrass_contraction_div_expr, WeierstrassContractionKind,
};

// ============================================================================
// Cotangent Half-Angle Difference Rule
// ============================================================================
// cot(u/2) - cot(u) = 1/sin(u) = csc(u)
//
// This is a common precalculus identity that avoids term explosion from
// brute-force expansion via cot→cos/sin + double angle formulas.
//
// Pattern matching:
// - cot(u/2) - cot(u) → 1/sin(u)
// - k*cot(u/2) - k*cot(u) → k/sin(u)
// - Works on n-ary sums via flatten_add

// =============================================================================
// WEIERSTRASS HALF-ANGLE TANGENT CONTRACTION RULES
// =============================================================================
// Recognize patterns with t = tan(x/2) and contract to sin(x), cos(x):
// - 2*t / (1 + t²) → sin(x)
// - (1 - t²) / (1 + t²) → cos(x)
// This is the CONTRACTION direction (safe, doesn't worsen expressions)

// Weierstrass Contraction Rule: 2*tan(x/2)/(1+tan²(x/2)) → sin(x)
// and (1-tan²(x/2))/(1+tan²(x/2)) → cos(x)
pub struct WeierstrassContractionRule;

fn format_weierstrass_contraction_desc(kind: WeierstrassContractionKind) -> &'static str {
    match kind {
        WeierstrassContractionKind::Sin => "2·tan(x/2)/(1 + tan²(x/2)) = sin(x)",
        WeierstrassContractionKind::Cos => "(1 - tan²(x/2))/(1 + tan²(x/2)) = cos(x)",
    }
}

impl crate::rule::Rule for WeierstrassContractionRule {
    fn name(&self) -> &str {
        "Weierstrass Half-Angle Contraction"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let rewrite = try_rewrite_weierstrass_contraction_div_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_weierstrass_contraction_desc(rewrite.kind)),
        )
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::DIV)
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

define_rule!(
    CotHalfAngleDifferenceRule,
    "Cotangent Half-Angle Difference",
    |ctx, expr| {
        let rewrite = try_rewrite_cot_half_angle_difference_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(format_cot_half_angle_difference_desc(rewrite.kind)),
        )
    }
);

fn format_cot_half_angle_difference_desc(kind: CotHalfAngleDifferenceRewriteKind) -> &'static str {
    match kind {
        CotHalfAngleDifferenceRewriteKind::Positive => "cot(u/2) - cot(u) = 1/sin(u)",
        CotHalfAngleDifferenceRewriteKind::Negative => "-cot(u/2) + cot(u) = -1/sin(u)",
    }
}

// =============================================================================
// TanDifferenceRule: tan(a - b) → (tan(a) - tan(b)) / (1 + tan(a)*tan(b))
// =============================================================================

define_rule!(TanDifferenceRule, "Tangent Difference", |ctx, expr| {
    let rewrite = try_rewrite_tan_difference_expr(ctx, expr)?;
    Some(Rewrite::new(rewrite.rewritten).desc("tan(a-b) = (tan(a)-tan(b))/(1+tan(a)·tan(b))"))
});

// =============================================================================
// HyperbolicTanhPythRule: 1 - tanh(x)² → 1/cosh(x)² (sech²)
// =============================================================================
// Canonical direction: contract to reciprocal form.

define_rule!(
    HyperbolicTanhPythRule,
    "Hyperbolic Tanh Pythagorean",
    |ctx, expr| {
        let rewrite = try_rewrite_tanh_pythagorean_add_chain(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc("1 - tanh²(x) = 1/cosh²(x)"))
    }
);

// =============================================================================
// HyperbolicHalfAngleSquaresRule: cosh(x/2)² → (cosh(x)+1)/2, sinh(x/2)² → (cosh(x)-1)/2
