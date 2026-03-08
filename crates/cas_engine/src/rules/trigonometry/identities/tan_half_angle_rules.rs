//! Tan half-angle and trigonometric quotient rules.
//!
//! This module contains rules for:
//! - Hyperbolic half-angle identities: cosh²(x/2) = (cosh(x)+1)/2
//! - Generalized sin·cos contraction: k·sin(t)·cos(t) = (k/2)·sin(2t)
//! - Trig quotient simplification: sin/cos → tan
//! - Tan double angle contraction

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::ExprId;
use cas_math::trig_canonicalization_support::try_rewrite_trig_quotient_div_expr;
use cas_math::trig_contraction_support::{
    try_rewrite_generalized_sin_cos_contraction_expr, try_rewrite_tan_double_angle_contraction_expr,
};
use cas_math::trig_half_angle_support::{
    try_rewrite_hyperbolic_half_angle_squares_expr, try_rewrite_trig_half_angle_squares_expr,
    HalfAngleSquareRewriteKind,
};

// =============================================================================

fn format_half_angle_square_desc(kind: HalfAngleSquareRewriteKind) -> &'static str {
    match kind {
        HalfAngleSquareRewriteKind::HyperbolicCosh => "cosh²(x/2) = (cosh(x)+1)/2",
        HalfAngleSquareRewriteKind::HyperbolicSinh => "sinh²(x/2) = (cosh(x)-1)/2",
        HalfAngleSquareRewriteKind::TrigSin => "sin²(x/2) = (1 - cos(x))/2",
        HalfAngleSquareRewriteKind::TrigCos => "cos²(x/2) = (1 + cos(x))/2",
    }
}

fn format_generalized_sin_cos_contraction_desc() -> &'static str {
    "k·sin(t)·cos(t) = (k/2)·sin(2t)"
}

fn format_tan_double_angle_contraction_desc() -> &'static str {
    "2·tan(t)/(1-tan²(t)) = tan(2t)"
}

define_rule!(
    HyperbolicHalfAngleSquaresRule,
    "Hyperbolic Half-Angle Squares",
    |ctx, expr| {
        let rewrite = try_rewrite_hyperbolic_half_angle_squares_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_half_angle_square_desc(rewrite.kind)))
    }
);

// =============================================================================
// TrigHalfAngleSquaresRule: sin(x/2)² → (1 - cos(x))/2, cos(x/2)² → (1 + cos(x))/2
// =============================================================================
// Trig analogue of HyperbolicHalfAngleSquaresRule. Only applies to squared forms
// (no sqrt branching). TRANSFORM-only to prevent oscillation with
// Cos2xAdditiveContractionRule (POST-only, contracts 1-2sin²→cos(2t)).
//
// Also matches Mul(sin(x/2), sin(x/2)) and Mul(cos(x/2), cos(x/2)) for cases
// where the AST uses product form instead of Pow.

pub struct TrigHalfAngleSquaresRule;

impl crate::rule::Rule for TrigHalfAngleSquaresRule {
    fn name(&self) -> &str {
        "Trig Half-Angle Squares"
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: ExprId,
        _parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<crate::rule::Rewrite> {
        let rewrite = try_rewrite_trig_half_angle_squares_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_half_angle_square_desc(rewrite.kind)))
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        // TRANSFORM only: prevents oscillation with Cos2xAdditiveContractionRule (POST-only)
        crate::phase::PhaseMask::TRANSFORM
    }

    fn target_types(&self) -> Option<crate::target_kind::TargetKindSet> {
        Some(crate::target_kind::TargetKindSet::POW.union(crate::target_kind::TargetKindSet::MUL))
    }

    fn importance(&self) -> crate::step::ImportanceLevel {
        crate::step::ImportanceLevel::High
    }
}

// =============================================================================
// GeneralizedSinCosContractionRule: k*sin(t)*cos(t) → (k/2)*sin(2t) for even k
// =============================================================================
// Extends DoubleAngleContractionRule to handle k*sin*cos where k is even (4, 6, 8, etc.)

define_rule!(
    GeneralizedSinCosContractionRule,
    "Generalized Sin Cos Contraction",
    |ctx, expr| {
        let rewrite = try_rewrite_generalized_sin_cos_contraction_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_generalized_sin_cos_contraction_desc()))
    }
);

// =============================================================================
// TrigQuotientToNamedRule: sin(t)/cos(t) → tan(t), 1/cos(t) → sec(t), etc.
// =============================================================================
// Canonicalize trig quotients to named functions for better normalization.
// This ensures that `sin(u)/cos(u)` and `tan(u)` converge to the same form.

define_rule!(
    TrigQuotientToNamedRule,
    "Trig Quotient to Named Function",
    |ctx, expr| {
        let rewrite = try_rewrite_trig_quotient_div_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// TanDoubleAngleContractionRule: 2*tan(t)/(1 - tan(t)²) → tan(2*t)
// =============================================================================
// This contracts the expanded tan(2t) form back to the double angle form.
// Prevents the engine from creating deeply nested fractions when tan²(t)
// appears in denominators.

define_rule!(
    TanDoubleAngleContractionRule,
    "Tan Double Angle Contraction",
    |ctx, expr| {
        let rewrite = try_rewrite_tan_double_angle_contraction_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_tan_double_angle_contraction_desc()))
    }
);
