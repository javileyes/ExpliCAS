//! Core trigonometric identity rules for evaluation at special values.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_core_identity_support::{
    should_block_angle_identity_expr, try_rewrite_angle_sum_diff_identity_expr,
    try_rewrite_legacy_evaluate_trig_expr, try_rewrite_pythagorean_identity_add_expr,
    try_rewrite_sin_cos_integer_pi_expr, try_rewrite_trig_odd_even_parity_expr,
    AngleSumDiffIdentityKind, LegacyTrigEvalRewriteKind, PythagoreanIdentityRewriteKind,
    SinCosIntegerPiKind, TrigOddEvenParityKind,
};

fn format_trig_odd_even_parity_desc(fn_name: &str, kind: TrigOddEvenParityKind) -> String {
    match kind {
        TrigOddEvenParityKind::Odd => format!("{fn_name}(-u) = -{fn_name}(u) [odd function]"),
        TrigOddEvenParityKind::Even => format!("{fn_name}(-u) = {fn_name}(u) [even function]"),
    }
}

fn format_sin_cos_integer_pi_desc(kind: SinCosIntegerPiKind, n: &num_bigint::BigInt) -> String {
    match kind {
        SinCosIntegerPiKind::SinIntegerPi => format!("sin({n}·π) = 0"),
        SinCosIntegerPiKind::CosIntegerPiEven => format!("cos({n}·π) = 1"),
        SinCosIntegerPiKind::CosIntegerPiOdd => format!("cos({n}·π) = -1"),
    }
}

fn format_legacy_trig_eval_desc(kind: &LegacyTrigEvalRewriteKind) -> String {
    match kind {
        LegacyTrigEvalRewriteKind::TableEval { fn_name } => {
            format!("{fn_name}(...) evaluated via table")
        }
        LegacyTrigEvalRewriteKind::ZeroEval { fn_name } => format!("{fn_name}(0) = 0"),
        LegacyTrigEvalRewriteKind::CosZero => "cos(0) = 1".to_string(),
        LegacyTrigEvalRewriteKind::ArccosZero => "arccos(0) = pi/2".to_string(),
        LegacyTrigEvalRewriteKind::ArcsinOne => "arcsin(1) = pi/2".to_string(),
        LegacyTrigEvalRewriteKind::ArccosOne => "arccos(1) = 0".to_string(),
        LegacyTrigEvalRewriteKind::ArctanOne => "arctan(1) = pi/4".to_string(),
        LegacyTrigEvalRewriteKind::ArcsinHalf => "arcsin(1/2) = pi/6".to_string(),
        LegacyTrigEvalRewriteKind::ArccosHalf => "arccos(1/2) = pi/3".to_string(),
        LegacyTrigEvalRewriteKind::PiZero { fn_name } => format!("{fn_name}(pi) = 0"),
        LegacyTrigEvalRewriteKind::CosPi => "cos(pi) = -1".to_string(),
        LegacyTrigEvalRewriteKind::SinPiOver2 => "sin(pi/2) = 1".to_string(),
        LegacyTrigEvalRewriteKind::CosPiOver2 => "cos(pi/2) = 0".to_string(),
        LegacyTrigEvalRewriteKind::TanPiOver2 => "tan(pi/2) = undefined".to_string(),
        LegacyTrigEvalRewriteKind::SinPiOver3 => "sin(π/3) = √3/2".to_string(),
        LegacyTrigEvalRewriteKind::CosPiOver3 => "cos(π/3) = 1/2".to_string(),
        LegacyTrigEvalRewriteKind::TanPiOver3 => "tan(π/3) = √3".to_string(),
        LegacyTrigEvalRewriteKind::PiOver4Trig { fn_name } => format!("{fn_name}(π/4) = √2/2"),
        LegacyTrigEvalRewriteKind::TanPiOver4 => "tan(π/4) = 1".to_string(),
        LegacyTrigEvalRewriteKind::SinPiOver6 => "sin(π/6) = 1/2".to_string(),
        LegacyTrigEvalRewriteKind::CosPiOver6 => "cos(π/6) = √3/2".to_string(),
        LegacyTrigEvalRewriteKind::TanPiOver6 => "tan(π/6) = 1/√3".to_string(),
        LegacyTrigEvalRewriteKind::SinNegative => "sin(-x) = -sin(x)".to_string(),
        LegacyTrigEvalRewriteKind::CosNegative => "cos(-x) = cos(x)".to_string(),
        LegacyTrigEvalRewriteKind::TanNegative => "tan(-x) = -tan(x)".to_string(),
    }
}

fn format_pythagorean_identity_desc(kind: PythagoreanIdentityRewriteKind) -> &'static str {
    match kind {
        PythagoreanIdentityRewriteKind::Empty => "Pythagorean Identity (empty)",
        PythagoreanIdentityRewriteKind::Standard => "Pythagorean Identity",
        PythagoreanIdentityRewriteKind::Negated => "Pythagorean Identity (negated)",
    }
}

fn format_angle_sum_diff_identity_desc(kind: AngleSumDiffIdentityKind) -> &'static str {
    match kind {
        AngleSumDiffIdentityKind::SinAdd => "sin(a + b) -> sin(a)cos(b) + cos(a)sin(b)",
        AngleSumDiffIdentityKind::SinSub => "sin(a - b) -> sin(a)cos(b) - cos(a)sin(b)",
        AngleSumDiffIdentityKind::SinDivAdd => {
            "sin((a + b)/c) -> sin(a/c)cos(b/c) + cos(a/c)sin(b/c)"
        }
        AngleSumDiffIdentityKind::CosAdd => "cos(a + b) -> cos(a)cos(b) - sin(a)sin(b)",
        AngleSumDiffIdentityKind::CosSub => "cos(a - b) -> cos(a)cos(b) + sin(a)sin(b)",
        AngleSumDiffIdentityKind::CosDivAdd => {
            "cos((a + b)/c) -> cos(a/c)cos(b/c) - sin(a/c)sin(b/c)"
        }
    }
}

// =============================================================================
// SinCosIntegerPiRule: Pre-order evaluation of sin(n·π) and cos(n·π)
// =============================================================================
// sin(n·π) = 0 for any integer n
// cos(n·π) = (-1)^n for any integer n
//
// This rule runs BEFORE any expansion rules (TripleAngle, DoubleAngle, etc.)
// to avoid unnecessary polynomial expansion of expressions like sin(3π).
//
// Priority: 100 (higher than most rules to ensure pre-order evaluation)

define_rule!(
    SinCosIntegerPiRule,
    "Evaluate Trig at Integer Multiple of π",
    priority: 100, // Run before expansion rules
    importance: crate::step::ImportanceLevel::High,
    |ctx, expr| {
        let rewrite = try_rewrite_sin_cos_integer_pi_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten)
                .desc(format_sin_cos_integer_pi_desc(rewrite.kind, &rewrite.n)),
        )
    }
);

// =============================================================================
// TrigOddEvenParityRule: sin(-u) = -sin(u), cos(-u) = cos(u), tan(-u) = -tan(u)
// =============================================================================
// sin, tan, csc, cot are ODD functions: f(-x) = -f(x)
// cos, sec are EVEN functions: f(-x) = f(x)
//
// This enables simplification of expressions like sin(pi/9)/sin(-pi/9) → -1

define_rule!(
    TrigOddEvenParityRule,
    "Trig Parity (Odd/Even)",
    |ctx, expr| {
        let rewrite = try_rewrite_trig_odd_even_parity_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_trig_odd_even_parity_desc(
                &rewrite.fn_name,
                rewrite.kind,
            )),
        )
    }
);

// NOTE: EvaluateTrigRule is deprecated - use EvaluateTrigTableRule from evaluation.rs instead
// This rule is kept for reference but should not be registered in the simplifier.
define_rule!(
    EvaluateTrigRule,
    "Evaluate Trigonometric Functions",
    |ctx, expr| {
        let rewrite = try_rewrite_legacy_evaluate_trig_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_legacy_trig_eval_desc(&rewrite.kind)))
    }
);

define_rule!(
    PythagoreanIdentityRule,
    "Pythagorean Identity",
    None,
    crate::phase::PhaseMask::TRANSFORM, // Match phase with TrigHiddenCubicIdentityRule
    |ctx, expr| {
        let rewrite = try_rewrite_pythagorean_identity_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(format_pythagorean_identity_desc(rewrite.kind)))
    }
);

// MANUAL IMPLEMENTATION: AngleIdentityRule with parent context guard
// Policy: angle-sum expansion is EXPAND-MODE ONLY.
// sin(a+b)/cos(a+b) expansion produces larger, non-canonical forms
// (e.g. cos(u²)·cos(1) - sin(u²)·sin(1)) that block convergence.
// Within expand_mode, keep additional guards to prevent blowups/loops.
pub struct AngleIdentityRule;

impl crate::rule::Rule for AngleIdentityRule {
    fn name(&self) -> &str {
        "Angle Sum/Diff Identity"
    }

    fn allowed_phases(&self) -> crate::phase::PhaseMask {
        crate::phase::PhaseMask::TRANSFORM
    }

    fn apply(
        &self,
        ctx: &mut cas_ast::Context,
        expr: cas_ast::ExprId,
        parent_ctx: &crate::parent_context::ParentContext,
    ) -> Option<Rewrite> {
        if should_block_angle_identity_expr(
            ctx,
            expr,
            parent_ctx.is_expand_mode(),
            parent_ctx.pattern_marks(),
            parent_ctx.is_trig_large_coeff_protected(),
        ) {
            return None;
        }

        let rewrite = try_rewrite_angle_sum_diff_identity_expr(ctx, expr)?;
        Some(
            Rewrite::new(rewrite.rewritten).desc(format_angle_sum_diff_identity_desc(rewrite.kind)),
        )
    }
}
