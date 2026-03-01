//! Core trigonometric identity rules for evaluation at special values.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_core_identity_support::{
    should_block_angle_identity_large_coeff, should_block_angle_identity_multiple_angle,
    should_block_angle_identity_non_expand_mode, try_rewrite_angle_sum_diff_identity_expr,
    try_rewrite_legacy_evaluate_trig_expr, try_rewrite_pythagorean_identity_add_expr,
    try_rewrite_sin_cos_integer_pi_expr, try_rewrite_trig_odd_even_parity_expr,
};

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
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
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
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// NOTE: EvaluateTrigRule is deprecated - use EvaluateTrigTableRule from evaluation.rs instead
// This rule is kept for reference but should not be registered in the simplifier.
define_rule!(
    EvaluateTrigRule,
    "Evaluate Trigonometric Functions",
    |ctx, expr| {
        let rewrite = try_rewrite_legacy_evaluate_trig_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    PythagoreanIdentityRule,
    "Pythagorean Identity",
    None,
    crate::phase::PhaseMask::TRANSFORM, // Match phase with TrigHiddenCubicIdentityRule
    |ctx, expr| {
        let rewrite = try_rewrite_pythagorean_identity_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
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
        // SURGICAL GATE: in normal simplify mode, only expand sin(a±b) / cos(a±b)
        // when BOTH summands contain at least one variable.
        // This allows canonical expansions like:
        //   sin(x+y) → sin(x)cos(y) + cos(x)sin(y)  ✅ (NF convergence)
        // But blocks non-canonical expansions that create cos/sin(constant) leaves:
        //   sin(u²+1) → cos(1)·sin(u²) + sin(1)·cos(u²)  ❌
        //   sin(x+π/4) → ...  ❌  (handled by EvaluateTrigTableRule instead)
        // In expand_mode, all expansions are allowed (with existing size guards).
        if !parent_ctx.is_expand_mode() && should_block_angle_identity_non_expand_mode(ctx, expr) {
            return None;
        }

        // --- Expand-mode anti-catastrophe guards (defense-in-depth) ---

        // GUARD 1: Don't expand sin/cos if the argument has a large coefficient.
        // sin(n*x) with |n| > 2 should NOT be expanded because it leads to
        // exponential explosion: sin(16x) → sin(13x+3x) → ... huge tree.
        if should_block_angle_identity_large_coeff(ctx, expr) {
            return None;
        }

        // GUARD 2: Don't expand sin(a+b)/cos(a+b) if part of sin²+cos²=1 pattern
        // The pattern marks are set by pre-scan before simplification
        if let Some(marks) = parent_ctx.pattern_marks() {
            if marks.is_trig_square_protected(expr) {
                return None; // Skip expansion to preserve Pythagorean identity
            }
        }

        // GUARD 3: Centralized anti-worsen for large trig coefficients.
        // If we're inside sin(n*x) with |n| > 2, block all trig expansions.
        // This prevents exponential explosion from recursive angle decomposition.
        if parent_ctx.is_trig_large_coeff_protected() {
            return None;
        }

        // GUARD 4: Anti-worsen for multiple angles.
        // Don't expand sin(a+b) or cos(a+b) if:
        // - Either a or b is already a multiple angle (n*x where |n| > 1)
        // - This would cause exponential expansion: sin(12x + 4x) → huge tree
        if should_block_angle_identity_multiple_angle(ctx, expr) {
            return None; // Block expansion - would cause exponential growth
        }

        let rewrite = try_rewrite_angle_sum_diff_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
}
