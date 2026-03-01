//! Multi-angle expansion rules (triple, quintuple, recursive) and trig-quotient helpers.
//!
//! Extracted from `expansion_rules.rs` to keep module size manageable.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_multi_angle_support::{
    should_block_high_order_trig_expansion_expr, try_rewrite_canonicalize_trig_square_pow_expr,
    try_rewrite_quintuple_angle_expr, try_rewrite_recursive_trig_expansion_expr,
    try_rewrite_triple_angle_contraction_expr, try_rewrite_triple_angle_expr,
};

// Triple Angle Shortcut Rule: sin(3x) → 3sin(x) - 4sin³(x), cos(3x) → 4cos³(x) - 3cos(x)
// This is a performance optimization to avoid recursive expansion via double-angle rules.
// Reduces ~23 rewrites to ~3-5 for triple angle expressions.
define_rule!(
    TripleAngleRule,
    "Triple Angle Identity",
    |ctx, expr, parent_ctx| {
        if should_block_high_order_trig_expansion_expr(
            ctx,
            expr,
            parent_ctx.pattern_marks(),
            parent_ctx.all_ancestors(),
            false,
        ) {
            return None;
        }

        let rewrite = try_rewrite_triple_angle_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// Quintuple Angle Rule: sin(5x) → 16sin⁵(x) - 20sin³(x) + 5sin(x)
// This is a direct expansion to avoid recursive explosion via double/triple angle.
define_rule!(
    QuintupleAngleRule,
    "Quintuple Angle Identity",
    |ctx, expr, parent_ctx| {
        if should_block_high_order_trig_expansion_expr(
            ctx,
            expr,
            parent_ctx.pattern_marks(),
            parent_ctx.all_ancestors(),
            false,
        ) {
            return None;
        }

        let rewrite = try_rewrite_quintuple_angle_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    RecursiveTrigExpansionRule,
    "Recursive Trig Expansion",
    |ctx, expr, parent_ctx| {
        if should_block_high_order_trig_expansion_expr(
            ctx,
            expr,
            parent_ctx.pattern_marks(),
            parent_ctx.all_ancestors(),
            true,
        ) {
            return None;
        }

        let rewrite = try_rewrite_recursive_trig_expansion_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    CanonicalizeTrigSquareRule,
    "Canonicalize Trig Square",
    importance: crate::step::ImportanceLevel::Low,
    |ctx, expr| {
        let rewrite = try_rewrite_canonicalize_trig_square_pow_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// Triple Angle CONTRACTION Rule
// =============================================================================
//
// Contracts the expanded form back into a triple angle:
//   3·sin(θ) − 4·sin³(θ)  →  sin(3θ)
//   4·cos³(θ) − 3·cos(θ)  →  cos(3θ)
//
// This is the reverse of TripleAngleRule. It fires on the additive form (Sub/Add
// nodes) and looks for matching pairs of linear and cubic trig terms.
//
// Cycle safety: the contraction produces sin(3θ) or cos(3θ). If the argument
// θ is compound (e.g. u²+1), distribution rewrites 3θ → 3u²+3, which no
// longer matches the Mul(3,x) pattern required by TripleAngleRule, so no
// reverse expansion occurs → no cycle.

define_rule!(
    TripleAngleContractionRule,
    "Triple Angle Contraction",
    |ctx, expr| {
        let rewrite = try_rewrite_triple_angle_contraction_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);
