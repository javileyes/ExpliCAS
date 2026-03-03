//! Expansion and contraction rules for trigonometric expressions.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_multi_angle_support::{
    should_block_double_angle_expr_with_context, try_rewrite_double_angle_function_expr,
};
use cas_math::trig_sum_product_support::try_rewrite_trig_sum_to_product_expr;

// =============================================================================
// STANDALONE SUM-TO-PRODUCT RULE
// sin(A)+sin(B) → 2·sin((A+B)/2)·cos((A-B)/2)
// sin(A)-sin(B) → 2·cos((A+B)/2)·sin((A-B)/2)
// cos(A)+cos(B) → 2·cos((A+B)/2)·cos((A-B)/2)
// cos(A)-cos(B) → -2·sin((A+B)/2)·sin((A-B)/2)
// =============================================================================
// This rule applies sum-to-product identities to standalone sums/differences
// of trig functions (not inside quotients handled by SinCosSumQuotientRule).
//
// GATING: Only apply when both arguments are rational multiples of π, ensuring
// the transformed expression can be evaluated via trig table lookup (π/4, π/6, etc.)
// This prevents unnecessary expansion of symbolic expressions like sin(a)+sin(b).
//
// MATCHERS: Uses semantic TrigSumMatch (unordered) and TrigDiffMatch (ordered)
// to ensure correct sign handling for difference identities.
define_rule!(
    TrigSumToProductRule,
    "Sum-to-Product Identity",
    |ctx, expr| {
        let rewrite = try_rewrite_trig_sum_to_product_expr(ctx, expr, crate::collect::collect)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

define_rule!(
    DoubleAngleRule,
    "Double Angle Identity",
    |ctx, expr, parent_ctx| {
        if should_block_double_angle_expr_with_context(
            ctx,
            parent_ctx.is_expand_mode(),
            parent_ctx.pattern_marks(),
            parent_ctx.all_ancestors(),
        ) {
            return None;
        }

        let rewrite = try_rewrite_double_angle_function_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);
