//! Power products and sum-to-product quotient rules.

use crate::define_rule;
use crate::rule::ChainedRewrite;
use crate::rule::Rewrite;
use cas_math::trig_power_identity_support::{
    try_rewrite_hidden_cubic_identity_add_expr, try_rewrite_sin_cos_quartic_sum_add_expr,
};
use cas_math::trig_sum_product_support::try_plan_sin_cos_sum_quotient_div_expr;

// =============================================================================
// HIDDEN CUBIC TRIG IDENTITY
// sin^6(x) + cos^6(x) + 3*sin^2(x)*cos^2(x) = (sin^2(x) + cos^2(x))^3
// =============================================================================

define_rule!(
    TrigHiddenCubicIdentityRule,
    "Hidden Cubic Trig Identity",
    None,
    crate::phase::PhaseMask::TRANSFORM,
    |ctx, expr| {
        let rewrite = try_rewrite_hidden_cubic_identity_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// QUARTIC TRIG IDENTITY
// sin^4(x) + cos^4(x) = 1 − 2·sin²(x)·cos²(x)
//
// Derivation: (sin²+cos²)² = sin⁴ + 2sin²cos² + cos⁴ = 1
// Therefore:  sin⁴ + cos⁴ = 1 − 2sin²cos²
// =============================================================================

define_rule!(
    SinCosQuarticSumRule,
    "Quartic Pythagorean Identity",
    None,
    crate::phase::PhaseMask::CORE,
    |ctx, expr| {
        let rewrite = try_rewrite_sin_cos_quartic_sum_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// SUM-TO-PRODUCT QUOTIENT RULE
// (sin(A)+sin(B))/(cos(A)+cos(B)) → sin((A+B)/2)/cos((A+B)/2)
// =============================================================================

// SinCosSumQuotientRule: Handles two patterns:
// 1. (sin(A)+sin(B))/(cos(A)+cos(B)) → tan((A+B)/2)  [uses sin sum identity]
// 2. (sin(A)-sin(B))/(cos(A)+cos(B)) → tan((A-B)/2)  [uses sin diff identity]
//
// Sum-to-product identities:
//   sin(A) + sin(B) = 2·sin((A+B)/2)·cos((A-B)/2)
//   sin(A) - sin(B) = 2·cos((A+B)/2)·sin((A-B)/2)
//   cos(A) + cos(B) = 2·cos((A+B)/2)·cos((A-B)/2)
//
// For sum case: common factor is 2·cos((A-B)/2) → result is tan((A+B)/2)
// For diff case: common factor is 2·cos((A+B)/2) → result is tan((A-B)/2)
//
// This rule runs BEFORE TripleAngleRule to avoid polynomial explosion.
define_rule!(
    SinCosSumQuotientRule,
    "Sum-to-Product Quotient",
    |ctx, expr| {
        let plan = try_plan_sin_cos_sum_quotient_div_expr(ctx, expr, crate::collect::collect)?;
        Some(
            Rewrite::new(plan.state_after_step1)
                .desc(plan.desc_step1)
                .local(plan.num_id, plan.intermediate_num)
                .chain(
                    ChainedRewrite::new(plan.state_after_step2)
                        .desc(plan.desc_step2)
                        .local(plan.den_id, plan.intermediate_den),
                )
                .chain(
                    ChainedRewrite::new(plan.rewritten)
                        .desc(plan.desc_step3)
                        .local(plan.state_after_step2, plan.rewritten),
                ),
        )
    }
);
