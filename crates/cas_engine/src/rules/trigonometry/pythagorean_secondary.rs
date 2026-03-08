//! Secondary Pythagorean-adjacent rules: reciprocal conversions and even-power differences.
//!
//! Extracted from `pythagorean.rs` to reduce file size.
//! - SecToRecipCosRule, CscToRecipSinRule, CotToCosSinRule
//! - TrigEvenPowerDifferenceRule (with extraction delegated to cas_math)

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_canonicalization_support::{
    try_rewrite_cot_to_cos_sin_function_expr, try_rewrite_csc_to_recip_sin_function_expr,
    try_rewrite_sec_to_recip_cos_function_expr,
};
use cas_math::trig_power_identity_support::{
    try_rewrite_trig_fourth_power_difference_add_expr, try_rewrite_trig_fourth_power_sum_add_expr,
};

// =============================================================================
// SecToRecipCosRule: sec(x) → 1/cos(x) (canonical expansion)
// =============================================================================
// This ensures sec unifies with 1/cos forms from tan²+1 simplification.

define_rule!(
    SecToRecipCosRule,
    "Secant to Reciprocal Cosine",
    |ctx, expr| {
        let rewrite = try_rewrite_sec_to_recip_cos_function_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// CscToRecipSinRule: csc(x) → 1/sin(x) (canonical expansion)
// =============================================================================

define_rule!(
    CscToRecipSinRule,
    "Cosecant to Reciprocal Sine",
    |ctx, expr| {
        let rewrite = try_rewrite_csc_to_recip_sin_function_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// CotToCosSinRule: cot(x) → cos(x)/sin(x) (canonical expansion)
// =============================================================================
// This ensures cot unifies with cos/sin forms for csc²-cot²=1 simplification.

define_rule!(
    CotToCosSinRule,
    "Cotangent to Cosine over Sine",
    |ctx, expr| {
        let rewrite = try_rewrite_cot_to_cos_sin_function_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// ============================================================================
// TrigEvenPowerDifferenceRule
// ============================================================================
// Detects pairs like k*sin⁴(u) + (-k)*cos⁴(u) in Add expressions and reduces
// them using the factorization: sin⁴ - cos⁴ = (sin² - cos²)(sin² + cos²) = sin² - cos²
//
// This is a degree-reducing rule (4 → 2) so it cannot loop.

define_rule!(
    TrigEvenPowerDifferenceRule,
    "Trig Fourth Power Difference",
    |ctx, expr| {
        let rewrite = try_rewrite_trig_fourth_power_difference_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// ============================================================================
// TrigEvenPowerSumRule
// ============================================================================
// Detects pairs like k*sin⁴(u) + k*cos⁴(u) in Add expressions and reduces
// them using: sin⁴ + cos⁴ = (sin² + cos²)² - 2·sin²·cos² = 1 - 2·sin²·cos²
//
// So: k·sin⁴(u) + k·cos⁴(u) → k·(1 - 2·sin²(u)·cos²(u))
//
// This is a degree-reducing rule (4 → 2) so it cannot loop.

define_rule!(
    TrigEvenPowerSumRule,
    "Trig Fourth Power Sum",
    |ctx, expr| {
        let rewrite = try_rewrite_trig_fourth_power_sum_add_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);
