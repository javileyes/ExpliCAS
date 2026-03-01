//! Phase shift and supplementary angle rules.
//!
//! Contains `SinSupplementaryAngleRule`.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_math::trig_phase_shift_support::try_rewrite_supplementary_angle_expr;

// =============================================================================
// Sin Supplementary Angle Rule
// =============================================================================
// sin(π - x) → sin(x)
// sin(k·π - x) → (-1)^(k+1) · sin(x) for integer k
// cos(π - x) → -cos(x)
//
// This enables simplification of expressions like sin(8π/9) = sin(π - π/9) = sin(π/9)

define_rule!(
    SinSupplementaryAngleRule,
    "Supplementary Angle",
    |ctx, expr| {
        let rewrite = try_rewrite_supplementary_angle_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);
