//! Root denesting rules.
//!
//! Contains `CubicConjugateTrapRule` and its helper functions.

use crate::define_rule;
use crate::rule::Rewrite;

// =============================================================================
// CUBIC CONJUGATE TRAP RULE
// Simplifies ∛(m+t) + ∛(m-t) when the result is a rational number.
// =============================================================================

define_rule!(
    CubicConjugateTrapRule,
    "Cubic Conjugate Identity",
    None,
    crate::phase::PhaseMask::TRANSFORM,
    |ctx, expr| {
        let rewrite = cas_math::root_forms::try_rewrite_cubic_conjugate_identity_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);
