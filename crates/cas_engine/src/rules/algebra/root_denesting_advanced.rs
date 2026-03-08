//! Advanced root denesting rules.
//!
//! Contains `DenestSqrtAddSqrtRule` (√(a+√b) → √m+√n) and
//! `DenestPerfectCubeInQuadraticFieldRule` (∛(A+B√n) → x+y√n),
//! plus their helper functions.

use crate::define_rule;
use crate::rule::Rewrite;

// =============================================================================
// DENEST SQRT(a + SQRT(b)) RULE
// Simplifies √(a + √b) → √m + √n where m,n = (a ± √(a²-b))/2
// =============================================================================

define_rule!(
    DenestSqrtAddSqrtRule,
    "Denest Nested Square Root",
    None,
    crate::phase::PhaseMask::TRANSFORM,
    |ctx, expr| {
        let rewrite = cas_math::root_forms::try_rewrite_denest_sqrt_add_sqrt_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

// =============================================================================
// DENEST PERFECT CUBE IN QUADRATIC FIELD RULE
// Simplifies ∛(A + B√n) → x + y√n where (x+y√n)³ = A+B√n
// =============================================================================

define_rule!(
    DenestPerfectCubeInQuadraticFieldRule,
    "Denest Cube Root in Quadratic Field",
    None,
    crate::phase::PhaseMask::TRANSFORM,
    |ctx, expr| {
        let rewrite =
            cas_math::root_forms::try_rewrite_denest_cube_quadratic_field_expr(ctx, expr)?;
        Some(Rewrite::new(rewrite.rewritten).desc(rewrite.desc))
    }
);

#[cfg(test)]
mod tests;
