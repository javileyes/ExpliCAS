//! Difference of Cubes Simplification for Cube Root Expressions
//!
//! This module provides a specialized pre-order rule for simplifying quotients
//! that match the "difference of cubes" factorization pattern with cube roots.
//!
//! ## Pattern:
//! ```text
//! (x - b³) / (x^(2/3) + b·x^(1/3) + b²) → x^(1/3) - b
//! ```

use crate::define_rule;
use crate::phase::PhaseMask;
use crate::rule::Rewrite;
use cas_ast::Expr;
use cas_math::difference_of_cubes_support::try_plan_cancel_cube_root_difference_expr;

// CancelCubeRootDifferenceRule: Simplifies (x - b³) / (x^(2/3) + b·x^(1/3) + b²) → x^(1/3) - b
//
// This is a pre-order rule that catches the specific algebraic pattern before
// the general fraction simplification machinery can cause oscillation.
define_rule!(
    CancelCubeRootDifferenceRule,
    "Cancel Cube Root Difference",
    None,
    PhaseMask::CORE, // Run early in Core phase
    |ctx, expr| {
        if !matches!(ctx.get(expr), Expr::Div(_, _)) {
            return None;
        }
        let plan = try_plan_cancel_cube_root_difference_expr(ctx, expr)?;
        let b_squared = &plan.b_value * &plan.b_value;

        use crate::rule::ChainedRewrite;
        use crate::ImplicitCondition;

        let factor_rw = Rewrite::new(plan.intermediate)
            .desc(format!(
                "Factor difference of cubes: x - {} = (x^(1/3) - {})·(x^(2/3) + {}·x^(1/3) + {})",
                plan.cube_value, plan.b_value, plan.b_value, b_squared
            ))
            .local(plan.numerator, plan.factored_numerator)
            .requires(ImplicitCondition::NonZero(plan.denominator));

        let cancel = ChainedRewrite::new(plan.final_factor)
            .desc("Cancel common factor")
            .local(plan.intermediate, plan.final_factor);

        Some(factor_rw.chain(cancel))
    }
);

/// Register the difference of cubes rules
pub fn register(simplifier: &mut crate::Simplifier) {
    // Register BEFORE general fraction simplification for pre-order behavior
    simplifier.add_rule(Box::new(CancelCubeRootDifferenceRule));
}

#[cfg(test)]
mod tests;
