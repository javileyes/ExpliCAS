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

        use crate::implicit_domain::ImplicitCondition;
        use crate::rule::ChainedRewrite;

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
mod tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_formatter::DisplayExpr;
    use num_bigint::BigInt;
    use num_rational::BigRational;

    #[test]
    fn test_cancel_cube_root_difference_basic() {
        let mut ctx = Context::new();

        // Build: (x - 27) / (x^(2/3) + 3*x^(1/3) + 9)
        let x = ctx.var("x");
        let c27 = ctx.num(27);
        let c3 = ctx.num(3);
        let c9 = ctx.num(9);

        // Numerator: x - 27 as Add(x, Neg(27))
        let neg_27 = ctx.add(Expr::Neg(c27));
        let num = ctx.add(Expr::Add(x, neg_27));

        // Exponents
        let one_third = ctx.add(Expr::Number(BigRational::new(
            BigInt::from(1),
            BigInt::from(3),
        )));
        let two_thirds = ctx.add(Expr::Number(BigRational::new(
            BigInt::from(2),
            BigInt::from(3),
        )));

        // Denominator terms
        let x_2_3 = ctx.add(Expr::Pow(x, two_thirds));
        let x_1_3 = ctx.add(Expr::Pow(x, one_third));
        let term_mid = ctx.add(Expr::Mul(c3, x_1_3));

        // Den: x^(2/3) + 3*x^(1/3) + 9
        let den_partial = ctx.add(Expr::Add(x_2_3, term_mid));
        let den = ctx.add(Expr::Add(den_partial, c9));

        // Full expression
        let expr = ctx.add(Expr::Div(num, den));

        let rule = CancelCubeRootDifferenceRule;
        let result = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );

        assert!(result.is_some(), "Rule should match this pattern");

        let rewrite = result.unwrap();
        let result_str = format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        );

        // Should be x^(1/3) - 3 or equivalent
        println!("Result: {}", result_str);
        assert!(
            result_str.contains("x") && result_str.contains("1/3"),
            "Result should contain cube root of x: got {}",
            result_str
        );
    }
}
