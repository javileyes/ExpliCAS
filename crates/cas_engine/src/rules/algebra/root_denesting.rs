//! Root denesting rules.
//!
//! Contains `CubicConjugateTrapRule` and its helper functions.

use crate::define_rule;
use crate::rule::Rewrite;
use cas_ast::Expr;

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
        use num_traits::Zero;

        // Match Add(Pow(A, 1/3), Pow(B, 1/3))
        let (left, right) = match ctx.get(expr) {
            Expr::Add(l, r) => (*l, *r),
            _ => return None,
        };

        // Extract cube root bases
        let base_a = cas_math::root_forms::extract_cube_root_base(ctx, left)?;
        let base_b = cas_math::root_forms::extract_cube_root_base(ctx, right)?;

        // Check if A and B are conjugates (m + t) and (m - t)
        let (m, t) = cas_math::root_forms::conjugate_numeric_surd_pair(ctx, base_a, base_b)?;

        // Compute S = A + B = 2m (directly, without simplify)
        // Since A = m + t and B = m - t, A + B = 2m
        let two = num_rational::BigRational::from_integer(2.into());

        // m must be a rational number for this to work
        let m_val = if let Expr::Number(n) = ctx.get(m) {
            n.clone()
        } else {
            return None; // m is not numeric, can't apply
        };

        let s_val = &two * &m_val; // S = 2m

        // Compute AB = m² - t² (directly)
        // t must also allow us to compute t² as rational
        // For t = sqrt(d) or k*sqrt(d), t² is rational
        let t_squared_val = cas_math::root_forms::surd_square_rational(ctx, t)?;

        let ab_val = &m_val * &m_val - &t_squared_val; // AB = m² - t²

        // P = ∛(AB) must be rational (perfect cube)
        let p_val = cas_math::root_forms::rational_cbrt_exact(&ab_val)?;

        // Form depressed cubic: x³ + px + q = 0
        // where p_coef = -3P and q_coef = -S
        // x³ - 3Px - S = 0  =>  x³ + (-3P)x + (-S) = 0
        let three = num_rational::BigRational::from_integer(3.into());
        let p_coef = -&three * &p_val; // coefficient of x
        let q_coef = -&s_val; // constant term

        // Guard: if p_coef > 0, cubic is strictly increasing => unique real root
        // This ensures we can trust the RRT result
        if p_coef <= num_rational::BigRational::zero() {
            return None; // Multiple real roots possible, skip
        }

        // Find rational root via RRT
        let root = cas_math::root_forms::find_rational_root_depressed_cubic(&p_coef, &q_coef)?;

        // Success! Return the root as the result
        let result = ctx.add(Expr::Number(root.clone()));

        Some(
            Rewrite::new(result)
                .desc_lazy(|| format!("Cubic conjugate identity: ∛(m+t) + ∛(m-t) = {}", root)),
        )
    }
);

#[cfg(test)]
mod cubic_conjugate_tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn test_cubic_conjugate_basic() {
        let mut ctx = Context::new();
        let expr = parse("(2 + 5^(1/2))^(1/3) + (2 - 5^(1/2))^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_cubic_conjugate_commuted() {
        let mut ctx = Context::new();
        // Reversed order
        let expr = parse("(2 - 5^(1/2))^(1/3) + (2 + 5^(1/2))^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_cubic_conjugate_no_match_different_surd() {
        let mut ctx = Context::new();
        // Different surds: sqrt(5) vs sqrt(6)
        let expr = parse("(2 + 5^(1/2))^(1/3) + (2 - 6^(1/2))^(1/3)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match different surds");
    }

    #[test]
    fn test_cubic_conjugate_no_match_different_exp() {
        let mut ctx = Context::new();
        // Different exponents: 1/3 vs 1/5
        let expr = parse("(2 + 5^(1/2))^(1/3) + (2 - 5^(1/2))^(1/5)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match different exponents");
    }

    #[test]
    fn test_prerequisite_negative_cube_root() {
        // Prerequisite: (-1)^(1/3) must equal -1 for the rule to work
        let mut ctx = Context::new();
        let expr = parse("(-1)^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "-1");
    }

    #[test]
    fn test_prerequisite_negative_8_cube_root() {
        // (-8)^(1/3) = -2
        let mut ctx = Context::new();
        let expr = parse("(-8)^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "-2");
    }

    #[test]
    fn test_cubic_conjugate_sqrt_function_form() {
        // Test with sqrt() function instead of ^(1/2)
        let mut ctx = Context::new();
        let expr = parse("(2 + sqrt(5))^(1/3) + (2 - sqrt(5))^(1/3)", &mut ctx).unwrap();

        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        let result_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        assert_eq!(result_str, "1");
    }

    #[test]
    fn test_cubic_conjugate_no_match_not_sum() {
        // Subtraction instead of sum - should not match
        let mut ctx = Context::new();
        let expr = parse("(2 + 5^(1/2))^(1/3) - (2 - 5^(1/2))^(1/3)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match subtraction");
    }

    #[test]
    fn test_cubic_conjugate_no_match_same_signs() {
        // Both addends have same sign: (m+t) + (m+t) style
        let mut ctx = Context::new();
        let expr = parse("(2 + 5^(1/2))^(1/3) + (2 + 5^(1/2))^(1/3)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(
            rewrite.is_none(),
            "Should not match when both have same sign"
        );
    }

    #[test]
    fn test_cubic_conjugate_no_match_irrational_root() {
        // (1 + √2)^(1/3) + (1 - √2)^(1/3)
        // AB = 1 - 2 = -1 is a cube, but cubic x³ + 3x - 2 = 0 has no rational root
        let mut ctx = Context::new();
        let expr = parse("(1 + 2^(1/2))^(1/3) + (1 - 2^(1/2))^(1/3)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        // The rule should not find a rational root (root is ~0.596)
        assert!(
            rewrite.is_none(),
            "Should not match when no rational root exists"
        );
    }

    #[test]
    fn test_cubic_conjugate_no_match_different_m() {
        // Different m values: (2+√5)^(1/3) + (3-√5)^(1/3)
        let mut ctx = Context::new();
        let expr = parse("(2 + 5^(1/2))^(1/3) + (3 - 5^(1/2))^(1/3)", &mut ctx).unwrap();

        let rule = CubicConjugateTrapRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match different m values");
    }
}
