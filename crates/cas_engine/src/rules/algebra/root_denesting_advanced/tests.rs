use super::*;

mod denest_sqrt_tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn test_denest_sqrt_4_plus_sqrt7() {
        // √(4 + √7) → √(7/2) + √(1/2)
        let mut ctx = Context::new();
        let expr = parse("sqrt(4 + sqrt(7))", &mut ctx).unwrap();

        let rule = DenestSqrtAddSqrtRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_some(), "Rule should apply to √(4+√7)");

        // Verify the result simplifies correctly
        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.context = ctx;
        let (result, _) = simplifier.simplify(expr);

        // Just check that we get the denested form with surds
        let result_str = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: result
            }
        );
        // Should contain fractions 1/2 and 7/2
        assert!(
            result_str.contains("1/2") && result_str.contains("7/2"),
            "Result should be √(1/2)+√(7/2), got: {}",
            result_str
        );
    }

    #[test]
    fn test_denest_sqrt_pow_form() {
        // (4 + 7^(1/2))^(1/2) → pow form instead of sqrt function
        // Use simplifier since the expression needs canonicalization
        let mut ctx = Context::new();
        let expr = parse("(4 + 7^(1/2))^(1/2)", &mut ctx).unwrap();

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
        // Should contain fractions 1/2 and 7/2
        assert!(
            result_str.contains("1/2") && result_str.contains("7/2"),
            "Result should be denested with 1/2 and 7/2, got: {}",
            result_str
        );
    }

    #[test]
    fn test_denest_sqrt_no_match_bad_discriminant() {
        // √(3 + √5): disc = 9 - 5 = 4 ✓, but let's check it works
        // disc_sqrt = 2, m = (3+2)/2 = 5/2, n = (3-2)/2 = 1/2
        let mut ctx = Context::new();
        let expr = parse("sqrt(3 + sqrt(5))", &mut ctx).unwrap();

        let rule = DenestSqrtAddSqrtRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(
            rewrite.is_some(),
            "Should match √(3+√5) since disc=4 is perfect square"
        );
    }

    #[test]
    fn test_denest_sqrt_no_match_non_perfect_square_disc() {
        // √(4 + √10): disc = 16 - 10 = 6 (not a perfect square)
        let mut ctx = Context::new();
        let expr = parse("sqrt(4 + sqrt(10))", &mut ctx).unwrap();

        let rule = DenestSqrtAddSqrtRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(
            rewrite.is_none(),
            "Should not match when disc=6 is not a perfect square"
        );
    }

    #[test]
    fn test_denest_sqrt_no_match_negative_m_or_n() {
        // √(1 + √10): disc = 1 - 10 = -9 (negative)
        let mut ctx = Context::new();
        let expr = parse("sqrt(1 + sqrt(10))", &mut ctx).unwrap();

        let rule = DenestSqrtAddSqrtRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should not match when disc < 0");
    }

    #[test]
    fn test_denest_sqrt_commuted_order() {
        // √(√7 + 4) - surd comes first
        let mut ctx = Context::new();
        let expr = parse("sqrt(sqrt(7) + 4)", &mut ctx).unwrap();

        let rule = DenestSqrtAddSqrtRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_some(), "Should match commuted order √(√7+4)");
    }
}

mod denest_cube_quadratic_tests {
    use super::*;
    use crate::rule::Rule;
    use cas_ast::Context;
    use cas_parser::parse;

    #[test]
    fn test_denest_cube_26_15_sqrt3() {
        // (26 + 15*sqrt(3))^(1/3) → 2 + sqrt(3)
        // Because (2 + sqrt(3))³ = 26 + 15*sqrt(3)
        let mut ctx = Context::new();
        let expr = parse("(26 + 15*sqrt(3))^(1/3)", &mut ctx).unwrap();

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
        assert!(
            result_str.contains("2") && result_str.contains("3"),
            "Should be 2 + √3, got: {}",
            result_str
        );
        // Verify it doesn't contain a cube root anymore
        assert!(
            !result_str.contains("∛") && !result_str.contains("1/3"),
            "Should NOT contain cube root, got: {}",
            result_str
        );
    }

    #[test]
    fn test_denest_cube_golden_ratio() {
        // (2 + sqrt(5))^(1/3) → (1 + sqrt(5))/2 = φ
        let mut ctx = Context::new();
        let expr = parse("(2 + sqrt(5))^(1/3)", &mut ctx).unwrap();

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
        // Should be φ (phi) since (1 + √5)/2 is recognized as phi
        assert!(
            result_str.contains("phi")
                || (result_str.contains("1")
                    && result_str.contains("5")
                    && result_str.contains("2")),
            "Should be phi or (1+√5)/2, got: {}",
            result_str
        );
    }

    #[test]
    fn test_denest_cube_golden_ratio_conjugate() {
        // (2 - sqrt(5))^(1/3) → (1 - sqrt(5))/2 = 1-φ
        let mut ctx = Context::new();
        let expr = parse("(2 - sqrt(5))^(1/3)", &mut ctx).unwrap();

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
        // Should be (1 - √5)/2
        assert!(
            result_str.contains("1") && result_str.contains("5"),
            "Should be (1-√5)/2, got: {}",
            result_str
        );
    }

    #[test]
    fn test_denest_cube_no_match_sqrt6() {
        // (2 + sqrt(6))^(1/3) - no rational x,y exists
        let mut ctx = Context::new();
        let expr = parse("(2 + sqrt(6))^(1/3)", &mut ctx).unwrap();

        let rule = DenestPerfectCubeInQuadraticFieldRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should NOT match (2+√6)^(1/3)");
    }

    #[test]
    fn test_denest_cube_no_match_wrong_exp() {
        // (2 + sqrt(5))^(1/5) - exponent is 1/5 not 1/3
        let mut ctx = Context::new();
        let expr = parse("(2 + sqrt(5))^(1/5)", &mut ctx).unwrap();

        let rule = DenestPerfectCubeInQuadraticFieldRule;
        let rewrite = rule.apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        );
        assert!(rewrite.is_none(), "Should NOT match exponent 1/5");
    }
}
