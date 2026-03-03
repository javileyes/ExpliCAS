use super::root_denesting::*;
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
