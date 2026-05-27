use crate::rule::Rule;
use crate::rules::trigonometry::identities::{
    CotHalfAngleDifferenceRule, HyperbolicCschFourthVerificationRule, TrigHiddenCubicIdentityRule,
};
use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

#[test]
fn test_cot_half_angle_basic() {
    let mut ctx = Context::new();
    let rule = CotHalfAngleDifferenceRule;

    // cot(x/2) - cot(x) → 1/sin(x)
    let expr = parse("cot(x/2) - cot(x)", &mut ctx).unwrap();
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );
    assert!(rewrite.is_some(), "Should match cot(x/2) - cot(x)");

    let result = rewrite.unwrap();
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: result.new_expr
        }
    );
    assert!(
        result_str.contains("sin"),
        "Result should contain sin, got: {}",
        result_str
    );
}

#[test]
fn test_cot_half_angle_no_match_different_args() {
    let mut ctx = Context::new();
    let rule = CotHalfAngleDifferenceRule;

    // cot(x/2) - cot(y) → no change (different args)
    let expr = parse("cot(x/2) - cot(y)", &mut ctx).unwrap();
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );
    assert!(rewrite.is_none(), "Should not match cot(x/2) - cot(y)");
}

#[test]
fn test_cot_half_angle_no_match_third() {
    let mut ctx = Context::new();
    let rule = CotHalfAngleDifferenceRule;

    // cot(x/3) - cot(x) → no change (not half-angle)
    let expr = parse("cot(x/3) - cot(x)", &mut ctx).unwrap();
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );
    assert!(rewrite.is_none(), "Should not match cot(x/3) - cot(x)");
}

#[test]
fn test_hyperbolic_csch_fourth_verification_rule_polynomial_cofactor() {
    let mut ctx = Context::new();
    let rule = HyperbolicCschFourthVerificationRule;
    let expr = parse(
        "2*k*x/(cosh(x^2+b)^2*tanh(x^2+b)^4) - 2*k*x/sinh(x^2+b)^4 - 2*k*x/(cosh(x^2+b)^2*tanh(x^2+b)^2)",
        &mut ctx,
    )
    .unwrap();

    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );
    assert!(
        rewrite.is_some(),
        "Should match csch^4 verification residual with polynomial cofactor"
    );

    let result = rewrite.unwrap();
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: result.new_expr
        }
    );
    assert_eq!(result_str, "0");
}

#[test]
fn test_hyperbolic_csch_fourth_verification_simplifier_polynomial_cofactor() {
    let mut ctx = Context::new();
    let expr = parse(
        "2*k*x/(cosh(x^2+b)^2*tanh(x^2+b)^4) - 2*k*x/sinh(x^2+b)^4 - 2*k*x/(cosh(x^2+b)^2*tanh(x^2+b)^2)",
        &mut ctx,
    )
    .unwrap();

    let mut simplifier = crate::Simplifier::with_default_rules();
    simplifier.context = ctx;
    let (result, _) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    assert_eq!(result_str, "0");
}

// =========================================================================
// TrigHiddenCubicIdentityRule tests
// =========================================================================

#[test]
fn test_hidden_cubic_basic() {
    let mut ctx = Context::new();
    let expr = parse("sin(x)^6 + cos(x)^6 + 3*sin(x)^2*cos(x)^2", &mut ctx).unwrap();

    let mut simplifier = crate::Simplifier::with_default_rules();
    simplifier.context = ctx;
    let (result, _) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    assert_eq!(result_str, "1");
}

#[test]
fn test_hidden_cubic_permutation_cos_first() {
    let mut ctx = Context::new();
    // Different order: cos^6 first
    let expr = parse("cos(x)^6 + 3*cos(x)^2*sin(x)^2 + sin(x)^6", &mut ctx).unwrap();

    let mut simplifier = crate::Simplifier::with_default_rules();
    simplifier.context = ctx;
    let (result, _) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    assert_eq!(result_str, "1");
}

#[test]
fn test_hidden_cubic_coeff_product_first() {
    let mut ctx = Context::new();
    // Coefficient product first
    let expr = parse("3*sin(x)^2*cos(x)^2 + sin(x)^6 + cos(x)^6", &mut ctx).unwrap();

    let mut simplifier = crate::Simplifier::with_default_rules();
    simplifier.context = ctx;
    let (result, _) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    assert_eq!(result_str, "1");
}

#[test]
fn test_hidden_cubic_equivalent_coeff() {
    let mut ctx = Context::new();
    // Coefficient 6/2 = 3
    let expr = parse("sin(x)^6 + cos(x)^6 + (6/2)*sin(x)^2*cos(x)^2", &mut ctx).unwrap();

    let mut simplifier = crate::Simplifier::with_default_rules();
    simplifier.context = ctx;
    let (result, _) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    assert_eq!(result_str, "1");
}

#[test]
fn test_hidden_cubic_no_match_wrong_coeff() {
    let mut ctx = Context::new();
    // Wrong coefficient: 2 instead of 3
    let expr = parse("sin(x)^6 + cos(x)^6 + 2*sin(x)^2*cos(x)^2", &mut ctx).unwrap();

    let rule = TrigHiddenCubicIdentityRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );
    assert!(rewrite.is_none(), "Should not match with coeff=2");
}

#[test]
fn test_hidden_cubic_no_match_different_args() {
    let mut ctx = Context::new();
    // Different arguments: x vs y
    let expr = parse("sin(x)^6 + cos(y)^6 + 3*sin(x)^2*cos(y)^2", &mut ctx).unwrap();

    let rule = TrigHiddenCubicIdentityRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );
    assert!(rewrite.is_none(), "Should not match with different args");
}

#[test]
fn test_hidden_cubic_no_match_extra_terms() {
    let mut ctx = Context::new();
    // Extra term: should not match partially
    let expr = parse("sin(x)^6 + cos(x)^6 + 3*sin(x)^2*cos(x)^2 + 1", &mut ctx).unwrap();

    let rule = TrigHiddenCubicIdentityRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );
    // flatten_add will produce 4 terms, so rule should not match (requires exactly 3)
    assert!(rewrite.is_none(), "Should not match with extra terms");
}
