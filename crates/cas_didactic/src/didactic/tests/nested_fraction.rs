use super::super::nested_fraction_analysis::{
    classify_nested_fraction, extract_combined_fraction_str, NestedFractionPattern,
};
use cas_ast::{Context, Expr};
use num_bigint::BigInt;
use num_rational::BigRational;

#[test]
fn test_nested_fraction_pattern_classification_p1() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));
    let one_over_x = ctx.add(Expr::Div(one, x));
    let denom = ctx.add(Expr::Add(one, one_over_x));
    let expr = ctx.add(Expr::Div(one, denom));

    let pattern = classify_nested_fraction(&ctx, expr);
    assert!(matches!(
        pattern,
        Some(NestedFractionPattern::OneOverSumWithUnitFraction)
    ));
}

#[test]
fn test_nested_fraction_pattern_classification_p3() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));
    let two = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(2))));
    let one_over_x = ctx.add(Expr::Div(one, x));
    let denom = ctx.add(Expr::Add(one, one_over_x));
    let expr = ctx.add(Expr::Div(two, denom));

    let pattern = classify_nested_fraction(&ctx, expr);
    assert!(matches!(
        pattern,
        Some(NestedFractionPattern::FractionOverSumWithFraction)
    ));
}

#[test]
fn test_extract_combined_fraction_simple() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));
    let one_over_x = ctx.add(Expr::Div(one, x));
    let add_expr = ctx.add(Expr::Add(one, one_over_x));

    let result = extract_combined_fraction_str(&ctx, add_expr);
    assert!(
        result.contains("x"),
        "Should contain denominator 'x': {}",
        result
    );
    assert!(
        result.contains("1"),
        "Should contain numerator '1': {}",
        result
    );
}

#[test]
fn test_extract_combined_fraction_complex_denominator() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));
    let x_plus_1 = ctx.add(Expr::Add(x, one));
    let x_over_xplus1 = ctx.add(Expr::Div(x, x_plus_1));
    let add_expr = ctx.add(Expr::Add(one, x_over_xplus1));

    let result = extract_combined_fraction_str(&ctx, add_expr);
    assert!(
        result.contains("\\frac"),
        "Should contain LaTeX \\frac: {}",
        result
    );
    assert!(
        result.contains("\\cdot"),
        "Should contain LaTeX \\cdot for multiplication: {}",
        result
    );
}
