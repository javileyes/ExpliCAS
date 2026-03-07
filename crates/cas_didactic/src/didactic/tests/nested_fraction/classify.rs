use super::super::super::nested_fraction_analysis::{
    classify_nested_fraction, NestedFractionPattern,
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
