use super::super::super::nested_fraction_analysis::extract_combined_fraction_str;
use cas_ast::{Context, Expr};
use num_bigint::BigInt;
use num_rational::BigRational;

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
        result.contains("x + 1"),
        "Should preserve the grouped denominator x + 1 in the combined fraction: {}",
        result
    );
    assert!(
        !result.contains("1 \\cdot"),
        "Should not keep a trivial 1· factor in the combined numerator: {}",
        result
    );
}

#[test]
fn test_extract_combined_fraction_two_reciprocals_uses_common_denominator() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");
    let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));
    let one_over_x = ctx.add(Expr::Div(one, x));
    let one_over_y = ctx.add(Expr::Div(one, y));
    let add_expr = ctx.add(Expr::Add(one_over_x, one_over_y));

    let result = extract_combined_fraction_str(&ctx, add_expr);
    assert!(
        result.contains("\\frac"),
        "Should contain LaTeX \\frac: {}",
        result
    );
    assert!(
        result.contains("x") && result.contains("y"),
        "Should contain both reciprocal denominators in the common fraction: {}",
        result
    );
    assert!(
        result.contains("x \\cdot y") || result.contains("y \\cdot x"),
        "Should use a common denominator product: {}",
        result
    );
    assert!(
        !result.contains("\\frac{1}{y} \\cdot x") && !result.contains("\\frac{1}{x} \\cdot y"),
        "Should not keep partially simplified reciprocal products: {}",
        result
    );
}
