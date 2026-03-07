use super::fraction_steps::generate_fraction_sum_substeps;
use super::fraction_sum_analysis::FractionSumInfo;
use super::nested_fraction_analysis::{
    classify_nested_fraction, extract_combined_fraction_str, NestedFractionPattern,
};
use super::shared_numeric::gcd_bigint;
use super::*;
use cas_ast::{Context, Expr};
use cas_math::expr_predicates::contains_division_like_term;
use num_bigint::BigInt;
use num_rational::BigRational;

#[test]
fn test_format_fraction() {
    let half = BigRational::new(BigInt::from(1), BigInt::from(2));
    assert_eq!(format_fraction(&half), "\\frac{1}{2}");

    let three = BigRational::from_integer(BigInt::from(3));
    assert_eq!(format_fraction(&three), "3");
}

#[test]
fn test_gcd_lcm() {
    let a = BigInt::from(12);
    let b = BigInt::from(8);
    assert_eq!(gcd_bigint(&a, &b), BigInt::from(4));
    assert_eq!(lcm_bigint(&a, &b), BigInt::from(24));
}

#[test]
fn test_fraction_sum_substeps() {
    let fractions = vec![
        BigRational::new(BigInt::from(1), BigInt::from(24)),
        BigRational::new(BigInt::from(1), BigInt::from(2)),
        BigRational::new(BigInt::from(1), BigInt::from(6)),
    ];
    let result: BigRational = fractions.iter().cloned().sum();

    let info = FractionSumInfo {
        fractions,
        result: result.clone(),
    };

    let substeps = generate_fraction_sum_substeps(&info);
    assert!(!substeps.is_empty());

    assert_eq!(result, BigRational::new(BigInt::from(17), BigInt::from(24)));
}

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

#[test]
fn test_contains_div_simple() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let one = ctx.add(Expr::Number(BigRational::from_integer(BigInt::from(1))));

    assert!(!contains_division_like_term(&ctx, x));

    let div = ctx.add(Expr::Div(one, x));
    assert!(contains_division_like_term(&ctx, div));

    let add = ctx.add(Expr::Add(one, div));
    assert!(contains_division_like_term(&ctx, add));
}

#[test]
fn test_build_cli_substeps_render_plan_fraction_sum_deduped() {
    let sub_steps = vec![SubStep::new(
        "Find common denominator for fractions",
        "",
        "",
    )];
    let plan = build_cli_substeps_render_plan(&sub_steps);
    assert_eq!(plan.header, Some("[Suma de fracciones en exponentes]"));
    assert!(plan.dedupe_once);
}

#[test]
fn test_latex_to_plain_text_converts_frac_and_text() {
    let input = r"\text{Paso}: \frac{1}{x+1} \cdot y";
    let output = latex_to_plain_text(input);
    assert!(output.contains("Paso"));
    assert!(output.contains("(1/x+1)"));
    assert!(output.contains("·"));
}
