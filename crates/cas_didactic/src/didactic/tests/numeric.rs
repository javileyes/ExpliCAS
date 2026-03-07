use super::super::shared_numeric::gcd_bigint;
use super::super::*;
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
