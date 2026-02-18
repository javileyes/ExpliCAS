//! Guard tests for prefer_division display.
//!
//! These tests ensure that the 1/k * X → X/k display transformation
//! handles edge cases correctly.

use cas_ast::root_style::{ParseStyleSignals, StylePreferences};
use cas_engine::Simplifier;
use cas_formatter::DisplayExprStyled;

fn simplify_and_format(input: &str) -> String {
    let mut simplifier = Simplifier::new();
    let signals = ParseStyleSignals::from_input_string(input);
    let expr = cas_parser::parse(input, &mut simplifier.context).expect("parse failed");
    let (simplified, _) = simplifier.simplify(expr);
    let style = StylePreferences::from_expression_with_signals(
        &simplifier.context,
        simplified,
        Some(&signals),
    );
    format!(
        "{}",
        DisplayExprStyled::new(&simplifier.context, simplified, &style)
    )
}

#[test]
fn test_prefer_division_rationalize_result() {
    // 1/(1+sqrt(2)) rationalizes and shows cleanly
    let result = simplify_and_format("1/(1+sqrt(2))");
    // Should be something like -1 + √2
    assert!(
        result.contains("√") || result.contains("sqrt"),
        "Got: {}",
        result
    );
    // Should NOT have 1/1 * pattern
    assert!(!result.contains("1/1"), "Unnecessary 1/1 in: {}", result);
}

#[test]
fn test_prefer_division_no_double_negative() {
    // After simplification, should never have "--"
    let result = simplify_and_format("-x/2");
    assert!(!result.contains("--"), "Double negative in: {}", result);
}

#[test]
fn test_prefer_division_level_15() {
    // Level 1.5 rationalization: x/(2*(1+√2)) shows result properly formatted
    let result = simplify_and_format("1/(2*(1+sqrt(2)))");
    // After rationalization, shows as (√2-1)/2 or similar (not 1/2 * ...)
    assert!(
        result.contains("√") || result.contains("sqrt"),
        "Got: {}",
        result
    );
}
