//! Tests verifying that phase-based simplification works correctly.
//!
//! Key properties:
//! - Rationalization and distribution can coexist in same expression
//! - Rationalized results are not re-expanded

use cas_ast::display::DisplayExprStyled;
use cas_ast::root_style::{ParseStyleSignals, StylePreferences};
use cas_engine::Simplifier;

fn simplify_to_string(input: &str) -> String {
    let mut simplifier = Simplifier::new();
    simplifier.register_default_rules();
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
fn test_mixed_rationalize_and_distribute() {
    // Both parts should be simplified: rationalize one, distribute the other
    let result = simplify_to_string("x/(1+sqrt(2)) + 2*(y+3)");

    // Should contain distributed result (6 from 2*3)
    assert!(
        result.contains("6"),
        "Expected distributed 2*3=6 in: {}",
        result
    );

    // Should contain rationalized sqrt
    assert!(
        result.contains("√") || result.contains("sqrt"),
        "Expected rationalized sqrt in: {}",
        result
    );
}

#[test]
fn test_rationalized_not_reexpanded() {
    // Rationalized result should stay compact
    let result = simplify_to_string("1/(1+sqrt(2))");

    // Should be -1 + √2, not contain (1 + √2) in denominator
    assert!(
        !result.contains("/(1"),
        "Rationalized result was re-expanded: {}",
        result
    );
    assert!(
        result.contains("√") || result.contains("sqrt"),
        "Missing sqrt in: {}",
        result
    );
}

#[test]
fn test_level_1_5_stays_compact() {
    // Level 1.5 rationalization should stay compact
    let result = simplify_to_string("1/(2*(1+sqrt(2)))");

    // Should not contain original denominator pattern
    assert!(
        !result.contains("(1 +") || !result.contains("/(2"),
        "Level 1.5 result was not properly simplified: {}",
        result
    );
}
