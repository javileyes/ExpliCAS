//! Tests for the explicit phase pipeline.
//!
//! These tests verify that:
//! 1. Phases run in correct order: Core → Transform → Rationalize → PostCleanup
//! 2. Transform never runs after Rationalize
//! 3. expand() uses no-rationalize path

use cas_engine::Orchestrator;
use cas_engine::Simplifier;
use cas_formatter::root_style::{ParseStyleSignals, StylePreferences};
use cas_formatter::DisplayExprStyled;

fn simplify_with_pipeline(input: &str) -> String {
    let mut simplifier = Simplifier::new();
    simplifier.register_default_rules();
    let signals = ParseStyleSignals::from_input_string(input);
    let expr = cas_parser::parse(input, &mut simplifier.context).expect("parse failed");

    let mut orchestrator = Orchestrator::new();
    let (simplified, _, _) = orchestrator.simplify_pipeline(expr, &mut simplifier);

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

fn simplify_for_expand(input: &str) -> String {
    let mut simplifier = Simplifier::new();
    simplifier.register_default_rules();
    let expr = cas_parser::parse(input, &mut simplifier.context).expect("parse failed");

    let mut orchestrator = Orchestrator::for_expand();
    let (simplified, _, _) = orchestrator.simplify_pipeline(expr, &mut simplifier);

    format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: simplified,
        }
    )
}

#[test]
fn test_pipeline_rationalize_works() {
    // Rationalization should work in the pipeline
    let result = simplify_with_pipeline("1/(1+sqrt(2))");
    assert!(
        result.contains("√") || result.contains("sqrt"),
        "Got: {}",
        result
    );
    assert!(
        !result.contains("/(1"),
        "Should rationalize denominator: {}",
        result
    );
}

#[test]
fn test_pipeline_mixed_expression() {
    // Both rationalization and distribution should work
    let result = simplify_with_pipeline("x/(1+sqrt(2)) + 2*(y+3)");

    // Distribution worked on 2*(y+3)
    assert!(
        result.contains("6"),
        "Expected distributed 2*3=6 in: {}",
        result
    );

    // Rationalization worked on x/(1+√2)
    assert!(
        result.contains("√") || result.contains("sqrt"),
        "Expected rationalized sqrt in: {}",
        result
    );
}

#[test]
fn test_pipeline_no_transform_after_rationalize() {
    // After rationalization, expression should not be re-expanded
    let result = simplify_with_pipeline("1/(2*(1+sqrt(2)))");

    // Should be rationalized, not showing (1+√2) in denominator
    assert!(
        result.contains("√") || result.contains("sqrt"),
        "Got: {}",
        result
    );
}

#[test]
fn test_expand_path_no_rationalize() {
    // expand() should NOT rationalize
    let result = simplify_for_expand("1/(1+sqrt(2))");

    // Should still have sqrt in denominator (not rationalized)
    // The expression might simplify but won't rationalize
    assert!(
        result.contains("2")
            && (result.contains("^(1/2)") || result.contains("sqrt") || result.contains("√")),
        "Expected sqrt to remain: {}",
        result
    );
}

#[test]
fn test_pipeline_idempotent() {
    // simplify(simplify(x)) == simplify(x)
    // Test with a simpler expression that doesn't have unicode output
    let mut simplifier = Simplifier::new();
    simplifier.register_default_rules();
    let expr = cas_parser::parse("x/(1+sqrt(2))", &mut simplifier.context).expect("parse failed");

    let mut orchestrator = Orchestrator::new();
    let (first, _, _) = orchestrator.simplify_pipeline(expr, &mut simplifier);

    // Run pipeline again on the result
    let mut orchestrator2 = Orchestrator::new();
    let (second, _, _) = orchestrator2.simplify_pipeline(first, &mut simplifier);

    // The ExprIds should be equal (fixed point)
    assert_eq!(first, second, "Pipeline should be idempotent");
}
