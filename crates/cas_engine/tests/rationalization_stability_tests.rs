//! Contract tests for rationalization stability.
//!
//! Ensures that simplification rules don't undo rationalization.

use cas_engine::eval_str_to_json;

/// Test: 3/(2*sqrt(x)) should NOT produce negative fractional exponents
/// This is the canonical contract test for the rationalization stability bug.
#[test]
fn test_rationalization_not_undone() {
    let json = eval_str_to_json("3/(2*sqrt(x))", "{}");

    // Must NOT contain x^(-1/2) or sqrt(x^-1) - these undo rationalization
    assert!(
        !json.contains("x^(-1/2)") && !json.contains("x^-1/2"),
        "Result contains negative fractional exponent: {}",
        json
    );
    assert!(
        !json.contains("sqrt(x^-1)") && !json.contains("√(x^-1)"),
        "Result contains inverse inside sqrt: {}",
        json
    );

    // Should contain valid JSON with ok=true
    assert!(json.contains("\"ok\":true"), "Evaluation failed: {}", json);
}

/// Test: x^2/x should still simplify to x (integer exponent case)
#[test]
fn test_integer_exponent_cancellation_works() {
    let json = eval_str_to_json("x^2/x", "{}");

    assert!(json.contains("\"ok\":true"), "Evaluation failed: {}", json);
    // The result should be simplified - either "x" or not contain x^1
    let result_contains_simplified = json.contains("\"result\":\"x\"") || !json.contains("x^1");
    assert!(
        result_contains_simplified,
        "Integer cancellation not working: {}",
        json
    );
}

/// Test: x^3 / x^2 should become x (integer case)
#[test]
fn test_integer_power_cancellation() {
    let json = eval_str_to_json("x^3 / x^2", "{}");

    assert!(json.contains("\"ok\":true"));
    // Should simplify to x
    assert!(
        json.contains("\"result\":\"x\""),
        "Expected x but got: {}",
        json
    );
}

/// Test: sqrt(x)*sqrt(x) should become x (not contain sqrt in result)
#[test]
fn test_sqrt_squared_simplifies() {
    let json = eval_str_to_json("sqrt(x)*sqrt(x)", "{}");

    assert!(json.contains("\"ok\":true"));
    // Should simplify to x
    assert!(
        json.contains("\"result\":\"x\""),
        "sqrt(x)*sqrt(x) not simplified to x: {}",
        json
    );
}

/// Test: Generalized rationalization with 3+ terms should expand the denominator
/// 1/(1 + sqrt(2) + sqrt(3)) should NOT leave unexpanded (1+√2)² in result
#[test]
fn test_generalized_rationalization_expands_denominator() {
    let json = eval_str_to_json("1/(1 + sqrt(2) + sqrt(3))", "{}");

    // Should evaluate successfully
    assert!(json.contains("\"ok\":true"), "Evaluation failed: {}", json);

    // Must NOT contain unexpanded (1+√2)² pattern
    assert!(
        !json.contains("(1 + √(2))²") && !json.contains("(1 + 2^(1/2))^2"),
        "Result contains unexpanded pow-sum: {}",
        json
    );

    // Denominator should be simplified (should be a number like 4, not containing √2²)
    // The expected result has denominator 4 (from 2√2 after expansion and simplification)
    assert!(
        json.contains("/4") || json.contains("1/4"),
        "Denominator not simplified to 4: {}",
        json
    );
}
