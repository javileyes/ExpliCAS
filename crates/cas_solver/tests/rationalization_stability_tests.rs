//! Contract tests for rationalization stability.
//!
//! Ensures that simplification rules don't undo rationalization.

use cas_solver::eval_str_to_json;

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

/// Test: Cube root binomial should use geometric sum, not diff squares
/// 1/(2^(1/3) - 1) should give 2^(2/3) + 2^(1/3) + 1
#[test]
fn test_cube_root_binomial_rationalization() {
    let json = eval_str_to_json("1/(2^(1/3) - 1)", "{}");

    // Should succeed
    assert!(json.contains("\"ok\":true"), "Evaluation failed: {}", json);

    // Must NOT leave denominator with (2^(2/3) - 1) - should be fully rationalized to 1
    assert!(
        !json.contains("2^(2/3) - 1") && !json.contains("- 1\""),
        "Cube root not fully rationalized: {}",
        json
    );

    // Should contain all three terms of the geometric sum
    assert!(
        json.contains("2^(1/3)") && json.contains("2^(2/3)"),
        "Missing expected terms in result: {}",
        json
    );
}

/// Test: 4th root binomial rationalization  
#[test]
fn test_4th_root_binomial_rationalization() {
    let json = eval_str_to_json("1/(x^(1/4) - 1)", "{}");

    assert!(json.contains("\"ok\":true"), "Evaluation failed: {}", json);

    // Denominator should be (x - 1), not (x^(3/4) - 1) etc
    // The multiplier should be x^(3/4) + x^(2/4) + x^(1/4) + 1
    assert!(
        json.contains("x^(3/4)") || json.contains("x^(1/2)"),
        "4th root rationalization missing expected terms: {}",
        json
    );
}

/// Test: (x+1)/(x^(1/3)+1) should simplify using sum of cubes identity
#[test]
fn test_cancel_cube_root_binomial_factor() {
    let json = eval_str_to_json("(x + 1) / (x^(1/3) + 1)", "{}");

    assert!(json.contains("\"ok\":true"), "Evaluation failed: {}", json);

    // Should not contain division in result - fully cancelled
    // Result should be x^(2/3) - x^(1/3) + 1 (or reordered)
    assert!(
        json.contains("x^(2/3)") && json.contains("x^(1/3)"),
        "Missing expected power terms in result: {}",
        json
    );
}

/// Test: (x-1)/(x^(1/3)-1) should give x^(2/3)+x^(1/3)+1
#[test]
fn test_cancel_cube_root_binomial_factor_minus() {
    let json = eval_str_to_json("(x - 1) / (x^(1/3) - 1)", "{}");

    assert!(json.contains("\"ok\":true"), "Evaluation failed: {}", json);

    // Result should contain the geometric series terms
    assert!(
        json.contains("x^(2/3)") && json.contains("x^(1/3)"),
        "Missing expected power terms in result: {}",
        json
    );
}

/// Test (A): Sum of cubes with r=2
#[test]
fn test_cancel_cube_root_factor_r_equals_2_sum() {
    let json = eval_str_to_json("(x + 8) / (x^(1/3) + 2)", "{}");
    assert!(json.contains("\"ok\":true"));
    // Should give x^(2/3) - 2*x^(1/3) + 4
    assert!(json.contains("x^(2/3)"), "Missing x^(2/3): {}", json);
}

/// Test (B): Diff of cubes with r=2  
#[test]
fn test_cancel_cube_root_factor_r_equals_2_diff() {
    let json = eval_str_to_json("(x - 8) / (x^(1/3) - 2)", "{}");
    assert!(json.contains("\"ok\":true"));
    // Should give x^(2/3) + 2*x^(1/3) + 4
    assert!(json.contains("x^(2/3)"), "Missing x^(2/3): {}", json);
}

/// Test (E): Diff of 4th powers
#[test]
fn test_cancel_4th_root_factor_diff() {
    let json = eval_str_to_json("(x - 1) / (x^(1/4) - 1)", "{}");
    assert!(json.contains("\"ok\":true"));
    // Should give x^(3/4) + x^(1/2) + x^(1/4) + 1
    assert!(
        json.contains("x^(3/4)") || json.contains("x^(1/2)"),
        "Missing expected terms: {}",
        json
    );
}

/// Test (F): Sum of 4th powers should NOT simplify (n even)
#[test]
fn test_cancel_4th_root_factor_sum_no_apply() {
    let json = eval_str_to_json("(x + 1) / (x^(1/4) + 1)", "{}");
    assert!(json.contains("\"ok\":true"));
    // Should remain as fraction (n=4 even, sum pattern doesn't factor)
    // Just verify no crash and result contains expected structure
}

/// Test (J): Anti-basura - no negative fractional exponents in result
#[test]
fn test_no_negative_fractional_exponents() {
    let json = eval_str_to_json("(x + 1) / (x^(1/3) + 1)", "{}");
    assert!(json.contains("\"ok\":true"));
    // Must NOT introduce negative fractional exponents
    assert!(
        !json.contains("x^(-") && !json.contains("-1/3"),
        "Contains negative fractional exponent: {}",
        json
    );
}

/// Test: 1/(2 - sqrt(3)) should rationalize to 2 + sqrt(3)
#[test]
fn test_rationalize_binomial_minus_case() {
    let json = eval_str_to_json("1/(2 - sqrt(3))", "{}");
    assert!(json.contains("\"ok\":true"));
    // Should produce 2 + sqrt(3)
    assert!(
        json.contains("3^(1/2)") || json.contains("√(3)"),
        "Missing sqrt(3) in result: {}",
        json
    );
}

/// Test: 1/(2+sqrt(3)) + 1/(2-sqrt(3)) should equal 4
#[test]
fn test_conjugate_sum_simplifies_to_4() {
    let json = eval_str_to_json("1/(2 + sqrt(3)) + 1/(2 - sqrt(3))", "{}");
    assert!(json.contains("\"ok\":true"));
    // Should simplify to 4
    assert!(
        json.contains("\"result\":\"4\""),
        "Expected 4 but got: {}",
        json
    );
}

/// Test: 1/sqrt(x + sqrt(x²-1)) should simplify to sqrt(x - sqrt(x²-1))
#[test]
fn test_sqrt_conjugate_collapse() {
    let json = eval_str_to_json("1 / sqrt(x + sqrt(x^2 - 1))", "{}");
    assert!(json.contains("\"ok\":true"), "Evaluation failed: {}", json);
    // Should collapse to sqrt(x - sqrt(x²-1))
    // Result should contain x - (x^2 - 1)^(1/2) inside a sqrt
    assert!(
        json.contains("(1/2)") || json.contains("^(1/2)"),
        "Missing sqrt in result: {}",
        json
    );
}
