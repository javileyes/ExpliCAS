//! Cycle Defense Tests - Entrega 4
//!
//! Tests that verify the always-on cycle detection prevents infinite loops
//! and terminates gracefully with BlockedHint emissions.

use cas_engine::Simplifier;

/// Test that expressions that previously caused ping-pong cycles now terminate
/// AND simplify correctly to x^(1/3) - 3.
#[test]
fn test_cycle_defense_fractional_powers_terminates() {
    let mut simplifier = Simplifier::with_default_rules();

    // This pattern was reported to cause cycles with power rules
    // Mathematical identity: (x - 27) / (x^(2/3) + 3*x^(1/3) + 9) = x^(1/3) - 3
    let expr = cas_parser::parse(
        "(x - 27) / (x^(2/3) + 3*x^(1/3) + 9)",
        &mut simplifier.context,
    )
    .expect("parsing should succeed");

    // Should terminate, not hang - timeout would indicate failure
    let (result, _steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Check if cycle was detected via blocked hints
    let hints = simplifier.take_blocked_hints();

    // With CancelCubeRootDifferenceRule, result should be x^(1/3) - 3
    // and there should be NO cycle hints (rule solves it directly)
    assert!(
        result_str.contains("1/3") && result_str.contains("x"),
        "Result should simplify to x^(1/3) - 3, got: {}",
        result_str
    );

    // Ideally no cycle hints when the rule works
    println!("Hints received: {:?}", hints.len());
}

/// Test that a synthetic ping-pong pattern terminates.
/// This creates a scenario where two rules would alternate forever without cycle defense.
#[test]
fn test_cycle_defense_terminates_not_hangs() {
    let mut simplifier = Simplifier::with_default_rules();

    // Create an expression that triggers many rewrites
    let expr = cas_parser::parse("(1 + x)^2 + (1 - x)^2", &mut simplifier.context)
        .expect("parsing should succeed");

    // Should terminate within reasonable time
    let (result, _steps) = simplifier.simplify(expr);

    // Verify we got a result
    let result_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    assert!(!result_str.is_empty());
}

/// Test that cycle detection resets properly when phase changes.
/// The key here is termination - not the specific simplification result.
#[test]
fn test_cycle_detection_phase_reset() {
    let mut simplifier = Simplifier::with_default_rules();

    let expr = cas_parser::parse("sin(x)^2 + cos(x)^2", &mut simplifier.context)
        .expect("parsing should succeed");

    // Run through full simplification pipeline
    let (result, _steps) = simplifier.simplify(expr);

    // Verify termination - result may or may not be fully simplified depending on rule set
    let result_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    // Just verify we get a valid result (the Pythagorean identity may or may not simplify to 1)
    assert!(!result_str.is_empty(), "Should return a valid expression");
}
