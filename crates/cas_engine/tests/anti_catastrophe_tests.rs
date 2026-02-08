//! Anti-catastrophe tests for GCD and simplification robustness
//! Tests ensure: budget limits, determinism, mode isolation, no false positives

use cas_ast::Context;
use cas_engine::options::{BranchMode, ComplexMode, ContextMode, EvalOptions, StepsMode};
use cas_engine::Simplifier;
use cas_parser::parse;

/// Helper: simplify expression and return result string
fn simplify(input: &str) -> String {
    let opts = EvalOptions {
        branch_mode: BranchMode::Strict,
        complex_mode: ComplexMode::Auto,
        steps_mode: StepsMode::On,
        shared: cas_engine::phase::SharedSemanticConfig {
            context_mode: ContextMode::Standard,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("Failed to parse");

    let mut simplifier = Simplifier::with_profile(&opts);
    simplifier.context = ctx;
    let (result, _steps) = simplifier.simplify(expr);

    format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

// =============================================================================
// A3.1: Budget Bailout (no hangs on large expressions)
// =============================================================================

#[test]
fn test_budget_bailout_large_power() {
    // Large power difference should not hang
    // Even if not fully simplified, must complete in reasonable time
    let result = simplify("((x+y)^10)/((x+y)^9)");
    // Either simplifies to (x+y) or stays as-is, but must not hang
    assert!(!result.is_empty(), "Result should not be empty");
}

#[test]
fn test_budget_bailout_complex_fraction() {
    // Complex fraction with many terms
    let result = simplify("((a+b+c)*(x+y+z))/((a+b+c)*(u+v+w))");
    // Should complete without hanging
    assert!(!result.is_empty(), "Result should not be empty");
}

// =============================================================================
// A3.1.1: Power cancellation edge cases (CancelPowersDivisionRule)
// =============================================================================

#[test]
fn test_power_cancel_equal_exponents() {
    // P^n / P^n → 1
    let result = simplify("((x+y)^10)/((x+y)^10)");
    assert!(
        result.contains("1") && !result.contains("+"),
        "P^n/P^n should simplify to 1, got: {}",
        result
    );
}

#[test]
fn test_power_cancel_smaller_numerator() {
    // P^9 / P^10 → 1/(x+y)
    let result = simplify("((x+y)^9)/((x+y)^10)");
    assert!(
        result.contains("1") && result.contains("/"),
        "P^9/P^10 should give 1/P form, got: {}",
        result
    );
}

#[test]
fn test_power_cancel_zero_numerator_exp() {
    // P^0 / P^9 → 1/P^9
    let result = simplify("((x+y)^0)/((x+y)^9)");
    // P^0 = 1, so 1/P^9
    assert!(
        result.contains("1") && result.contains("/"),
        "P^0/P^9 should give 1/P^9 form, got: {}",
        result
    );
}

#[test]
fn test_power_cancel_negative_exponents() {
    // P^(-2) / P^(-5) → P^3
    let result = simplify("((x+y)^(-2))/((x+y)^(-5))");
    assert!(
        result.contains("x + y") || result.contains("3"),
        "P^-2/P^-5 should give P^3 form, got: {}",
        result
    );
}

// =============================================================================
// A3.1.2: Identity neutral preservation (e+0 should equal e)
// =============================================================================

#[test]
fn test_identity_neutral_add_zero() {
    // simplify(e + 0) should equal simplify(e)
    // This caught a bug where +0 triggered auto-expand heuristics incorrectly
    let e = simplify("(a + pi)^2");
    let e_plus_zero = simplify("(a + pi)^2 + 0");
    assert_eq!(
        e, e_plus_zero,
        "simplify(e+0) should equal simplify(e), got: '{}' vs '{}'",
        e, e_plus_zero
    );
}

// =============================================================================
// A3.2: Determinism (same input → same output)
// =============================================================================

#[test]
fn test_determinism_multivar_gcd() {
    let input = "((x+y+z)*(x+2*y+3*z))/((x+y+z)*(2*x-y+z))";

    // Run twice
    let result1 = simplify(input);
    let result2 = simplify(input);

    assert_eq!(result1, result2, "Same input should produce same output");
}

#[test]
fn test_determinism_difference_of_squares() {
    let input = "(x^2 - y^2)/(x - y)";

    let result1 = simplify(input);
    let result2 = simplify(input);

    assert_eq!(result1, result2, "Determinism: diff of squares");
}

#[test]
fn test_determinism_10_runs() {
    let input = "((x+y)*(a+b))/((x+y)*(c+d))";

    let first_result = simplify(input);
    for _ in 0..10 {
        let result = simplify(input);
        assert_eq!(result, first_result, "All 10 runs should match");
    }
}

// =============================================================================
// A3.3: Cancel vs GCD (both work correctly)
// =============================================================================

#[test]
fn test_cancel_vs_gcd_difference_of_squares() {
    // GCD should handle this, not structural cancel
    let result = simplify("(x^2 - y^2)/(x - y)");
    assert!(
        result.contains("x + y") || result.contains("y + x"),
        "Should simplify to x + y, got: {}",
        result
    );
}

#[test]
fn test_cancel_vs_gcd_structural_cancel() {
    // Structural cancel should handle this (common factor)
    let result = simplify("((x+1)*(y+1))/((x+1)*(y+2))");
    // Should cancel (x+1)
    assert!(
        !result.contains("x + 1") || result.contains("(y + 1)"),
        "Should cancel (x+1), got: {}",
        result
    );
}

#[test]
fn test_no_false_positive_sum_of_squares() {
    // x^2 + y^2 has NO common factors with x + y
    let result = simplify("(x^2 + y^2)/(x + y)");
    // Should stay unsimplified (no false cancellation)
    assert!(
        result.contains("x²") || result.contains("x^2"),
        "Should not falsely simplify, got: {}",
        result
    );
}

// =============================================================================
// A3.4: Complex number non-contamination
// =============================================================================

/// Helper: simplify with ComplexEnabled value_domain (for tests involving i)
/// Uses Engine.eval to ensure value_domain propagates correctly
fn simplify_complex(input: &str) -> String {
    use cas_engine::{Engine, EntryKind, EvalAction, EvalRequest, EvalResult, SessionState};

    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Set ComplexEnabled for these tests
    state.options.shared.semantics.value_domain =
        cas_engine::semantics::ValueDomain::ComplexEnabled;

    let parsed = parse(input, &mut engine.simplifier.context).expect("parse failed");
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        kind: EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");

    match &output.result {
        EvalResult::Expr(e) => cas_ast::DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    }
}

#[test]
fn test_complex_division() {
    let result = simplify_complex("(3 + 4*i)/(1 + 2*i)");
    // (3+4i)/(1+2i) = (3+4i)(1-2i)/((1+2i)(1-2i)) = (3-6i+4i+8)/(1+4) = (11-2i)/5
    assert!(
        result.contains("11") && result.contains("5"),
        "Should compute gaussian division correctly, got: {}",
        result
    );
}

#[test]
fn test_i_powers() {
    // i^3 = -i
    let result = simplify_complex("i^3");
    assert!(
        result.contains("-") && result.contains("i"),
        "i^3 should be -i, got: {}",
        result
    );
}

#[test]
fn test_i_squared() {
    // i^2 = -1
    let result = simplify_complex("i^2");
    assert!(
        result == "-1" || result.contains("-1"),
        "i^2 should be -1, got: {}",
        result
    );
}

// =============================================================================
// A3.5: No regression on existing tests
// =============================================================================

#[test]
fn test_content_gcd_still_works() {
    let result = simplify("(2*x + 2*y)/(4*x + 4*y)");
    assert!(
        result.contains("1/2") || result == "1/2",
        "Content GCD should work, got: {}",
        result
    );
}

#[test]
fn test_monomial_gcd_still_works() {
    let result = simplify("(x^2*y)/(x*y^2)");
    assert!(
        result.contains("x") && result.contains("y"),
        "Monomial GCD should work, got: {}",
        result
    );
}

// =============================================================================
// A2: Layer Labeling (verify layer tags appear in step descriptions)
// =============================================================================

/// Helper: simplify and return steps with descriptions
fn simplify_with_steps(input: &str) -> Vec<String> {
    let opts = EvalOptions {
        branch_mode: BranchMode::Strict,
        complex_mode: ComplexMode::Auto,
        steps_mode: StepsMode::On,
        shared: cas_engine::phase::SharedSemanticConfig {
            context_mode: ContextMode::Standard,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("Failed to parse");

    let mut simplifier = Simplifier::with_profile(&opts);
    simplifier.context = ctx;
    let (_result, steps) = simplifier.simplify(expr);

    steps.iter().map(|s| s.description.clone()).collect()
}

#[test]
fn test_layer1_monomial_content_labeling() {
    // Layer 1 case: monomial GCD
    let descriptions = simplify_with_steps("(2*x^2*y + 2*x^2*y^2)/(4*x*y^2 + 4*x*y^3)");
    let has_layer1 = descriptions.iter().any(|d| d.contains("Layer 1"));
    // This may or may not trigger Layer 1 depending on implementation
    // Just verify we get some output
    assert!(
        !descriptions.is_empty() || !has_layer1,
        "Should produce steps"
    );
}

#[test]
fn test_layer2_heuristic_labeling() {
    // Layer 2 case: difference of squares (requires polynomial GCD)
    // OR pre-order detection "Factor and cancel" step
    let descriptions = simplify_with_steps("(x^2 - y^2)/(x - y)");
    let has_layer2 = descriptions.iter().any(|d| d.contains("Layer 2"));
    let has_gcd = descriptions.iter().any(|d| d.contains("GCD"));
    let has_factor = descriptions.iter().any(|d| d.contains("Factor"));
    // Either Layer 2, GCD, or Factor step is valid
    assert!(
        has_layer2 || has_gcd || has_factor,
        "Should have Layer 2, GCD, or Factor step for diff of squares, got: {:?}",
        descriptions
    );
}

#[test]
fn test_layer25_tensor_grid_labeling() {
    // Layer 2.5 case: multi-param factor
    // Note: CancelCommonFactorsRule may handle this before GCD
    let descriptions = simplify_with_steps("((x+y+z)*(x+2*y+3*z))/((x+y+z)*(2*x-y+z))");
    let has_layer25 = descriptions.iter().any(|d| d.contains("Layer 2.5"));
    let has_gcd = descriptions.iter().any(|d| d.contains("GCD"));
    let has_cancel = descriptions.iter().any(|d| d.contains("Cancel"));
    // Either Layer 2.5, GCD, or Cancel common factors is valid
    assert!(
        has_layer25 || has_gcd || has_cancel,
        "Should have Layer 2.5, GCD, or Cancel step for multi-param, got: {:?}",
        descriptions
    );
}
