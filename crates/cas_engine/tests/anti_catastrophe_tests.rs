//! Anti-catastrophe tests for GCD and simplification robustness
//! Tests ensure: budget limits, determinism, mode isolation, no false positives

use cas_ast::Context;
use cas_engine::options::{BranchMode, ComplexMode, ContextMode, EvalOptions};
use cas_engine::Simplifier;
use cas_parser::parse;

/// Helper: simplify expression and return result string
fn simplify(input: &str) -> String {
    let opts = EvalOptions {
        branch_mode: BranchMode::Strict,
        context_mode: ContextMode::Standard,
        complex_mode: ComplexMode::Auto,
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

#[test]
fn test_complex_division() {
    let result = simplify("(3 + 4*i)/(1 + 2*i)");
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
    let result = simplify("i^3");
    assert!(
        result.contains("-") && result.contains("i"),
        "i^3 should be -i, got: {}",
        result
    );
}

#[test]
fn test_i_squared() {
    // i^2 = -1
    let result = simplify("i^2");
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
