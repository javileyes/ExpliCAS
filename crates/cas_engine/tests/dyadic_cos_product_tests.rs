//! Dyadic Cosine Product Rule Tests
//!
//! Tests for the identity: 2^n · ∏_{k=0}^{n-1} cos(2^k·θ) = sin(2^n·θ)/sin(θ)
//!
//! This covers the Olympiad problem: 8·cos(π/9)·cos(2π/9)·cos(4π/9) = 1

use cas_ast::Context;
use cas_engine::options::{EvalOptions, StepsMode};
use cas_engine::Simplifier;
use cas_parser::parse;

fn simplify(input: &str) -> String {
    let opts = EvalOptions {
        steps_mode: StepsMode::On,
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

/// The main test case: 8·cos(π/9)·cos(2π/9)·cos(4π/9) = 1
#[test]
fn test_dyadic_cos_product_pi_9() {
    let result = simplify("8 * cos(pi/9) * cos(2*pi/9) * cos(4*pi/9)");
    assert_eq!(
        result, "1",
        "8·cos(π/9)·cos(2π/9)·cos(4π/9) should simplify to 1"
    );
}

/// Test the supplementary angle rule: sin(8π/9) = sin(π/9)
#[test]
fn test_sin_supplementary_angle() {
    let result = simplify("sin(8*pi/9)");
    // 8/9 = 1 - 1/9, so sin(8π/9) = sin(π - π/9) = sin(π/9)
    assert!(
        result.contains("1/9") && result.contains("sin"),
        "sin(8π/9) should simplify to sin(π/9), got: {}",
        result
    );
}

/// Verify that the domain check works - sin(θ)≠0 is proven for π/9
#[test]
fn test_domain_check_proven() {
    // π/9 is not an integer multiple of π, so sin(π/9) ≠ 0 is provable
    // The rule should apply without blocking
    let result = simplify("8 * cos(pi/9) * cos(2*pi/9) * cos(4*pi/9)");
    assert_eq!(result, "1", "Should simplify when sin(θ)≠0 is provable");
}

/// Test with permuted factor order - should still simplify to 1
#[test]
fn test_dyadic_cos_product_permuted_order() {
    // Different orderings should all work due to multiset matching
    let result1 = simplify("8 * cos(2*pi/9) * cos(pi/9) * cos(4*pi/9)");
    let result2 = simplify("8 * cos(4*pi/9) * cos(2*pi/9) * cos(pi/9)");
    let result3 = simplify("cos(pi/9) * 8 * cos(4*pi/9) * cos(2*pi/9)");

    assert_eq!(
        result1, "1",
        "Permuted order 1 should simplify to 1, got: {}",
        result1
    );
    assert_eq!(
        result2, "1",
        "Permuted order 2 should simplify to 1, got: {}",
        result2
    );
    assert_eq!(
        result3, "1",
        "Permuted order 3 should simplify to 1, got: {}",
        result3
    );
}

/// In Generic mode with symbolic θ, the rule should be BLOCKED because sin(θ)≠0 cannot be proven
#[test]
fn test_dyadic_cos_product_generic_symbolic_blocked() {
    // Generic mode (default) with symbolic 'a' - should NOT transform
    let result = simplify("8 * cos(a) * cos(2*a) * cos(4*a)");

    // Should remain as product of cosines (not transformed to sin ratio)
    assert!(
        result.contains("cos"),
        "Generic mode with symbolic θ should NOT apply dyadic rule, got: {}",
        result
    );
    assert!(
        !result.contains("sin(8"),
        "Should not produce sin(8a) in Generic mode with symbolic θ"
    );
}

/// In Assume mode with symbolic θ, the rule should apply with warning
#[test]
fn test_dyadic_cos_product_assume_symbolic_allowed() {
    use cas_engine::domain::DomainMode;

    let opts = EvalOptions {
        steps_mode: StepsMode::On,
        domain_mode: DomainMode::Assume,
        ..Default::default()
    };
    let mut ctx = Context::new();
    let expr = parse("8 * cos(a) * cos(2*a) * cos(4*a)", &mut ctx).expect("Failed to parse");

    let mut simplifier = Simplifier::with_profile(&opts);
    simplifier.context = ctx;
    let (result, _steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // In Assume mode, transformation should occur
    assert!(
        result_str.contains("sin"),
        "Assume mode with symbolic θ SHOULD allow dyadic rule, got: {}",
        result_str
    );
}
