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
        cas_formatter::DisplayExpr {
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
    use cas_engine::DomainMode;

    let opts = EvalOptions {
        steps_mode: StepsMode::On,
        shared: cas_engine::phase::SharedSemanticConfig {
            semantics: cas_engine::semantics::EvalConfig {
                domain_mode: DomainMode::Assume,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut ctx = Context::new();
    let expr = parse("8 * cos(a) * cos(2*a) * cos(4*a)", &mut ctx).expect("Failed to parse");

    let mut simplifier = Simplifier::with_profile(&opts);
    simplifier.context = ctx;
    let (result, _steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
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

// =============================================================================
// prove_nonzero tests for sin(k·π)
// =============================================================================

/// sin(π/9) should be provably non-zero (1/9 is not an integer)
#[test]
fn test_prove_nonzero_sin_pi_over_9() {
    use cas_engine::helpers::prove_nonzero;
    use cas_engine::Proof;

    let mut ctx = Context::new();
    let pi = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::Pi));
    let nine = ctx.num(9);
    let pi_over_9 = ctx.add(cas_ast::Expr::Div(pi, nine));
    let sin_pi_9 = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![pi_over_9]);

    assert_eq!(
        prove_nonzero(&ctx, sin_pi_9),
        Proof::Proven,
        "sin(π/9) should be Proven non-zero"
    );
}

/// sin(π) should be provably zero (1 is an integer)
#[test]
fn test_prove_nonzero_sin_pi() {
    use cas_engine::helpers::prove_nonzero;
    use cas_engine::Proof;

    let mut ctx = Context::new();
    let pi = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::Pi));
    let sin_pi = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![pi]);

    assert_eq!(
        prove_nonzero(&ctx, sin_pi),
        Proof::Disproven,
        "sin(π) should be Disproven (zero)"
    );
}

/// sin(18π/9) = sin(2π) should be provably zero (18/9 = 2 is an integer)
#[test]
fn test_prove_nonzero_sin_18pi_over_9() {
    use cas_engine::helpers::prove_nonzero;
    use cas_engine::Proof;

    let mut ctx = Context::new();
    let pi = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::Pi));
    let coeff = ctx.add(cas_ast::Expr::Number(num_rational::BigRational::new(
        18.into(),
        9.into(),
    )));
    let arg = ctx.add(cas_ast::Expr::Mul(coeff, pi));
    let sin_expr = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![arg]);

    assert_eq!(
        prove_nonzero(&ctx, sin_expr),
        Proof::Disproven,
        "sin(18π/9) = sin(2π) should be Disproven (zero)"
    );
}

/// sin(a) with symbolic a should be Unknown
#[test]
fn test_prove_nonzero_sin_symbolic() {
    use cas_engine::helpers::prove_nonzero;
    use cas_engine::Proof;

    let mut ctx = Context::new();
    let a = ctx.var("a");
    let sin_a = ctx.call_builtin(cas_ast::BuiltinFn::Sin, vec![a]);

    assert_eq!(
        prove_nonzero(&ctx, sin_a),
        Proof::Unknown,
        "sin(a) with symbolic a should be Unknown"
    );
}

// =============================================================================
// SinCosIntegerPiRule tests: pre-order evaluation without expansion
// =============================================================================

fn simplify_with_steps(input: &str) -> (String, Vec<String>) {
    let opts = EvalOptions {
        steps_mode: StepsMode::On,
        ..Default::default()
    };
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("Failed to parse");

    let mut simplifier = Simplifier::with_profile(&opts);
    simplifier.context = ctx;
    let (result, steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    let step_names: Vec<String> = steps.iter().map(|s| s.rule_name.clone()).collect();

    (result_str, step_names)
}

/// sin(3·π) should evaluate to 0 without using Triple Angle Identity
#[test]
fn test_sin_3pi_no_triple_angle() {
    let (result, step_names) = simplify_with_steps("sin(3*pi)");

    assert_eq!(result, "0", "sin(3π) should be 0");
    assert!(
        !step_names.iter().any(|n| n.contains("Triple Angle")),
        "Should NOT use Triple Angle Identity, got steps: {:?}",
        step_names
    );
    assert!(
        step_names.iter().any(|n| n.contains("Integer Multiple")),
        "Should use 'Integer Multiple of π' rule, got steps: {:?}",
        step_names
    );
}

/// sin(81·π/27) = sin(3·π) should also evaluate to 0 without Triple Angle
#[test]
fn test_sin_81pi_over_27_no_triple_angle() {
    let (result, step_names) = simplify_with_steps("sin(81*pi/27)");

    assert_eq!(result, "0", "sin(81π/27) = sin(3π) should be 0");
    assert!(
        !step_names.iter().any(|n| n.contains("Triple Angle")),
        "Should NOT use Triple Angle Identity, got steps: {:?}",
        step_names
    );
}

/// cos(3·π) should evaluate to -1 (3 is odd)
#[test]
fn test_cos_3pi() {
    let result = simplify("cos(3*pi)");
    assert_eq!(result, "-1", "cos(3π) should be -1");
}

/// cos(4·π) should evaluate to 1 (4 is even)
#[test]
fn test_cos_4pi() {
    let result = simplify("cos(4*pi)");
    assert_eq!(result, "1", "cos(4π) should be 1");
}

// =============================================================================
// Context-aware AddFractions gating tests
// =============================================================================

/// sin(a + pi/9) should NOT combine to sin((9a+pi)/9)
/// The structure should be preserved to allow trig identity matching
#[test]
fn test_sin_symbolic_plus_pi_not_combined() {
    let result = simplify("sin(a + pi/9)");
    // Should NOT contain combined form like (9a + pi)/9 or (9·a + pi)/9
    assert!(
        !result.contains("(9") || !result.contains("+") || !result.contains(")/9"),
        "sin(a + pi/9) should NOT combine to sin((9a+pi)/9), got: {}",
        result
    );
}

/// sin(pi/9 + pi/6) SHOULD combine to sin(5*pi/18)
/// Pure constant fractions inside trig should still combine
#[test]
fn test_sin_pi_fractions_do_combine() {
    let result = simplify("sin(pi/9 + pi/6)");
    // 1/9 + 1/6 = 2/18 + 3/18 = 5/18
    // Should contain 5/18 or 5·pi/18 or similar
    assert!(
        result.contains("5/18") || result.contains("5·pi/18") || result.contains("5*pi/18"),
        "sin(pi/9 + pi/6) should simplify to sin(5π/18), got: {}",
        result
    );
}
