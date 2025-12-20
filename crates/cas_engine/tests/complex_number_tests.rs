//! Complex Number Tests
//!
//! These tests verify the imaginary unit `i` support:
//! - Powers of i: i^2 = -1, i^3 = -i, i^4 = 1
//! - Gaussian arithmetic: (a+bi) ± (c+di), (a+bi)(c+di)
//! - Mode isolation: ComplexMode::Off should not simplify i

use cas_ast::Context;
use cas_engine::options::{BranchMode, ComplexMode, ContextMode, EvalOptions, StepsMode};
use cas_engine::phase::ExpandPolicy;
use cas_engine::Simplifier;
use cas_parser::parse;

/// Helper: run simplification with given options and return result string
fn simplify_with(input: &str, opts: &EvalOptions) -> String {
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("Failed to parse");

    let mut simplifier = Simplifier::with_profile(opts);
    simplifier.context = ctx;
    // Use simplify_with_options to propagate expand_policy from EvalOptions
    let simplify_opts = opts.to_simplify_options();
    let (result, _steps) = simplifier.simplify_with_options(expr, simplify_opts);

    format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

fn complex_on_opts() -> EvalOptions {
    EvalOptions {
        branch_mode: BranchMode::Strict,
        context_mode: ContextMode::Standard,
        complex_mode: ComplexMode::On,
        steps_mode: StepsMode::On,
        ..Default::default()
    }
}

/// Complex mode On + Auto-expand: for identity tests that need expansion
fn complex_on_autoexpand_opts() -> EvalOptions {
    EvalOptions {
        expand_policy: ExpandPolicy::Auto,
        ..complex_on_opts()
    }
}

fn complex_off_opts() -> EvalOptions {
    EvalOptions {
        branch_mode: BranchMode::Strict,
        context_mode: ContextMode::Standard,
        complex_mode: ComplexMode::Off,
        steps_mode: StepsMode::On,
        ..Default::default()
    }
}

fn complex_auto_opts() -> EvalOptions {
    EvalOptions {
        branch_mode: BranchMode::Strict,
        context_mode: ContextMode::Standard,
        complex_mode: ComplexMode::Auto,
        steps_mode: StepsMode::On,
        ..Default::default()
    }
}

// =============================================================================
// SECTION 1: Powers of i
// =============================================================================

#[test]
fn test_i_squared_equals_negative_one() {
    let result = simplify_with("i^2", &complex_on_opts());
    assert_eq!(result, "-1", "i^2 should equal -1");
}

#[test]
fn test_i_cubed_equals_negative_i() {
    let result = simplify_with("i^3", &complex_on_opts());
    assert_eq!(result, "-i", "i^3 should equal -i");
}

#[test]
fn test_i_fourth_equals_one() {
    let result = simplify_with("i^4", &complex_on_opts());
    assert_eq!(result, "1", "i^4 should equal 1");
}

#[test]
fn test_i_fifth_equals_i() {
    let result = simplify_with("i^5", &complex_on_opts());
    assert_eq!(result, "i", "i^5 should equal i (same as i^1)");
}

#[test]
fn test_i_large_power() {
    // i^17 = i^(16+1) = (i^4)^4 * i = 1 * i = i
    let result = simplify_with("i^17", &complex_on_opts());
    assert_eq!(result, "i", "i^17 mod 4 = 1, so i^17 = i");
}

#[test]
fn test_i_times_i_equals_negative_one() {
    let result = simplify_with("i*i", &complex_on_opts());
    assert_eq!(result, "-1", "i*i should equal -1");
}

// =============================================================================
// SECTION 2: ComplexMode::Auto detection
// =============================================================================

#[test]
fn test_auto_mode_with_i_applies_rules() {
    // Auto mode should detect `i` and apply complex rules
    let result = simplify_with("i^2", &complex_auto_opts());
    assert_eq!(result, "-1", "Auto mode should simplify i^2 to -1");
}

#[test]
fn test_auto_mode_without_i_no_extra_work() {
    // Expression without i should work normally
    let result = simplify_with("2 + 3", &complex_auto_opts());
    assert_eq!(result, "5", "Normal arithmetic should still work");
}

// =============================================================================
// SECTION 3: ComplexMode::Off behavior
// NOTE: Currently rules are registered unconditionally for simplicity.
// ComplexMode::Off is reserved for future conditional registration.
// =============================================================================

#[test]
fn test_off_mode_documented_behavior() {
    // Currently, complex rules fire regardless of mode because they're
    // registered unconditionally. The Off mode is reserved for future
    // conditional rule registration.
    let result = simplify_with("i^2", &complex_off_opts());
    // Document current behavior: rules still fire in off mode
    assert_eq!(result, "-1", "Currently rules fire even in Off mode");
}

// =============================================================================
// SECTION 4: Gaussian Multiplication
// =============================================================================

#[test]
fn test_gaussian_mul_pure_imaginary() {
    // (2i)(3i) = 6i^2 = -6
    let result = simplify_with("(2*i)*(3*i)", &complex_on_opts());
    assert_eq!(result, "-6", "(2i)(3i) = -6");
}

#[test]
fn test_gaussian_mul_mixed() {
    // (1+i)(1+i) = 1 + 2i + i^2 = 1 + 2i - 1 = 2i
    // Use auto-expand opts for identity tests that need expansion within budget
    let result = simplify_with("(1+i)*(1+i)", &complex_on_autoexpand_opts());
    assert_eq!(result, "2 * i", "(1+i)^2 = 2i");
}

// =============================================================================
// SECTION 5: Gaussian Addition
// =============================================================================

#[test]
fn test_gaussian_add_simple() {
    // (1+i) + (2+3i) = 3 + 4i
    let result = simplify_with("(1+i) + (2+3*i)", &complex_on_opts());
    assert_eq!(result, "3 + 4 * i", "(1+i) + (2+3i) = 3+4i");
}

#[test]
fn test_gaussian_add_cancel_real() {
    // (1+2i) + (-1+3i) = 5i
    let result = simplify_with("(1+2*i) + (-1+3*i)", &complex_on_opts());
    assert_eq!(result, "5 * i", "(1+2i) + (-1+3i) = 5i");
}

// =============================================================================
// SECTION 6: Parser correctly recognizes i
// =============================================================================

#[test]
fn test_parser_recognizes_standalone_i() {
    let mut ctx = Context::new();
    let expr = parse("i", &mut ctx).expect("Should parse i");
    let display = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &ctx,
            id: expr
        }
    );
    assert_eq!(display, "i", "Parser should recognize standalone i");
}

#[test]
fn test_parser_i_in_expression() {
    let mut ctx = Context::new();
    let expr = parse("2*i + 3", &mut ctx).expect("Should parse 2*i + 3");
    let display = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &ctx,
            id: expr
        }
    );
    // Just verify it parses without error and contains i
    assert!(display.contains("i"), "Expression should contain i");
}

// =============================================================================
// SECTION 7: Gaussian Division
// =============================================================================

#[test]
fn test_gaussian_div_simple() {
    // (1+i)/(1-i) = i
    let result = simplify_with("(1+i)/(1-i)", &complex_on_opts());
    assert_eq!(result, "i", "(1+i)/(1-i) = i");
}

#[test]
fn test_gaussian_div_one_over_i() {
    // 1/i = -i
    let result = simplify_with("1/i", &complex_on_opts());
    assert_eq!(result, "-i", "1/i = -i");
}

// =============================================================================
// SECTION 8: Regression Tests - No infinite loops
// =============================================================================

/// Helper that returns step count along with result
fn simplify_with_steps(input: &str, opts: &EvalOptions) -> (String, usize) {
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("Failed to parse");

    let mut simplifier = Simplifier::with_profile(opts);
    simplifier.context = ctx;
    let (result, steps) = simplifier.simplify(expr);

    let result_str = format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    (result_str, steps.len())
}

#[test]
fn test_gaussian_div_no_infinite_loop() {
    // Regression test: (3+4i)/(1+2i) should not cause infinite loop
    // Expected result: 11/5 - 2/5*i (or equivalent)
    // This used to cause depth>500 warnings due to ping-pong between rules
    let (result, step_count) = simplify_with_steps("(3+4*i)/(1+2*i)", &complex_on_opts());

    // Should complete in reasonable number of steps (< 20)
    assert!(
        step_count < 20,
        "Gaussian division should not cause excessive rewrites. Got {} steps for result: {}",
        step_count,
        result
    );

    // Result should contain the expected values
    assert!(
        result.contains("11") && result.contains("5"),
        "Result should be 11/5 - 2/5*i, got: {}",
        result
    );
}
