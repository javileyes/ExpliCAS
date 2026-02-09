//! Domain mode contract tests.
//!
//! These tests verify the behavior of factor cancellation under different
//! domain modes (Strict, Assume, Generic).
//!
//! # Contract
//!
//! - **Generic** (default): Allow `x/x → 1` (legacy CAS behavior)
//! - **Strict**: Only cancel if factor is provably non-zero
//! - **Assume**: Like Strict, but uses user-provided assumptions (future)
//!
//! # Tests organized by mode:
//!
//! - `test_generic_*`: Current behavior (should pass now)
//! - `test_strict_*`: Expected behavior after gate (will fail until implemented)

use cas_ast::Context;
use cas_engine::Simplifier;
use cas_parser::parse;

/// Helper: simplify with default options (Generic mode)
fn simplify_generic(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).expect("parse failed");
    let (result, _) = simplifier.simplify(expr);
    format!(
        "{}",
        cas_ast::display::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

/// Helper: simplify with Strict domain mode
fn simplify_strict(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input, &mut simplifier.context).expect("parse failed");

    // Use Strict domain mode
    let opts = cas_engine::SimplifyOptions {
        shared: cas_engine::phase::SharedSemanticConfig {
            semantics: cas_engine::semantics::EvalConfig {
                domain_mode: cas_engine::DomainMode::Strict,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let (result, _) = simplifier.simplify_with_options(expr, opts);
    format!(
        "{}",
        cas_ast::display::DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

// =============================================================================
// Generic Mode Tests (should pass now - preserves existing behavior)
// =============================================================================

#[test]
fn test_generic_x_div_x_simplifies_to_1() {
    // Generic mode: x/x → 1 (classic CAS behavior)
    let result = simplify_generic("x/x");
    assert_eq!(result, "1", "Generic mode should simplify x/x to 1");
}

#[test]
fn test_generic_2_div_2_simplifies_to_1() {
    // Both modes: 2/2 → 1 (2 is provably non-zero)
    let result = simplify_generic("2/2");
    assert_eq!(result, "1");
}

#[test]
fn test_generic_pi_div_pi_simplifies_to_1() {
    // Both modes: π/π → 1 (π is provably non-zero)
    let result = simplify_generic("pi/pi");
    assert_eq!(result, "1");
}

// =============================================================================
// Strict Mode Tests (will fail until gate is implemented)
// =============================================================================

#[test]
fn test_strict_x_div_x_stays_unchanged() {
    // Strict mode: x/x should NOT simplify (x could be 0)
    let result = simplify_strict("x/x");
    assert_eq!(result, "x / x", "Strict mode should NOT simplify x/x");
}

#[test]
fn test_strict_2_div_2_simplifies_to_1() {
    // Strict mode: 2/2 → 1 (2 is provably non-zero)
    let result = simplify_strict("2/2");
    assert_eq!(result, "1", "Strict mode should simplify 2/2 to 1");
}

#[test]
fn test_strict_neg3_div_neg3_simplifies_to_1() {
    // Strict mode: (-3)/(-3) → 1 (-3 is provably non-zero)
    let result = simplify_strict("(-3)/(-3)");
    assert_eq!(result, "1", "Strict mode should simplify (-3)/(-3) to 1");
}

#[test]
fn test_strict_pi_div_pi_simplifies_to_1() {
    // Strict mode: π/π → 1 (π is provably non-zero)
    let result = simplify_strict("pi/pi");
    assert_eq!(result, "1", "Strict mode should simplify pi/pi to 1");
}

#[test]
fn test_strict_2x_div_2x_stays_unchanged() {
    // Strict mode: (2*x)/(2*x) should NOT simplify to 1 (x could be 0)
    // It might partially simplify but shouldn't collapse to 1
    let result = simplify_strict("(2*x)/(2*x)");
    // The result should either be unchanged OR be x/x (partial cancel of 2)
    // It should NOT be "1"
    assert_ne!(
        result, "1",
        "Strict mode should NOT fully simplify (2*x)/(2*x) to 1"
    );
}

#[test]
fn test_strict_x_squared_div_x_squared_stays_unchanged() {
    // Strict mode: x²/x² should NOT simplify (x could be 0)
    let result = simplify_strict("x^2/x^2");
    assert_ne!(result, "1", "Strict mode should NOT simplify x^2/x^2 to 1");
}

// =============================================================================
// Strict Mode Power Tests
// =============================================================================

#[test]
fn test_strict_x_pow_0_stays_unchanged() {
    // Strict mode: x^0 should NOT simplify to 1 (x could be 0, and 0^0 is indeterminate)
    let result = simplify_strict("x^0");
    assert_ne!(result, "1", "Strict mode should NOT simplify x^0 to 1");
}

#[test]
fn test_strict_2_pow_0_simplifies_to_1() {
    // Strict mode: 2^0 → 1 (2 is provably non-zero)
    let result = simplify_strict("2^0");
    assert_eq!(result, "1", "Strict mode should simplify 2^0 to 1");
}

#[test]
fn test_strict_pi_pow_0_simplifies_to_1() {
    // Strict mode: π^0 → 1 (π is provably non-zero)
    let result = simplify_strict("pi^0");
    assert_eq!(result, "1", "Strict mode should simplify pi^0 to 1");
}

#[test]
fn test_strict_0_pow_x_stays_unchanged() {
    // Strict mode: 0^x should NOT simplify to 0 (x could be 0 or negative)
    let result = simplify_strict("0^x");
    // Should remain 0^x or be undefined, not "0"
    assert_ne!(result, "0", "Strict mode should NOT simplify 0^x to 0");
}

#[test]
fn test_strict_0_pow_2_simplifies_to_0() {
    // Strict mode: 0^2 → 0 (exponent 2 is provably positive)
    let result = simplify_strict("0^2");
    assert_eq!(result, "0", "Strict mode should simplify 0^2 to 0");
}

// =============================================================================
// Strict Mode Log/Exp Inverse Tests
// =============================================================================

#[test]
fn test_strict_ln_exp_x_simplifies() {
    // NEW CONTRACT: In RealOnly mode, ln(e^x) DOES simplify to x
    // Because x ∈ ℝ by contract, and e^x > 0 for all real x
    let result = simplify_strict("ln(exp(x))");
    assert_eq!(
        result, "x",
        "RealOnly+Strict: ln(exp(x)) should simplify to x (x ∈ ℝ by contract), got: {}",
        result
    );
}

#[test]
fn test_strict_exp_ln_x_stays_unchanged() {
    // Strict mode: e^ln(x) should NOT simplify to x (requires x > 0)
    let result = simplify_strict("exp(ln(x))");
    // Should NOT be just "x"
    assert_ne!(
        result, "x",
        "Strict mode should NOT simplify exp(ln(x)) to x"
    );
}

#[test]
fn test_strict_exp_ln_2_simplifies() {
    // Strict mode: e^ln(2) → 2 (2 is provably positive)
    let result = simplify_strict("exp(ln(2))");
    // Should simplify to 2 (numeric)
    assert_eq!(result, "2", "Strict mode should simplify exp(ln(2)) to 2");
}

#[test]
fn test_strict_ln_exp_2_simplifies() {
    // Strict mode: ln(e^2) → 2 (numeric exponent, always valid)
    let result = simplify_strict("ln(exp(2))");
    assert_eq!(result, "2", "Strict mode should simplify ln(exp(2)) to 2");
}

// =============================================================================
// Strict Mode Root Tests
// =============================================================================

#[test]
fn test_strict_sqrt_x_squared_stays_unchanged() {
    // Strict mode: sqrt(x^2) should NOT simplify to x (x could be negative, result is |x|)
    let result = simplify_strict("sqrt(x^2)");
    assert_ne!(
        result, "x",
        "Strict mode should NOT simplify sqrt(x^2) to x"
    );
}

#[test]
fn test_strict_sqrt_4_simplifies() {
    // Strict mode: sqrt(4) → 2 (numeric)
    let result = simplify_strict("sqrt(4)");
    assert_eq!(result, "2", "Strict mode should simplify sqrt(4) to 2");
}

// =============================================================================
// Strict Mode Composite Expression Tests
// =============================================================================

#[test]
fn test_strict_x_minus_1_div_x_minus_1_stays_unchanged() {
    // Strict mode: (x-1)/(x-1) should NOT simplify (x-1 could be 0 when x=1)
    let result = simplify_strict("(x-1)/(x-1)");
    assert_ne!(
        result, "1",
        "Strict mode should NOT simplify (x-1)/(x-1) to 1"
    );
}

#[test]
fn test_strict_sin_x_div_sin_x_stays_unchanged() {
    // Strict mode: sin(x)/sin(x) should NOT simplify (sin(x) could be 0)
    let result = simplify_strict("sin(x)/sin(x)");
    assert_ne!(
        result, "1",
        "Strict mode should NOT simplify sin(x)/sin(x) to 1"
    );
}

#[test]
fn test_strict_x_plus_y_div_x_plus_y_stays_unchanged() {
    // Strict mode: (x+y)/(x+y) should NOT simplify (x+y could be 0)
    let result = simplify_strict("(x+y)/(x+y)");
    assert_ne!(
        result, "1",
        "Strict mode should NOT simplify (x+y)/(x+y) to 1"
    );
}

// =============================================================================
// Proof Helper Tests
// =============================================================================

#[test]
fn test_prove_nonzero_numbers() {
    use cas_engine::helpers::prove_nonzero;
    use cas_engine::Proof;

    let mut ctx = Context::new();

    let zero = ctx.num(0);
    let two = ctx.num(2);
    let neg_three = ctx.num(-3);
    let half = ctx.add(cas_ast::Expr::Number(num_rational::BigRational::new(
        1.into(),
        2.into(),
    )));

    assert_eq!(prove_nonzero(&ctx, zero), Proof::Disproven);
    assert_eq!(prove_nonzero(&ctx, two), Proof::Proven);
    assert_eq!(prove_nonzero(&ctx, neg_three), Proof::Proven);
    assert_eq!(prove_nonzero(&ctx, half), Proof::Proven);
}

#[test]
fn test_prove_nonzero_constants() {
    use cas_engine::helpers::prove_nonzero;
    use cas_engine::Proof;

    let mut ctx = Context::new();

    let pi = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::Pi));
    let e = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::E));
    let i = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::I));

    assert_eq!(prove_nonzero(&ctx, pi), Proof::Proven);
    assert_eq!(prove_nonzero(&ctx, e), Proof::Proven);
    assert_eq!(prove_nonzero(&ctx, i), Proof::Proven);
}

#[test]
fn test_prove_nonzero_variables() {
    use cas_engine::helpers::prove_nonzero;
    use cas_engine::Proof;

    let mut ctx = Context::new();

    let x = ctx.var("x");
    let y = ctx.var("y");

    assert_eq!(prove_nonzero(&ctx, x), Proof::Unknown);
    assert_eq!(prove_nonzero(&ctx, y), Proof::Unknown);
}

#[test]
fn test_prove_nonzero_products() {
    use cas_engine::helpers::prove_nonzero;
    use cas_engine::Proof;

    let mut ctx = Context::new();

    let two = ctx.num(2);
    let three = ctx.num(3);
    let zero = ctx.num(0);
    let x = ctx.var("x");

    // 2 * 3 = 6, proven nonzero
    let two_times_three = ctx.add(cas_ast::Expr::Mul(two, three));
    assert_eq!(prove_nonzero(&ctx, two_times_three), Proof::Proven);

    // 2 * 0 = 0, disproven nonzero
    let two_times_zero = ctx.add(cas_ast::Expr::Mul(two, zero));
    assert_eq!(prove_nonzero(&ctx, two_times_zero), Proof::Disproven);

    // 2 * x = unknown (x could be 0)
    let two_times_x = ctx.add(cas_ast::Expr::Mul(two, x));
    assert_eq!(prove_nonzero(&ctx, two_times_x), Proof::Unknown);
}

// =============================================================================
// prove_positive Helper Tests
// =============================================================================

#[test]
fn test_prove_positive_numbers() {
    use cas_engine::helpers::prove_positive;
    use cas_engine::semantics::ValueDomain;
    use cas_engine::Proof;

    let mut ctx = Context::new();
    let vd = ValueDomain::RealOnly; // Use RealOnly for these basic tests

    let zero = ctx.num(0);
    let two = ctx.num(2);
    let neg_three = ctx.num(-3);

    assert_eq!(prove_positive(&ctx, zero, vd), Proof::Disproven);
    assert_eq!(prove_positive(&ctx, two, vd), Proof::Proven);
    assert_eq!(prove_positive(&ctx, neg_three, vd), Proof::Disproven);
}

#[test]
fn test_prove_positive_constants() {
    use cas_engine::helpers::prove_positive;
    use cas_engine::semantics::ValueDomain;
    use cas_engine::Proof;

    let mut ctx = Context::new();
    let vd = ValueDomain::RealOnly;

    let pi = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::Pi));
    let e = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::E));
    let i = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::I));

    assert_eq!(prove_positive(&ctx, pi, vd), Proof::Proven);
    assert_eq!(prove_positive(&ctx, e, vd), Proof::Proven);
    assert_eq!(prove_positive(&ctx, i, vd), Proof::Unknown); // i is not real-positive
}

#[test]
fn test_prove_positive_products() {
    use cas_engine::helpers::prove_positive;
    use cas_engine::semantics::ValueDomain;
    use cas_engine::Proof;

    let mut ctx = Context::new();
    let vd = ValueDomain::RealOnly;

    let two = ctx.num(2);
    let three = ctx.num(3);
    let neg_two = ctx.num(-2);

    // 2 * 3 = 6, proven positive
    let two_times_three = ctx.add(cas_ast::Expr::Mul(two, three));
    assert_eq!(prove_positive(&ctx, two_times_three, vd), Proof::Proven);

    // 2 * (-2) = -4, NOT proven positive (but could be - depends on logic)
    // Our implementation returns Unknown for mixed signs
    let two_times_neg = ctx.add(cas_ast::Expr::Mul(two, neg_two));
    assert_eq!(prove_positive(&ctx, two_times_neg, vd), Proof::Unknown);
}

#[test]
fn test_prove_positive_exp() {
    use cas_engine::helpers::prove_positive;
    use cas_engine::semantics::ValueDomain;
    use cas_engine::Proof;

    let mut ctx = Context::new();

    let x = ctx.var("x");
    let exp_x = ctx.call_builtin(cas_ast::BuiltinFn::Exp, vec![x]);

    // NEW CONTRACT:
    // In RealOnly: exp(x) > 0 for ALL x (x is real by contract)
    assert_eq!(
        prove_positive(&ctx, exp_x, ValueDomain::RealOnly),
        Proof::Proven,
        "RealOnly: exp(x) should be Proven positive (x ∈ ℝ by contract)"
    );

    // In ComplexEnabled: exp(x) with variable x is Unknown (could be complex)
    assert_eq!(
        prove_positive(&ctx, exp_x, ValueDomain::ComplexEnabled),
        Proof::Unknown,
        "ComplexEnabled: exp(x) should be Unknown (x could be complex)"
    );

    // exp(literal): Proven in both modes (literal is always real)
    let two = ctx.num(2);
    let exp_two = ctx.call_builtin(cas_ast::BuiltinFn::Exp, vec![two]);
    assert_eq!(
        prove_positive(&ctx, exp_two, ValueDomain::RealOnly),
        Proof::Proven
    );
    assert_eq!(
        prove_positive(&ctx, exp_two, ValueDomain::ComplexEnabled),
        Proof::Proven
    );

    // exp(π): Proven in both modes (constant is real)
    let pi = ctx.add(cas_ast::Expr::Constant(cas_ast::Constant::Pi));
    let exp_pi = ctx.call_builtin(cas_ast::BuiltinFn::Exp, vec![pi]);
    assert_eq!(
        prove_positive(&ctx, exp_pi, ValueDomain::RealOnly),
        Proof::Proven
    );
    assert_eq!(
        prove_positive(&ctx, exp_pi, ValueDomain::ComplexEnabled),
        Proof::Proven
    );
}

// =============================================================================
// Log Expansion Gate Tests (Integration)
// =============================================================================

/// Helper: simplify with DomainMode and ValueDomain via Engine
fn simplify_with_domain_value(
    input: &str,
    domain: cas_engine::DomainMode,
    value: cas_engine::semantics::ValueDomain,
) -> (String, Vec<String>) {
    use cas_engine::{Engine, EntryKind, EvalAction, EvalRequest, EvalResult, SessionState};

    let mut engine = Engine::new();
    let mut state = SessionState::new();

    state.options.shared.semantics.domain_mode = domain;
    state.options.shared.semantics.value_domain = value;

    let parsed = cas_parser::parse(input, &mut engine.simplifier.context).expect("parse failed");
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        kind: EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");

    let result_str = match &output.result {
        EvalResult::Expr(e) => cas_ast::DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    };

    let warnings: Vec<String> = output
        .domain_warnings
        .iter()
        .map(|w| w.message.clone())
        .collect();

    (result_str, warnings)
}

/// CONTRACT: ln(x*y) does NOT expand in Strict mode (variables not provably positive)
#[test]
fn log_product_strict_no_expand_variables() {
    use cas_engine::semantics::ValueDomain;
    let (result, warnings) = simplify_with_domain_value(
        "ln(x*y)",
        cas_engine::DomainMode::Strict,
        ValueDomain::RealOnly,
    );

    // Should remain unexpanded because x, y are not provably > 0
    // Check that result contains ln or log, both variables, and does NOT contain '+' (which would mean expansion)
    let has_log = result.contains("log") || result.contains("ln");
    let is_unexpanded =
        has_log && result.contains("x") && result.contains("y") && !result.contains("+");
    assert!(
        is_unexpanded,
        "Expected ln(x*y) to remain unexpanded in Strict mode, got: {}",
        result
    );

    assert!(
        warnings.is_empty(),
        "No warnings expected in Strict mode, got: {:?}",
        warnings
    );
}

/// CONTRACT: ln(x*y) does NOT expand automatically in Assume mode anymore
/// LogExpansionRule is now opt-in via the `expand_log` command.
/// This test verifies that automatic simplification does NOT expand logs.
#[test]
fn log_product_assume_no_auto_expand() {
    use cas_engine::semantics::ValueDomain;
    let (result, _warnings) = simplify_with_domain_value(
        "ln(x*y)",
        cas_engine::DomainMode::Assume,
        ValueDomain::RealOnly,
    );

    // Should NOT expand automatically (LogExpansionRule not in defaults)
    // Use `expand_log ln(x*y)` command for explicit expansion
    assert!(
        !result.contains("+"),
        "Expected ln(x*y) to NOT auto-expand (use expand_log command), got: {}",
        result
    );
}

/// CONTRACT: ln(z*w) does NOT expand in Complex domain (branch cut safety)
#[test]
fn log_product_complex_never_expands() {
    use cas_engine::semantics::ValueDomain;
    let (result, _) = simplify_with_domain_value(
        "ln(z*w)",
        cas_engine::DomainMode::Assume,
        ValueDomain::ComplexEnabled,
    );

    // Should NOT expand due to branch cut concerns - should still contain a product
    assert!(
        result.contains("z") && result.contains("w") && !result.contains("+"),
        "Expected ln(z*w) to NOT expand in Complex domain, got: {}",
        result
    );
}

/// CONTRACT: ln(2*pi) does NOT expand automatically even though both factors are provably positive
/// LogExpansionRule is now opt-in via the `expand_log` command.
#[test]
fn log_product_provable_positive_no_auto_expand() {
    use cas_engine::semantics::ValueDomain;
    let (result, _warnings) = simplify_with_domain_value(
        "ln(2*pi)",
        cas_engine::DomainMode::Strict,
        ValueDomain::RealOnly,
    );

    // Should NOT expand automatically (LogExpansionRule not in defaults)
    // Use `expand_log ln(2*pi)` command for explicit expansion
    assert!(
        !result.contains("+"),
        "Expected ln(2*pi) to NOT auto-expand (use expand_log command), got: {}",
        result
    );
}

// =============================================================================
// Same-Denominator Fraction Combination Tests
// =============================================================================

/// CONTRACT: 1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1) simplifies to 0
/// This is the user's original failing case that motivated the rule
#[test]
fn same_denom_fractions_combine_to_zero() {
    let result = simplify_generic("1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)");

    assert_eq!(
        result, "0",
        "Expected expression to simplify to 0, got: {}",
        result
    );
}

/// CONTRACT: a/d + b/d combines into single fraction
#[test]
fn same_denom_fractions_combine_simple() {
    let result = simplify_generic("x/(2*x+1) + y/(2*x+1)");

    // Should combine into single fraction - count number of '/'
    let div_count = result.matches('/').count();
    assert!(
        div_count <= 1,
        "Expected fractions to combine into one, got: {} (/ count: {})",
        result,
        div_count
    );
}

/// CONTRACT: 1 + a/d - a/d simplifies to 1 (fractions cancel even with integer mixed in)
#[test]
fn same_denom_mixed_cancellation() {
    let result = simplify_generic("1 + x/(x+1) - x/(x+1)");

    assert_eq!(
        result, "1",
        "Expected 1 + x/(x+1) - x/(x+1) to simplify to 1, got: {}",
        result
    );
}

// =============================================================================
// ValueDomain Complex Rules Tests
// =============================================================================

/// CONTRACT: In RealOnly mode, 10/(3+4i) does NOT get Gaussian rationalized
/// (i is treated as an ordinary symbol, not the imaginary unit)
#[test]
fn value_domain_real_no_gaussian_division() {
    use cas_engine::semantics::ValueDomain;
    let (result, _) = simplify_with_domain_value(
        "10/(3+4*i)",
        cas_engine::DomainMode::Generic,
        ValueDomain::RealOnly,
    );

    // Should NOT be rationalized - i is just a symbol
    // Result should still contain division by expression with i
    assert!(
        result.contains("/") && result.contains("i"),
        "Expected 10/(3+4i) to remain unchanged in RealOnly mode, got: {}",
        result
    );
}

/// CONTRACT: In ComplexEnabled mode, 10/(3+4i) IS Gaussian rationalized
#[test]
fn value_domain_complex_gaussian_division() {
    use cas_engine::semantics::ValueDomain;
    let (result, _) = simplify_with_domain_value(
        "10/(3+4*i)",
        cas_engine::DomainMode::Generic,
        ValueDomain::ComplexEnabled,
    );

    // Should be rationalized to 6/5 - 8/5*i or equivalent
    // Key check: no longer has denominator with i
    let has_complex_denom = result.contains("/(") && result.contains("i)");
    assert!(
        !has_complex_denom,
        "Expected 10/(3+4i) to be Gaussian rationalized in ComplexEnabled mode, got: {}",
        result
    );
}

/// CONTRACT: In RealOnly mode, i*i does NOT simplify to -1
#[test]
fn value_domain_real_no_i_squared() {
    use cas_engine::semantics::ValueDomain;
    let (result, _) = simplify_with_domain_value(
        "i*i",
        cas_engine::DomainMode::Generic,
        ValueDomain::RealOnly,
    );

    // Should NOT simplify to -1 (i is just a symbol)
    assert_ne!(
        result, "-1",
        "Expected i*i to remain unchanged in RealOnly mode, got: {}",
        result
    );
}

/// CONTRACT: In ComplexEnabled mode, i*i simplifies to -1
#[test]
fn value_domain_complex_i_squared() {
    use cas_engine::semantics::ValueDomain;
    let (result, _) = simplify_with_domain_value(
        "i*i",
        cas_engine::DomainMode::Generic,
        ValueDomain::ComplexEnabled,
    );

    // Should simplify to -1
    assert_eq!(
        result, "-1",
        "Expected i*i to simplify to -1 in ComplexEnabled mode, got: {}",
        result
    );
}

/// CONTRACT: In RealOnly mode, i^2 does NOT simplify to -1
#[test]
fn value_domain_real_no_i_power() {
    use cas_engine::semantics::ValueDomain;
    let (result, _) = simplify_with_domain_value(
        "i^2",
        cas_engine::DomainMode::Generic,
        ValueDomain::RealOnly,
    );

    // Should NOT simplify to -1
    assert_ne!(
        result, "-1",
        "Expected i^2 to remain unchanged in RealOnly mode, got: {}",
        result
    );
}

/// CONTRACT: In ComplexEnabled mode, i^4 simplifies to 1
#[test]
fn value_domain_complex_i_fourth() {
    use cas_engine::semantics::ValueDomain;
    let (result, _) = simplify_with_domain_value(
        "i^4",
        cas_engine::DomainMode::Generic,
        ValueDomain::ComplexEnabled,
    );

    // Should simplify to 1
    assert_eq!(
        result, "1",
        "Expected i^4 to simplify to 1 in ComplexEnabled mode, got: {}",
        result
    );
}

// =============================================================================
// Blocked Hints Tests (V1.3.1)
// =============================================================================

/// CONTRACT: exp(ln(x)) in Generic mode SIMPLIFIES to x with x > 0 as implicit require.
///
/// V2.15.4: Changed from blocking to allowing with implicit domain.
/// The condition x > 0 is IMPLICIT from ln(x), so it's not a new assumption.
/// Similar to how sqrt(x)^2 → x with requires x ≥ 0.
#[test]
fn exp_ln_x_generic_emits_positive_require() {
    use cas_engine::{Engine, EntryKind, EvalAction, EvalRequest, EvalResult, SessionState};

    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Ensure Generic mode (default)
    state.options.shared.semantics.domain_mode = cas_engine::DomainMode::Generic;

    let parsed =
        cas_parser::parse("exp(ln(x))", &mut engine.simplifier.context).expect("parse failed");
    let req = EvalRequest {
        raw_input: "exp(ln(x))".to_string(),
        parsed,
        kind: EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");

    // Result SHOULD be simplified to x (with implicit requires)
    let result_str = match &output.result {
        EvalResult::Expr(e) => cas_ast::DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    };

    assert_eq!(
        result_str, "x",
        "exp(ln(x)) should simplify to x in Generic mode with implicit requires"
    );

    // Should have x > 0 in required conditions (implicit domain)
    let required = &output.required_conditions;
    let has_positive_x = required.iter().any(|cond| {
        let display = cond.display(&engine.simplifier.context);
        display.contains("x") && display.contains(">")
    });
    assert!(
        has_positive_x,
        "Should require x > 0 for ln(x), got: {:?}",
        required
    );
}

/// CONTRACT: exp(ln(5)) in Generic mode does NOT emit hint (5 is provably positive)
#[test]
fn blocked_hint_exp_ln_5_no_hint() {
    use cas_engine::{Engine, EntryKind, EvalAction, EvalRequest, EvalResult, SessionState};

    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Ensure Generic mode
    state.options.shared.semantics.domain_mode = cas_engine::DomainMode::Generic;

    let parsed =
        cas_parser::parse("exp(ln(5))", &mut engine.simplifier.context).expect("parse failed");
    let req = EvalRequest {
        raw_input: "exp(ln(5))".to_string(),
        parsed,
        kind: EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");

    // Result should be 5 (simplification allowed)
    let result_str = match &output.result {
        EvalResult::Expr(e) => cas_ast::DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    };
    assert_eq!(result_str, "5", "exp(ln(5)) should simplify to 5");

    // No blocked hints expected
    let hints = engine.simplifier.take_blocked_hints();
    assert!(
        hints.is_empty(),
        "No hints expected for provably positive argument"
    );
}

// =============================================================================
// Timeline Assumption Tracking Tests
// =============================================================================

/// Contract test: Verify that (x*y)/x in Generic mode produces a step with assumption_events.
///
/// This tests the "proven vs assumed" feature for timeline transparency.
#[test]
fn step_tracks_assumed_nonzero_in_generic() {
    use cas_engine::{Engine, EntryKind, EvalAction, EvalRequest, EvalResult, SessionState};

    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Ensure Generic mode
    state.options.shared.semantics.domain_mode = cas_engine::DomainMode::Generic;

    let parsed =
        cas_parser::parse("(x*y)/x", &mut engine.simplifier.context).expect("parse failed");
    let req = EvalRequest {
        raw_input: "(x*y)/x".to_string(),
        parsed,
        kind: EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");

    // Should simplify to y
    let result_str = match &output.result {
        EvalResult::Expr(e) => cas_ast::DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    };
    assert_eq!(result_str, "y", "(x*y)/x should simplify to y");

    // Check that at least one step has assumption_events with NonZero(x)
    let has_nonzero_assumption = output.steps.iter().any(|step| {
        step.assumption_events().iter().any(|event| {
            matches!(
                event.key,
                cas_engine::assumptions::AssumptionKey::NonZero { .. }
            )
        })
    });

    assert!(
        has_nonzero_assumption,
        "At least one step should have NonZero assumption event for x"
    );
}

/// Contract test: Verify that exp(ln(x)) produces x > 0 as implicit require.
///
/// V2.15.4: Now uses required_conditions (implicit domain) instead of assumption_events.
/// This tests the "implicit domain requirement" pattern where ln(x) already implies x > 0.
#[test]
fn step_tracks_assumed_positive_in_assume() {
    use cas_engine::{Engine, EntryKind, EvalAction, EvalRequest, EvalResult, SessionState};

    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Use Generic mode - now works with implicit requires
    state.options.shared.semantics.domain_mode = cas_engine::DomainMode::Generic;

    let parsed =
        cas_parser::parse("exp(ln(x))", &mut engine.simplifier.context).expect("parse failed");
    let req = EvalRequest {
        raw_input: "exp(ln(x))".to_string(),
        parsed,
        kind: EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");

    // Should simplify to x
    let result_str = match &output.result {
        EvalResult::Expr(e) => cas_ast::DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    };
    assert_eq!(
        result_str, "x",
        "exp(ln(x)) should simplify to x with implicit requires"
    );

    // Check that required_conditions contain x > 0 (implicit domain)
    let has_positive_require = output.required_conditions.iter().any(|cond| {
        let display = cond.display(&engine.simplifier.context);
        display.contains("x") && display.contains("> 0")
    });

    assert!(
        has_positive_require,
        "Should have x > 0 in required_conditions (implicit from ln(x)). Got: {:?}",
        output
            .required_conditions
            .iter()
            .map(|c| c.display(&engine.simplifier.context))
            .collect::<Vec<_>>()
    );
}

// =============================================================================
// Sqrt Conjugate Collapse Rule Tests (V2.1)
// =============================================================================

/// CONTRACT: sqrt(x + sqrt(x²-1)) * (x - sqrt(x²-1)) does NOT collapse in Generic mode
/// The rule requires NonNegative condition (Analytic), which is blocked in Generic.
#[test]
fn sqrt_conjugate_collapse_blocked_in_generic() {
    use cas_engine::{Engine, EntryKind, EvalAction, EvalRequest, SessionState};

    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Ensure Generic mode (default)
    state.options.shared.semantics.domain_mode = cas_engine::DomainMode::Generic;
    state.options.hints_enabled = true;

    // Expression: sqrt(x + sqrt(x^2 - 1)) * (x - sqrt(x^2 - 1))
    let input = "sqrt(x + sqrt(x^2 - 1)) * (x - sqrt(x^2 - 1))";
    let parsed = cas_parser::parse(input, &mut engine.simplifier.context).expect("parse failed");
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        kind: EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");

    let result_str = match &output.result {
        cas_engine::EvalResult::Expr(e) => cas_ast::DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    };

    // Should NOT be collapsed to sqrt(x - sqrt(x^2 - 1))
    // The result should still contain the multiplication
    assert!(
        result_str.contains("*") || result_str.contains("·"),
        "Generic mode should NOT collapse sqrt conjugate product, got: {}",
        result_str
    );

    // Should emit a blocked hint for NonNegative condition
    let hints = engine.simplifier.take_blocked_hints();
    let has_nonnegative_hint = hints
        .iter()
        .any(|h| h.rule == "Collapse Sqrt Conjugate Product");

    // Note: hint may not be emitted if rule doesn't match structurally
    // The key test is that the expression is NOT collapsed
    if !hints.is_empty() {
        assert!(
            has_nonnegative_hint,
            "If hints present, should include Collapse Sqrt Conjugate Product, got: {:?}",
            hints.iter().map(|h| &h.rule).collect::<Vec<_>>()
        );
    }
}

/// CONTRACT: In Assume mode, sqrt conjugate product CAN collapse with assumption recorded
#[test]
fn sqrt_conjugate_collapse_allowed_in_assume() {
    use cas_engine::{Engine, EntryKind, EvalAction, EvalRequest, SessionState};

    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // Use Assume mode
    state.options.shared.semantics.domain_mode = cas_engine::DomainMode::Assume;

    // Expression: sqrt(x + sqrt(x^2 - 1)) * (x - sqrt(x^2 - 1))
    let input = "sqrt(x + sqrt(x^2 - 1)) * (x - sqrt(x^2 - 1))";
    let parsed = cas_parser::parse(input, &mut engine.simplifier.context).expect("parse failed");
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        kind: EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");

    let result_str = match &output.result {
        cas_engine::EvalResult::Expr(e) => cas_ast::DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    };

    // In Assume mode, the rule should be allowed to apply
    // Result may or may not collapse depending on other simplifications
    // The key is that if it DOES collapse, there should be a NonNegative assumption

    let has_nonnegative_assumption = output.steps.iter().any(|step| {
        step.assumption_events().iter().any(|event| {
            matches!(
                event.key,
                cas_engine::assumptions::AssumptionKey::NonNegative { .. }
            )
        })
    });

    // If the expression was collapsed, we should see NonNegative assumption
    // Note: The expression may not collapse if other rules prevent pattern matching
    if !result_str.contains("*") && !result_str.contains("·") {
        assert!(
            has_nonnegative_assumption,
            "If collapsed in Assume mode, should have NonNegative assumption event"
        );
    }
}

// =============================================================================
// RequiredConditions Tests (V2.x - Semantic Channel for Implicit Domain)
// =============================================================================

/// Helper: simplify with Generic mode and extract required_conditions
fn simplify_generic_with_required(input: &str) -> (String, Vec<String>) {
    use cas_engine::{Engine, EntryKind, EvalAction, EvalRequest, EvalResult, SessionState};

    let mut engine = Engine::new();
    let mut state = SessionState::new();

    state.options.shared.semantics.domain_mode = cas_engine::DomainMode::Generic;

    let parsed = cas_parser::parse(input, &mut engine.simplifier.context).expect("parse failed");
    let req = EvalRequest {
        raw_input: input.to_string(),
        parsed,
        kind: EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    let output = engine.eval(&mut state, req).expect("eval failed");

    let result_str = match &output.result {
        EvalResult::Expr(e) => cas_ast::DisplayExpr {
            context: &engine.simplifier.context,
            id: *e,
        }
        .to_string(),
        _ => "error".to_string(),
    };

    let required: Vec<String> = output
        .required_conditions
        .iter()
        .map(|cond| cond.display(&engine.simplifier.context))
        .collect();

    (result_str, required)
}

/// CONTRACT: sqrt(x)^2 simplifies to x in Generic mode with Requires: x ≥ 0
#[test]
fn required_conditions_sqrt_x_squared_simplifies_with_requires() {
    let (result, required) = simplify_generic_with_required("sqrt(x)^2");

    assert_eq!(
        result, "x",
        "sqrt(x)^2 should simplify to x in Generic mode"
    );
    assert!(
        required
            .iter()
            .any(|c| c.contains("x ≥ 0") || c.contains("x >= 0")),
        "Should require x ≥ 0, got: {:?}",
        required
    );
}

/// CONTRACT: sqrt(x)^2 does NOT simplify in Strict mode
#[test]
fn required_conditions_sqrt_x_squared_blocked_in_strict() {
    let result = simplify_strict("sqrt(x)^2");

    // Should remain unchanged (blocked by airbag)
    assert!(
        result.contains("√") || result.contains("sqrt") || result.contains("^"),
        "sqrt(x)^2 should NOT simplify to just x in Strict mode, got: {}",
        result
    );
}

/// CONTRACT: When sqrt functions remain in output, requires ARE still emitted (current behavior)
///
/// NOTE: "Witness survival" (suppressing requires when the witness sqrt() survives in output)
/// is a future optimization. Current system always emits non-negativity requirements.
///
/// For (x-y)/(sqrt(x)-sqrt(y)) → sqrt(x)+sqrt(y):
/// - Result contains sqrt(x) and sqrt(y) (witnesses)
/// - x ≥ 0 and y ≥ 0 are STILL emitted as requires (current behavior)
/// - A future optimization could suppress these since the sqrt()s prove the condition
#[test]
fn required_conditions_sqrt_expr_emits_nonneg_requires() {
    let (result, required) = simplify_generic_with_required("(x-y)/(sqrt(x)-sqrt(y))");

    // Should simplify to √x + √y (or equivalent)
    assert!(
        result.contains("√") || result.contains("sqrt") || result.contains("^(1/2)"),
        "Should contain sqrt in result, got: {}",
        result
    );

    // Current behavior: x≥0 and y≥0 ARE in required (witness survival not implemented)
    // This is mathematically correct - the requires are always emitted
    let has_x_nonneg = required.iter().any(|c| {
        c.contains("x ≥ 0") || c.contains("x >= 0") || (c.contains("x") && c.contains("≥"))
    });
    let has_y_nonneg = required.iter().any(|c| {
        c.contains("y ≥ 0") || c.contains("y >= 0") || (c.contains("y") && c.contains("≥"))
    });

    // At least one of x≥0 or y≥0 should be present (current behavior)
    let has_nonneg = has_x_nonneg || has_y_nonneg;
    assert!(
        has_nonneg,
        "Should have x≥0 or y≥0 in required (current behavior), got: {:?}",
        required
    );
}

/// CONTRACT: Numeric sqrt is always safe, no Requires needed
#[test]
fn required_conditions_numeric_sqrt_no_requires() {
    let (result, required) = simplify_generic_with_required("sqrt(4)^2");

    assert_eq!(result, "4", "sqrt(4)^2 should simplify to 4");
    assert!(
        required.is_empty(),
        "Numeric sqrt should not require anything, got: {:?}",
        required
    );
}
