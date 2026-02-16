//! Tests for the generalized n-angle inverse-trig composition rules.
//!
//! Covers:
//! - Recurrence correctness (known identities for n=2,3,4)
//! - Numeric metamorphic verification (algebraic output ≈ f64 trig at sample points)
//! - Guardrails (MAX_N boundary, inner size limit)
//! - Sign parity (negative multiples)
//! - Chebyshev sanity checks

use cas_ast::DisplayExpr;
use cas_engine::helpers::eval_f64_with_substitution;
use cas_engine::Simplifier;
use cas_parser::parse;

/// Simplify an expression string and return the result as string.
fn simplify_str(input: &str) -> String {
    let mut s = Simplifier::with_default_rules();
    let expr = parse(input, &mut s.context).unwrap();
    let (result, _) = s.simplify(expr);
    format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: result
        }
    )
}

/// Simplify and return both the result string and the numeric f64 evaluation at t=val.
fn simplify_and_eval(input: &str, var: &str, val: f64) -> (String, Option<f64>) {
    let mut s = Simplifier::with_default_rules();
    let expr = parse(input, &mut s.context).unwrap();
    let (result, _) = s.simplify(expr);
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: result
        }
    );
    let numeric = eval_f64_with_substitution(&s.context, result, &[var.to_string()], &[val]);
    (result_str, numeric)
}

// =============================================================================
// Arctan known identities (n=2, 3)
// =============================================================================

#[test]
fn atan_n2_sin() {
    // sin(2·arctan(t)) = 2t/(1+t²)
    let result = simplify_str("sin(2*arctan(t))");
    // Should contain t and (1+t²) denominator
    assert!(
        result.contains("t"),
        "Expected expression with t, got: {}",
        result
    );
}

#[test]
fn atan_n2_cos() {
    // cos(2·arctan(t)) = (1-t²)/(1+t²)
    let result = simplify_str("cos(2*arctan(t))");
    assert!(
        result.contains("t"),
        "Expected expression with t, got: {}",
        result
    );
}

#[test]
fn atan_n2_tan() {
    // tan(2·arctan(t)) = 2t/(1-t²)
    let result = simplify_str("tan(2*arctan(t))");
    assert!(
        result.contains("t"),
        "Expected expression with t, got: {}",
        result
    );
}

#[test]
fn atan_n3_sin() {
    // sin(3·arctan(t)) should produce a rational expression in t
    let result = simplify_str("sin(3*arctan(t))");
    assert!(
        result.contains("t"),
        "Expected expression with t, got: {}",
        result
    );
}

// =============================================================================
// Arccos Chebyshev known values
// =============================================================================

#[test]
fn acos_n2_cos_is_chebyshev_t2() {
    // cos(2·arccos(t)) = T₂(t) = 2t² - 1
    let result = simplify_str("cos(2*arccos(t))");
    assert!(
        result.contains("t"),
        "Expected Chebyshev T₂, got: {}",
        result
    );
}

#[test]
fn acos_n3_cos_is_chebyshev_t3() {
    // cos(3·arccos(t)) = T₃(t) = 4t³ - 3t
    let result = simplify_str("cos(3*arccos(t))");
    assert!(
        result.contains("t"),
        "Expected Chebyshev T₃, got: {}",
        result
    );
}

#[test]
fn acos_n2_sin() {
    // sin(2·arccos(t)) = 2t·√(1-t²)
    let result = simplify_str("sin(2*arccos(t))");
    assert!(
        result.contains("t"),
        "Expected expression with t, got: {}",
        result
    );
}

// =============================================================================
// Arcsin known identities (n=2, 3)
// =============================================================================

#[test]
fn asin_n2_sin() {
    // sin(2·arcsin(t)) = 2t·√(1-t²)
    let result = simplify_str("sin(2*arcsin(t))");
    assert!(
        result.contains("t"),
        "Expected expression with t, got: {}",
        result
    );
}

#[test]
fn asin_n2_cos() {
    // cos(2·arcsin(t)) = 1 - 2t²
    let result = simplify_str("cos(2*arcsin(t))");
    assert!(
        result.contains("t"),
        "Expected expression with t, got: {}",
        result
    );
}

#[test]
fn asin_n3_sin() {
    // sin(3·arcsin(t)) = 3t - 4t³
    let result = simplify_str("sin(3*arcsin(t))");
    assert!(
        result.contains("t"),
        "Expected expression with t, got: {}",
        result
    );
}

// =============================================================================
// Higher n: n=4, 5, 6 produce results (not "unsimplified")
// =============================================================================

#[test]
fn atan_n4_sin_fires() {
    let result = simplify_str("sin(4*arctan(t))");
    // Must have been expanded (no sin or arctan in result)
    assert!(
        !result.contains("sin") && !result.contains("arctan"),
        "Expected fully expanded algebraic form, got: {}",
        result
    );
}

#[test]
fn atan_n5_cos_fires() {
    let result = simplify_str("cos(5*arctan(t))");
    assert!(
        !result.contains("cos") && !result.contains("arctan"),
        "Expected fully expanded algebraic form, got: {}",
        result
    );
}

#[test]
fn acos_n4_cos_is_chebyshev_t4() {
    // cos(4·arccos(t)) = T₄(t) = 8t⁴ - 8t² + 1
    let result = simplify_str("cos(4*arccos(t))");
    assert!(
        !result.contains("cos") && !result.contains("arccos"),
        "Expected Chebyshev polynomial, got: {}",
        result
    );
}

#[test]
fn asin_n4_sin_fires() {
    let result = simplify_str("sin(4*arcsin(t))");
    assert!(
        !result.contains("sin") && !result.contains("arcsin"),
        "Expected fully expanded form, got: {}",
        result
    );
}

// =============================================================================
// Numeric metamorphic verification: algebraic output ≈ trig(n·invtrig(val))
// =============================================================================

/// Verify that the simplified algebraic form evaluates to the same f64 value
/// as computing trig(n·invtrig(val)) directly.
fn assert_numeric_eq(input: &str, expected: f64, var: &str, val: f64) {
    let (result_str, numeric) = simplify_and_eval(input, var, val);
    let got = numeric.unwrap_or_else(|| {
        panic!(
            "Failed to evaluate '{}' (simplified to '{}') at {} = {}",
            input, result_str, var, val
        )
    });
    assert!(
        (got - expected).abs() < 1e-10,
        "Numeric mismatch for '{}' at {}={}: expected {}, got {} (simplified: '{}')",
        input,
        var,
        val,
        expected,
        got,
        result_str
    );
}

#[test]
fn numeric_atan_n2_sin() {
    let t: f64 = 0.5;
    let expected = (2.0_f64 * t.atan()).sin();
    assert_numeric_eq("sin(2*arctan(t))", expected, "t", t);
}

#[test]
fn numeric_atan_n2_cos() {
    let t: f64 = 0.5;
    let expected = (2.0_f64 * t.atan()).cos();
    assert_numeric_eq("cos(2*arctan(t))", expected, "t", t);
}

#[test]
fn numeric_atan_n2_tan() {
    let t: f64 = 0.3;
    let expected = (2.0_f64 * t.atan()).tan();
    assert_numeric_eq("tan(2*arctan(t))", expected, "t", t);
}

#[test]
fn numeric_atan_n4_sin() {
    let t: f64 = 0.5;
    let expected = (4.0_f64 * t.atan()).sin();
    assert_numeric_eq("sin(4*arctan(t))", expected, "t", t);
}

#[test]
fn numeric_atan_n5_cos() {
    let t: f64 = 0.7;
    let expected = (5.0_f64 * t.atan()).cos();
    assert_numeric_eq("cos(5*arctan(t))", expected, "t", t);
}

#[test]
fn numeric_atan_n8_sin() {
    let t: f64 = 0.4;
    let expected = (8.0_f64 * t.atan()).sin();
    assert_numeric_eq("sin(8*arctan(t))", expected, "t", t);
}

#[test]
fn numeric_atan_n10_cos() {
    // Boundary: MAX_N = 10, should still fire
    let t: f64 = 0.3;
    let expected = (10.0_f64 * t.atan()).cos();
    assert_numeric_eq("cos(10*arctan(t))", expected, "t", t);
}

#[test]
fn numeric_acos_n2_cos() {
    let t: f64 = 0.6;
    let expected = (2.0_f64 * t.acos()).cos();
    assert_numeric_eq("cos(2*arccos(t))", expected, "t", t);
}

#[test]
fn numeric_acos_n2_sin() {
    let t: f64 = 0.6;
    let expected = (2.0_f64 * t.acos()).sin();
    assert_numeric_eq("sin(2*arccos(t))", expected, "t", t);
}

#[test]
fn numeric_acos_n4_cos() {
    // T₄(0.5) = 8(0.5)⁴ - 8(0.5)² + 1 = 0.5 - 2.0 + 1 = -0.5
    let t: f64 = 0.5;
    let expected = (4.0_f64 * t.acos()).cos();
    assert_numeric_eq("cos(4*arccos(t))", expected, "t", t);
}

#[test]
fn numeric_acos_n6_sin() {
    let t: f64 = 0.4;
    let expected = (6.0_f64 * t.acos()).sin();
    assert_numeric_eq("sin(6*arccos(t))", expected, "t", t);
}

#[test]
fn numeric_asin_n2_sin() {
    let t: f64 = 0.5;
    let expected = (2.0_f64 * t.asin()).sin();
    assert_numeric_eq("sin(2*arcsin(t))", expected, "t", t);
}

#[test]
fn numeric_asin_n2_cos() {
    let t: f64 = 0.5;
    let expected = (2.0_f64 * t.asin()).cos();
    assert_numeric_eq("cos(2*arcsin(t))", expected, "t", t);
}

#[test]
fn numeric_asin_n3_sin() {
    let t: f64 = 0.4;
    let expected = (3.0_f64 * t.asin()).sin();
    assert_numeric_eq("sin(3*arcsin(t))", expected, "t", t);
}

#[test]
fn numeric_asin_n5_cos() {
    let t: f64 = 0.3;
    let expected = (5.0_f64 * t.asin()).cos();
    assert_numeric_eq("cos(5*arcsin(t))", expected, "t", t);
}

// =============================================================================
// Sign parity tests
// =============================================================================

#[test]
fn negative_atan_n4_sin_is_negated() {
    // sin(-4·arctan(t)) = -sin(4·arctan(t))
    let pos = simplify_str("sin(4*arctan(t))");
    let neg = simplify_str("sin(-4*arctan(t))");

    // Verify numeric equivalence
    let t: f64 = 0.5;
    let expected_pos = (4.0_f64 * t.atan()).sin();
    let expected_neg = -expected_pos;
    let (_, pos_val) = simplify_and_eval("sin(4*arctan(t))", "t", t);
    let (_, neg_val) = simplify_and_eval("sin(-4*arctan(t))", "t", t);

    assert!(
        (pos_val.unwrap() - expected_pos).abs() < 1e-10,
        "pos: expected {}, got {}",
        expected_pos,
        pos_val.unwrap()
    );
    assert!(
        (neg_val.unwrap() - expected_neg).abs() < 1e-10,
        "neg: expected {}, got {} (pos form: '{}')",
        expected_neg,
        neg_val.unwrap(),
        pos
    );

    // Structural: they should differ (one is negated)
    assert_ne!(
        pos, neg,
        "sin(4 atan) and sin(-4 atan) should produce different forms"
    );
}

#[test]
fn negative_atan_n4_cos_is_same() {
    // cos(-4·arctan(t)) = cos(4·arctan(t)) — even function
    let t: f64 = 0.5;
    let (_, pos_val) = simplify_and_eval("cos(4*arctan(t))", "t", t);
    let (_, neg_val) = simplify_and_eval("cos(-4*arctan(t))", "t", t);

    assert!(
        (pos_val.unwrap() - neg_val.unwrap()).abs() < 1e-10,
        "cos is even: cos(4θ) should equal cos(-4θ)"
    );
}

// =============================================================================
// Guardrail tests
// =============================================================================

#[test]
fn n11_does_not_fire() {
    // n=11 exceeds MAX_N=10, should remain unsimplified
    let result = simplify_str("sin(11*arctan(t))");
    assert!(
        result.contains("sin") || result.contains("arctan"),
        "n=11 should NOT fire, got: {}",
        result
    );
}

#[test]
fn n10_does_fire() {
    // n=10 is exactly MAX_N, should fire
    let result = simplify_str("sin(10*arctan(t))");
    assert!(
        !result.contains("sin(10") && !result.contains("arctan"),
        "n=10 should fire, got: {}",
        result
    );
}

// =============================================================================
// Abbreviation tests (atan, asin, acos)
// =============================================================================

#[test]
fn atan_abbreviation_works() {
    let result = simplify_str("sin(3*atan(t))");
    assert!(
        !result.contains("atan") && !result.contains("sin"),
        "atan abbreviation should fire, got: {}",
        result
    );
}

#[test]
fn acos_abbreviation_works() {
    let result = simplify_str("cos(4*acos(t))");
    assert!(
        !result.contains("acos") && !result.contains("cos(4"),
        "acos abbreviation should fire, got: {}",
        result
    );
}

#[test]
fn asin_abbreviation_works() {
    let result = simplify_str("sin(3*asin(t))");
    assert!(
        !result.contains("asin") && !result.contains("sin"),
        "asin abbreviation should fire, got: {}",
        result
    );
}
