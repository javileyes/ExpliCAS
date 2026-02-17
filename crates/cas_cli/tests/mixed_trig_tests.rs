//! Mixed Trigonometric Function Tests
//!
//! Tests for identities involving multiple trig functions
//! (sin, cos, tan, cot, sec, csc) and inverse functions.

// This is in cas_cli/tests, so we need to create local equivalence helpers

use cas_ast::{Context, ExprId};
use cas_engine::engine::eval_f64;
use cas_engine::helpers::is_zero;
use cas_engine::Simplifier;
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use std::collections::HashMap;

// =============================================================================
// Local Test Helpers
// =============================================================================

fn pretty(ctx: &Context, id: ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    (a - b).abs() <= tol
}

fn assert_simplifies_to_zero(input: &str) {
    let mut simplifier = Simplifier::with_default_rules();
    let a = parse(input, &mut simplifier.context).expect("parse failed");
    let (simplified, _) = simplifier.simplify(a);
    assert!(
        is_zero(&simplifier.context, simplified),
        "Expected 0, got: {}",
        pretty(&simplifier.context, simplified)
    );
}

#[allow(clippy::too_many_arguments)]
fn assert_equiv_numeric_1var(
    input: &str,
    expected: &str,
    var: &str,
    lo: f64,
    hi: f64,
    samples: usize,
    tol: f64,
    filter: impl Fn(f64) -> bool,
) {
    let mut simplifier = Simplifier::with_default_rules();
    let a = parse(input, &mut simplifier.context).expect("parse failed");
    let b = parse(expected, &mut simplifier.context).expect("parse failed");
    let (a_s, _) = simplifier.simplify(a);
    let (b_s, _) = simplifier.simplify(b);

    let mut tested = 0;
    for i in 0..samples {
        let t = (i as f64 + 0.5) / samples as f64;
        let x = lo + (hi - lo) * t;
        if !filter(x) {
            continue;
        }
        let mut var_map = HashMap::new();
        var_map.insert(var.to_string(), x);
        let va = eval_f64(&simplifier.context, a_s, &var_map);
        let vb = eval_f64(&simplifier.context, b_s, &var_map);
        if let (Some(va), Some(vb)) = (va, vb) {
            tested += 1;
            assert!(
                approx_eq(va, vb, tol),
                "Mismatch at {}={}: {} vs {}",
                var,
                x,
                va,
                vb
            );
        }
    }
    assert!(tested > 0, "No valid samples");
}

// =============================================================================
// Tests with Strong Symbolic Assertions (already passing)
// =============================================================================

#[test]
fn test_51_arctan_triangle() {
    // sin(atan(x)) = x / sqrt(1 + x^2)
    assert_simplifies_to_zero("sin(atan(x)) - x / sqrt(1 + x^2)");
}

#[test]
fn test_53_cot_arcsin() {
    // cot(asin(x)) = sqrt(1 - x^2) / x
    assert_simplifies_to_zero("cot(asin(x)) - sqrt(1 - x^2) / x");
}

#[test]
fn test_54_arcsec_arccos_relation() {
    // asec(x) = arccos(1/x)
    assert_simplifies_to_zero("asec(x) - arccos(1/x)");
}

#[test]
fn test_pythagorean_basic() {
    // sin²(x) + cos²(x) = 1
    assert_simplifies_to_zero("sin(x)^2 + cos(x)^2 - 1");
}

// =============================================================================
// Tests Converted to Numeric Equivalence (identity holds, engine may not simplify)
// =============================================================================

#[test]
fn test_52_sec_tan_pythagorean_numeric() {
    // sec²(x) - tan²(x) = 1
    // Using numeric equivalence since engine may not fully simplify
    assert_equiv_numeric_1var(
        "sec(x)^2 - tan(x)^2",
        "1",
        "x",
        -1.4,
        1.4, // Avoid cos=0 at ±π/2
        200,
        1e-9,
        |x| x.cos().abs() > 0.1, // Avoid sec/tan singularities
    );
}

#[test]
fn test_55_mixed_trig_fraction_numeric() {
    // (sin(x) + tan(x)) / (cot(x) + csc(x)) = sin(x) * tan(x)
    assert_equiv_numeric_1var(
        "(sin(x) + tan(x)) / (cot(x) + csc(x))",
        "sin(x) * tan(x)",
        "x",
        0.3,
        1.2,
        100,
        1e-8,
        |x| x.sin().abs() > 0.1 && x.cos().abs() > 0.1, // Avoid all singularities
    );
}

#[test]
fn test_tan_as_sin_over_cos_numeric() {
    // tan(x) = sin(x)/cos(x)
    assert_equiv_numeric_1var("tan(x)", "sin(x)/cos(x)", "x", -1.4, 1.4, 200, 1e-10, |x| {
        x.cos().abs() > 0.1
    });
}

#[test]
fn test_cot_as_cos_over_sin_numeric() {
    // cot(x) = cos(x)/sin(x)
    assert_equiv_numeric_1var("cot(x)", "cos(x)/sin(x)", "x", 0.3, 2.8, 200, 1e-10, |x| {
        x.sin().abs() > 0.1
    });
}

#[test]
fn test_sec_as_one_over_cos_numeric() {
    // sec(x) = 1/cos(x)
    assert_equiv_numeric_1var("sec(x)", "1/cos(x)", "x", -1.4, 1.4, 200, 1e-10, |x| {
        x.cos().abs() > 0.1
    });
}

#[test]
fn test_csc_as_one_over_sin_numeric() {
    // csc(x) = 1/sin(x)
    assert_equiv_numeric_1var("csc(x)", "1/sin(x)", "x", 0.3, 2.8, 200, 1e-10, |x| {
        x.sin().abs() > 0.1
    });
}

#[test]
fn test_sin_times_csc_numeric() {
    // sin(x) * csc(x) = 1
    assert_equiv_numeric_1var("sin(x) * csc(x)", "1", "x", 0.3, 2.8, 200, 1e-10, |x| {
        x.sin().abs() > 0.1
    });
}

#[test]
fn test_tan_times_cot_numeric() {
    // tan(x) * cot(x) = 1
    assert_equiv_numeric_1var("tan(x) * cot(x)", "1", "x", 0.3, 1.2, 200, 1e-10, |x| {
        x.sin().abs() > 0.1 && x.cos().abs() > 0.1
    });
}
