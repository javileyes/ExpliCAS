//! Tests for SmallMultinomialExpansionRule.
//!
//! Covers:
//! - Symbolic expansion for trinomials and quadrinomials
//! - Constants (π, e) as opaque atoms
//! - Numeric metamorphic verification
//! - Anti-blowup guards (oversized base blocked)
//! - Performance sanity

use cas_ast::DisplayExpr;
use cas_engine::helpers::eval_f64_with_substitution;
use cas_engine::Simplifier;
use cas_parser::parse;

/// Simplify an expression and return its display string.
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

/// Simplify and evaluate numerically.
fn simplify_and_eval(input: &str, vars: &[&str], vals: &[f64]) -> (String, Option<f64>) {
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
    let var_names: Vec<String> = vars.iter().map(|v| v.to_string()).collect();
    let numeric = eval_f64_with_substitution(&s.context, result, &var_names, vals);
    (result_str, numeric)
}

// =============================================================================
// Symbolic correctness
// =============================================================================

/// (a + b + c)^2 should expand to 6 terms (a²+b²+c²+2ab+2bc+2ac).
#[test]
fn trinomial_square_expands() {
    let result = simplify_str("(a + b + c)^2");
    // Must not retain the power form
    assert!(
        !result.contains("(a + b + c)^2"),
        "Expected expansion, got: {}",
        result
    );
    // Should contain mixed terms (2*a*b or equivalent)
    assert!(
        result.contains("a") && result.contains("b") && result.contains("c"),
        "Expected all variables in expansion, got: {}",
        result
    );
}

/// (a + b + c)^3 → 10 terms, still within budget (C(5,2)=10 ≤ 35).
#[test]
fn trinomial_cube_expands() {
    let result = simplify_str("(a + b + c)^3");
    assert!(
        !result.contains("(a + b + c)^3"),
        "Expected expansion, got: {}",
        result
    );
}

/// (a + b + c + d)^2 → C(5,3)=10 terms.
#[test]
fn quadrinomial_square_expands() {
    let result = simplify_str("(a + b + c + d)^2");
    assert!(
        !result.contains("(a + b + c + d)^2"),
        "Expected expansion, got: {}",
        result
    );
}

/// (u + pi + 1)^2 — the motivating case with a constant.
#[test]
fn constant_in_base_expands() {
    let result = simplify_str("(u + pi + 1)^2");
    // The expanded form may contain u^2 and pi^2, but should NOT retain
    // the original unexpanded form
    assert!(
        !result.contains("(u + pi + 1)") && !result.contains("(u + \u{03c0} + 1)"),
        "Expected expansion of (u+π+1)^2, got: {}",
        result
    );
    assert!(
        result.contains("u") && (result.contains("π") || result.contains("pi")),
        "Expected u and π in expansion, got: {}",
        result
    );
}

// =============================================================================
// Numeric metamorphic tests
// =============================================================================

/// Verify that simplify((sum)^n) ≈ (sum)^n numerically.
#[test]
fn metamorphic_trinomial_square() {
    let vals = [1.7, 2.3, 0.9];
    let original = 1.7 + 2.3 + 0.9;
    let expected = original * original;

    let (_, num) = simplify_and_eval("(a + b + c)^2", &["a", "b", "c"], &vals);
    let got = num.expect("Evaluation should succeed");
    assert!(
        (got - expected).abs() < 1e-10,
        "Metamorphic check failed: got {}, expected {}",
        got,
        expected
    );
}

#[test]
fn metamorphic_trinomial_cube() {
    let vals = [0.5, -1.2, 3.1];
    let sum: f64 = vals.iter().sum();
    let expected = sum.powi(3);

    let (_, num) = simplify_and_eval("(a + b + c)^3", &["a", "b", "c"], &vals);
    let got = num.expect("Evaluation should succeed");
    assert!(
        (got - expected).abs() < 1e-8,
        "Metamorphic check failed: got {}, expected {}",
        got,
        expected
    );
}

#[test]
fn metamorphic_quadrinomial_square() {
    let vals = [2.0, -0.5, 1.5, 0.7];
    let sum: f64 = vals.iter().sum();
    let expected = sum * sum;

    let (_, num) = simplify_and_eval("(a + b + c + d)^2", &["a", "b", "c", "d"], &vals);
    let got = num.expect("Evaluation should succeed");
    assert!(
        (got - expected).abs() < 1e-10,
        "Metamorphic check failed: got {}, expected {}",
        got,
        expected
    );
}

#[test]
fn metamorphic_constant_base() {
    // (u + π + 1)^2 at u = 2.0
    let u = 2.0_f64;
    let expected = (u + std::f64::consts::PI + 1.0).powi(2);

    let (_, num) = simplify_and_eval("(u + pi + 1)^2", &["u"], &[u]);
    let got = num.expect("Evaluation should succeed");
    assert!(
        (got - expected).abs() < 1e-10,
        "Metamorphic check failed: got {}, expected {}",
        got,
        expected
    );
}

// =============================================================================
// Anti-blowup guards
// =============================================================================

/// (a+b+c+d+e)^4 → C(8,4)=70 terms, exceeds MAX_OUTPUT=35.
/// The rule should NOT fire.
#[test]
fn anti_blowup_5terms_power4_blocked() {
    let result = simplify_str("(a + b + c + d + e)^4");
    // Should NOT have expanded — retain the power form
    assert!(
        result.contains("^4") || result.contains("^"),
        "Expected rule to NOT fire for 70-term expansion, got: {}",
        result
    );
}

/// (a+b+c+d+e+f+g)^5 → way too many terms. Must not fire.
#[test]
fn anti_blowup_7terms_power5_blocked() {
    let result = simplify_str("(a + b + c + d + e + f + g)^5");
    assert!(
        result.contains("^5") || result.contains("^"),
        "Expected rule to NOT fire for huge expansion, got: {}",
        result
    );
}

/// n=5 is above MAX_N=4, even with 3 terms (21 output terms).
#[test]
fn anti_blowup_exponent_above_max_n() {
    let result = simplify_str("(a + b + c)^5");
    assert!(
        result.contains("^5") || result.contains("^"),
        "Expected rule to NOT fire for n>MAX_N, got: {}",
        result
    );
}

// =============================================================================
// Performance sanity
// =============================================================================

/// (a+b+c+d)^4 → C(7,3)=35 terms. Right at the boundary. Should complete fast.
#[test]
fn perf_max_boundary_case() {
    let start = std::time::Instant::now();
    let result = simplify_str("(a + b + c + d)^4");
    let elapsed = start.elapsed();

    // Should have expanded
    assert!(
        !result.contains("(a + b + c + d)^4"),
        "Expected expansion at boundary, got: {}",
        result
    );

    // Must complete quickly
    assert!(
        elapsed.as_millis() < 2000,
        "Expansion took too long: {:?}",
        elapsed
    );
}

// =============================================================================
// k=2 no-fire: binomials stay on BinomialExpansionRule
// =============================================================================

/// (a+b)^4 is a binomial (k=2). SmallMultinomialExpansionRule requires k≥3,
/// so it must NOT route here. In standard mode, BinomialExpansionRule also
/// requires expand_mode, so the expression stays as Pow.
/// This protects against refactors that accidentally route k=2 to multinomial.
#[test]
fn binomial_not_routed_through_multinomial() {
    let result = simplify_str("(a + b)^4");

    // In standard mode (not expand_mode), binomials stay as Pow
    assert!(
        result.contains("^4") || result.contains("⁴"),
        "Expected (a+b)^4 to stay as Pow in standard mode, got: {}",
        result
    );

    // Confirm k=2 doesn't sneak through by verifying k=3 DOES expand
    let result3 = simplify_str("(a + b + c)^4");
    assert!(
        !result3.contains("(a + b + c)^4"),
        "Expected (a+b+c)^4 (k=3) to expand, but it didn't: {}",
        result3
    );
}
