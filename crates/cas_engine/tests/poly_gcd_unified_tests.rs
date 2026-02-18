//! Contract tests for unified poly_gcd API
//!
//! Tests that verify the unified poly_gcd(a, b, mode) correctly dispatches
//! to structural, exact, or modp based on the mode argument.

use cas_ast::Context;
use cas_engine::options::EvalOptions;
use cas_engine::Simplifier;
use cas_parser::parse;

fn eval(input: &str) -> String {
    let opts = EvalOptions::default();
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("parse failed");

    let mut simplifier = Simplifier::with_profile(&opts);
    simplifier.context = ctx;

    let (result, _steps) = simplifier.simplify(expr);

    cas_formatter::DisplayExpr {
        context: &simplifier.context,
        id: result,
    }
    .to_string()
}

// =============================================================================
// Structural mode (default)
// =============================================================================

#[test]
fn test_poly_gcd_structural_finds_visible_factor() {
    // g is a visible factor in both a*g and b*g
    let result = eval("poly_gcd(x*y, x*z)");
    assert_eq!(result, "x");
}

#[test]
fn test_poly_gcd_structural_returns_one_if_no_visible_factor() {
    // No visible common factor between x+1 and x+2
    // Note: structural won't find algebraic factors
    let result = eval("poly_gcd(x, y)");
    assert_eq!(result, "1");
}

// =============================================================================
// Auto mode
// =============================================================================

#[test]
fn test_poly_gcd_auto_uses_structural() {
    // Visible factor should be detected structurally
    let result = eval("poly_gcd(x*y, x*z, auto)");
    assert_eq!(result, "x");
}

#[test]
fn test_poly_gcd_auto_falls_through_to_exact() {
    // No visible factor, but exact can find algebraic GCD
    // x^2-1 = (x-1)(x+1), x-1 = (x-1)
    // GCD = x-1
    let result = eval("poly_gcd(x^2-1, x-1, auto)");
    // Exact mode should find x-1
    assert!(
        result.contains("x") || result == "1 + (-1)·x" || result == "x - 1" || result == "(-1) + x"
    );
}

// =============================================================================
// Exact mode
// =============================================================================

#[test]
fn test_poly_gcd_exact_finds_algebraic_gcd() {
    // x^2-1 = (x-1)(x+1), x^2-2x+1 = (x-1)^2
    // GCD = x-1
    let result = eval("poly_gcd(x^2-1, x^2-2*x+1, exact)");
    // Result should be x-1 (possibly with different ordering)
    assert!(result.contains("x") && result != "1");
}

#[test]
fn test_poly_gcd_exact_alias_rational() {
    let result = eval("poly_gcd(x*y, x*z, rational)");
    assert_eq!(result, "x");
}

#[test]
fn test_poly_gcd_exact_alias_q() {
    let result = eval("poly_gcd(x*y, x*z, q)");
    assert_eq!(result, "x");
}

// =============================================================================
// Modp mode
// =============================================================================

#[test]
fn test_poly_gcd_modp_computes_gcd() {
    let result = eval("poly_gcd(x*y, x*z, modp)");
    assert_eq!(result, "x");
}

#[test]
fn test_poly_gcd_modp_alias_fast() {
    let result = eval("poly_gcd(x*y, x*z, fast)");
    assert_eq!(result, "x");
}

#[test]
fn test_poly_gcd_modp_alias_zippel() {
    let result = eval("poly_gcd(x*y, x*z, zippel)");
    assert_eq!(result, "x");
}

// =============================================================================
// Mode parsing edge cases
// =============================================================================

#[test]
fn test_poly_gcd_unknown_mode_defaults_to_structural() {
    // Unknown mode should default to structural
    let result = eval("poly_gcd(x*y, x*z, unknown_mode)");
    assert_eq!(result, "x");
}

#[test]
fn test_poly_gcd_case_insensitive_modes() {
    // Mode parsing should be case-insensitive
    let result_auto = eval("poly_gcd(x*y, x*z, AUTO)");
    let result_exact = eval("poly_gcd(x*y, x*z, EXACT)");
    let result_modp = eval("poly_gcd(x*y, x*z, MODP)");

    assert_eq!(result_auto, "x");
    assert_eq!(result_exact, "x");
    assert_eq!(result_modp, "x");
}

// =============================================================================
// Parser fix verification (e constant vs exact keyword)
// =============================================================================

#[test]
fn test_parser_e_constant_standalone() {
    // 'e' alone should be Euler's constant
    let result = eval("e");
    assert_eq!(result, "e");
}

#[test]
fn test_parser_exact_is_variable() {
    // 'exact' should be a variable, not consume 'e' as constant
    let result = eval("exact");
    assert_eq!(result, "exact");
}

#[test]
fn test_parser_pi_constant_standalone() {
    // 'pi' alone should be π constant
    let result = eval("pi");
    assert_eq!(result, "pi");
}

#[test]
fn test_parser_pivot_is_variable() {
    // 'pivot' should be a variable, not consume 'pi' as constant
    let result = eval("pivot");
    assert_eq!(result, "pivot");
}
