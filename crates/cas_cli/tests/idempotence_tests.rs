//! Tests for idempotence of rationalization rules.
//!
//! Ensures simplify(simplify(x)) == simplify(x) for rationalized expressions.

use cas_engine::Simplifier;
use cas_parser::parse;

fn simplify_expr(input: &str) -> String {
    let mut s = Simplifier::new();
    let expr = parse(input, &mut s.context).expect("parse failed");
    let (simplified, _) = s.simplify(expr);
    format!(
        "{}",
        cas_ast::DisplayExpr {
            context: &s.context,
            id: simplified
        }
    )
}

fn check_idempotence(input: &str) {
    let once = simplify_expr(input);
    let twice = simplify_expr(&once);
    assert_eq!(
        once, twice,
        "Not idempotent!\nInput: {}\nOnce: {}\nTwice: {}",
        input, once, twice
    );
}

#[test]
fn test_idempotence_single_surd_denominator() {
    // Level 0: 1/√2 → √2/2
    check_idempotence("1/sqrt(2)");
    check_idempotence("a/sqrt(2)");
    check_idempotence("1/(2*sqrt(2))");
}

#[test]
fn test_idempotence_binomial_surd_denominator() {
    // Level 1: 1/(1+√2) → √2-1
    check_idempotence("1/(1 + sqrt(2))");
    check_idempotence("x/(1 + sqrt(2))");
    check_idempotence("1/(3 - 2*sqrt(5))");
}

#[test]
fn test_idempotence_multisurd_denominator() {
    // Level 2: 1/(1+√2+√3) → generalized rationalization
    // Now passes: term order is stable after canonicalization fixes
    check_idempotence("1/(1 + sqrt(2) + sqrt(3))");
}

#[test]
fn test_level1_does_not_apply_to_multisurd() {
    // Ensure Level 1 doesn't apply when denominator has multiple surds
    // (should go to Level 2 instead)
    let result = simplify_expr("1/((1 + sqrt(2)) + sqrt(3))");
    // Should NOT have the shape of binomial conjugate alone
    // Level 2 should rationalize it completely
    assert!(
        !result.contains("(1 + sqrt(2))"),
        "Level 1 should not partially apply to multi-surd: {}",
        result
    );
}
