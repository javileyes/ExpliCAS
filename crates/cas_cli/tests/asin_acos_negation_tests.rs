#![allow(clippy::format_in_format_args)]
//! Tests for InverseTrigSumRule negation pattern
//! Verifies that asin + acos AND -asin - acos both work

use cas_ast::{DisplayExpr, Expr};
use cas_engine::Simplifier;

fn simplify_str(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let mut ctx = simplifier.context.clone();

    match cas_parser::parse(input, &mut ctx) {
        Ok(expr) => {
            simplifier.context = ctx;
            let (result, _) = simplifier.simplify(expr);
            format!(
                "{}",
                cas_ast::DisplayExpr {
                    context: &simplifier.context,
                    id: result,
                }
            )
        }
        Err(e) => panic!("Parse error: {:?}", e),
    }
}

/// Assert that `expr` simplifies to a value algebraically equivalent to `expected`.
/// Uses semantic equivalence: simplify(expr - expected) should equal 0.
fn assert_simplify_equiv(expr: &str, expected: &str, msg: &str) {
    let mut simplifier = Simplifier::with_default_rules();

    let e = cas_parser::parse(expr, &mut simplifier.context).expect("Failed to parse expr");
    let ex =
        cas_parser::parse(expected, &mut simplifier.context).expect("Failed to parse expected");

    // Simplify both
    let (se, _) = simplifier.simplify(e);
    let (sx, _) = simplifier.simplify(ex);

    // Compute diff = se - sx
    let diff = simplifier.context.add(Expr::Sub(se, sx));
    let (diff_simplified, _) = simplifier.simplify(diff);

    // Check if diff is zero
    let diff_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: diff_simplified,
        }
    );

    assert!(
        diff_str == "0",
        "{}\n  input: {}\n  expected equiv to: {}\n  got: {}\n  diff simplified to: {} (should be 0)",
        msg,
        expr,
        expected,
        format!("{}", DisplayExpr { context: &simplifier.context, id: se }),
        diff_str
    );
}

// ============ Positive Pairs ============

#[test]
fn test_asin_acos_positive_basic() {
    let result = simplify_str("asin(x) + acos(x)");
    assert_eq!(result, "1/2 * pi", "asin(x) + acos(x) should equal π/2");
}

#[test]
fn test_arcsin_arccos_positive_basic() {
    let result = simplify_str("arcsin(x) + arccos(x)");
    assert_eq!(result, "1/2 * pi", "arcsin(x) + arccos(x) should equal π/2");
}

#[test]
fn test_asin_acos_with_extra_terms() {
    // Semantic equivalence: result should equal y + z + π/2
    assert_simplify_equiv(
        "asin(x) + y + acos(x) + z",
        "y + z + pi/2",
        "Should simplify to y + z + π/2",
    );
}

// ============ Negated Pairs ============

#[test]
fn test_asin_acos_negated_basic() {
    let result = simplify_str("-asin(x) - acos(x)");
    assert!(
        result.contains("-") && result.contains("1/2 * pi"),
        "Should equal -π/2, got: {}",
        result
    );
}

#[test]
fn test_arcsin_arccos_negated_basic() {
    let result = simplify_str("-arcsin(x) - arccos(x)");
    assert!(
        result.contains("-") && result.contains("1/2 * pi"),
        "Should equal -π/2, got: {}",
        result
    );
}

#[test]
fn test_asin_acos_negated_with_variable() {
    // Semantic equivalence: result should equal y - π/2
    assert_simplify_equiv(
        "-asin(x) - acos(x) + y",
        "y - pi/2",
        "Should simplify to y - π/2",
    );
}

#[test]
fn test_asin_acos_negated_scattered() {
    // Semantic equivalence: result should equal a + b + c - π/2
    assert_simplify_equiv(
        "a - asin(x) + b - acos(x) + c",
        "a + b + c - pi/2",
        "Should simplify to a + b + c - π/2",
    );
}

// ============ Mixed/Edge Cases ============

#[test]
fn test_asin_acos_mixed_signs_no_match() {
    // After canonicalization, should be arcsin - arccos (both canonicalized but still mixed signs)
    let result = simplify_str("asin(x) - acos(x)");
    assert!(
        result.contains("arcsin") && result.contains("arccos"),
        "Mixed signs should not simplify but should be canonicalized, got: {}",
        result
    );
}

#[test]
fn test_asin_acos_both_pairs() {
    // One positive + one negated pair = 0
    let result = simplify_str("asin(x) + acos(x) - asin(y) - acos(y)");
    assert_eq!(result, "0", "π/2 - π/2 should equal 0");
}

#[test]
fn test_asin_acos_cancellation() {
    // Negated pair cancels with positive π/2
    let result = simplify_str("-asin(x) - acos(x) + pi/2");
    assert_eq!(result, "0", "Should cancel to 0");
}

// ============ N-ary Robustness ============

#[test]
fn test_asin_acos_nary_positive() {
    // Semantic equivalence: result should equal a + b + c + π/2
    assert_simplify_equiv(
        "a + asin(x) + b + acos(x) + c",
        "a + b + c + pi/2",
        "N-ary should find positive pair",
    );
}

#[test]
fn test_asin_acos_nary_negated() {
    // Semantic equivalence: result should equal a + b + c - π/2
    assert_simplify_equiv(
        "a - asin(x) + b - acos(x) + c",
        "a + b + c - pi/2",
        "N-ary should find negated pair",
    );
}

// ============ Consistency Tests ============

#[test]
fn test_asin_acos_generalization_consistency() {
    // Same result whether using asin or arcsin
    let result1 = simplify_str("-asin(x) - acos(x)");
    let result2 = simplify_str("-arcsin(x) - arccos(x)");
    assert_eq!(
        result1, result2,
        "Should be consistent across function name variants"
    );
}
