//! Tests for InverseTrigSumRule negation pattern
//! Verifies that asin + acos AND -asin - acos both work

use cas_engine::Simplifier;
use cas_parser;

fn simplify_str(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let mut ctx = simplifier.context.clone();

    match cas_parser::parse(input, &mut ctx) {
        Ok(expr) => {
            simplifier.context = ctx;
            let (result, _) = simplifier.simplify(expr);
            format!(
                "{}",
                cas_ast::expression::DisplayExpr {
                    context: &simplifier.context,
                    id: result,
                }
            )
        }
        Err(e) => panic!("Parse error: {:?}", e),
    }
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
    let result = simplify_str("asin(x) + y + acos(x) + z");
    assert!(
        result.contains("1/2 * pi") && result.contains("y") && result.contains("z"),
        "Should find pair with extra terms, got: {}",
        result
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
    // The case that BEATS Sympy! (like atan)
    let result = simplify_str("-asin(x) - acos(x) + y");
    assert!(
        result.contains("y")
            && result.contains("-")
            && result.contains("1/2 * pi")
            && !result.contains("asin"),
        "Should simplify to y - π/2, got: {}",
        result
    );
}

#[test]
fn test_asin_acos_negated_scattered() {
    let result = simplify_str("a - asin(x) + b - acos(x) + c");
    assert!(
        result.contains("-") && result.contains("1/2 * pi") && !result.contains("asin"),
        "Should find scattered negated pair, got: {}",
        result
    );
}

// ============ Mixed/Edge Cases ============

#[test]
fn test_asin_acos_mixed_signs_no_match() {
    // Positive + negative should NOT match
    let result = simplify_str("asin(x) - acos(x)");
    assert!(
        result.contains("asin") && result.contains("acos"),
        "Mixed signs should not simplify, got: {}",
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
    // 5 terms with one pair
    let result = simplify_str("a + asin(x) + b + acos(x) + c");
    assert!(
        result.contains("1/2 * pi") && !result.contains("asin"),
        "N-ary should find positive pair, got: {}",
        result
    );
}

#[test]
fn test_asin_acos_nary_negated() {
    // 5 terms with one negated pair
    let result = simplify_str("a - asin(x) + b - acos(x) + c");
    assert!(
        result.contains("-") && result.contains("1/2 * pi") && !result.contains("asin"),
        "N-ary should find negated pair, got: {}",
        result
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
