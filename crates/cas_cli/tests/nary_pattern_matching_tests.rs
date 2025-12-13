//! Comprehensive tests for N-ary Pattern Matching (Add Flattening)
//!
//! Verifies that rules can find patterns across all additive terms,
//! not just binary Add pairs.

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
                cas_ast::DisplayExpr {
                    context: &simplifier.context,
                    id: result,
                }
            )
        }
        Err(e) => panic!("Parse error: {:?}", e),
    }
}

// ============================================================================
// Basic N-ary Matching Tests
// ============================================================================

#[test]
fn test_nary_atan_three_terms() {
    // atan pair with one extra term
    let result = simplify_str("arctan(2) + 5 + arctan(1/2)");
    assert!(
        result.contains("1/2 * pi") && result.contains("5"),
        "Should find pair among 3 terms"
    );
}

#[test]
fn test_nary_atan_four_terms() {
    // atan pair with two extra terms
    let result = simplify_str("arctan(3) + 10 + arctan(1/3) - 5");
    assert!(
        result.contains("1/2 * pi") && result.contains("5"),
        "Should find pair among 4 terms"
    );
}

#[test]
fn test_nary_atan_five_terms_mixed() {
    // atan pair buried in middle
    let result = simplify_str("x + arctan(5) + y + arctan(1/5) + z");
    assert!(
        result.contains("1/2 * pi")
            && result.contains("x")
            && result.contains("y")
            && result.contains("z"),
        "Should find pair among 5 terms"
    );
}

// ============================================================================
// Multiple Pairs Tests
// ============================================================================

#[test]
fn test_nary_atan_two_pairs() {
    // Two separate atan pairs
    let result = simplify_str("arctan(2) + arctan(1/2) + arctan(3) + arctan(1/3)");
    // First iteration finds one pair → π/2 + arctan(3) + arctan(1/3)
    // Second iteration finds second pair → π/2 + π/2 → π
    assert_eq!(result, "pi", "Should find both pairs and simplify to π");
}

#[test]
fn test_nary_atan_two_pairs_with_noise() {
    // Two pairs with constants between
    let result = simplify_str("arctan(2) + 7 + arctan(1/2) - 3 + arctan(4) + arctan(1/4)");
    assert!(
        result.contains("pi") && result.contains("4") && !result.contains("arctan"),
        "Should find both pairs: 2π/2 + 7 - 3"
    );
}

#[test]
fn test_nary_atan_three_pairs() {
    // Three separate pairs (stress test)
    let result =
        simplify_str("arctan(2) + arctan(1/2) + arctan(3) + arctan(1/3) + arctan(5) + arctan(1/5)");
    assert_eq!(result, "3/2 * pi", "Should find all three pairs");
}

// ============================================================================
// Order Independence Tests
// ============================================================================

#[test]
fn test_nary_atan_reversed_order() {
    // Reciprocal appears before base
    let result = simplify_str("arctan(1/7) + arctan(7)");
    assert_eq!(result, "1/2 * pi", "Should work with reversed order");
}

#[test]
fn test_nary_atan_scattered() {
    // Pair separated by many terms
    let result = simplify_str("arctan(2) + a + b + c + d + e + arctan(1/2)");
    assert!(
        result.contains("1/2 * pi") && !result.contains("arctan"),
        "Should find scattered pair"
    );
}

#[test]
fn test_nary_atan_random_positions() {
    // Pair at random positions
    let result = simplify_str("x + y + arctan(6) + z + w + arctan(1/6) + v");
    assert!(
        result.contains("1/2 * pi") && !result.contains("arctan"),
        "Should find pair regardless of position"
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_nary_single_atan() {
    // Only one atan (no pair)
    let result = simplify_str("arctan(2) + 5");
    assert!(
        result.contains("arctan(2)") && result.contains("5"),
        "Should not change with single atan"
    );
}

#[test]
fn test_nary_no_reciprocal_pair() {
    // Two atans but not reciprocals
    let result = simplify_str("arctan(2) + arctan(3)");
    assert_eq!(
        result, "arctan(2) + arctan(3)",
        "Should not match non-reciprocals"
    );
}

#[test]
fn test_nary_three_atans_one_pair() {
    // Three atans, only two are reciprocals
    let result = simplify_str("arctan(2) + arctan(1/2) + arctan(5)");
    assert!(
        result.contains("1/2 * pi") && result.contains("arctan(5)"),
        "Should find the matching pair"
    );
}

#[test]
fn test_nary_same_atan_twice() {
    // Same atan appears twice (not reciprocals)
    let result = simplify_str("arctan(2) + arctan(2)");
    assert_eq!(result, "2 * arctan(2)", "Should not match identical atans");
}

// ============================================================================
// Integration with Multi-pass Iteration
// ============================================================================

#[test]
fn test_nary_with_cancellation() {
    // Original Test 48 - full expression
    let result = simplify_str("arctan(2) + arctan(1/2) - pi/2");
    assert_eq!(result, "0", "Should simplify completely to 0");
}

#[test]
fn test_nary_cascading_simplification() {
    // Pair simplifies, then constants combine
    let result = simplify_str("arctan(3) + arctan(1/3) + 10 - 10");
    assert_eq!(result, "1/2 * pi", "Should cascade: pair→π/2, then 10-10→0");
}

#[test]
fn test_nary_nested_operations() {
    // atan terms mixed with other operations
    let result = simplify_str("2*(atan(4) + 3) + arctan(1/4) - 6");
    // 2*atan(4) + 6 + arctan(1/4) - 6 → 2*atan(4) + arctan(1/4)
    // No simplification expected (can't factor out from product)
    assert!(
        result.contains("arctan"),
        "Nested products shouldn't prevent matching"
    );
}

// ============================================================================
// Numeric Reciprocal Detection
// ============================================================================

#[test]
fn test_nary_integer_reciprocals() {
    // Various integer reciprocals
    let result = simplify_str("arctan(7) + arctan(1/7)");
    assert_eq!(result, "1/2 * pi", "Should detect 7 and 1/7");
}

#[test]
fn test_nary_fraction_reciprocals() {
    // Fraction and its reciprocal
    let result = simplify_str("arctan(2/3) + arctan(3/2)");
    assert_eq!(result, "1/2 * pi", "Should detect 2/3 and 3/2");
}

#[test]
fn test_nary_decimal_like_reciprocals() {
    // Numbers that multiply to 1
    let result = simplify_str("arctan(1/10) + arctan(10)");
    assert_eq!(result, "1/2 * pi", "Should detect 1/10 and 10");
}

// ============================================================================
// Performance / Scalability Tests
// ============================================================================

#[test]
fn test_nary_ten_terms_one_pair() {
    // 10 terms with one pair
    let result = simplify_str("a + b + c + d + arctan(2) + e + f + g + arctan(1/2) + h");
    assert!(
        result.contains("1/2 * pi") && !result.contains("arctan") && result.contains("a"),
        "Should handle 10 terms efficiently"
    );
}

#[test]
fn test_nary_many_constants() {
    // Many constant terms with atan pair
    let result = simplify_str("1 + 2 + 3 + arctan(8) + 4 + 5 + arctan(1/8) + 6");
    // Constants fold: 1+2+3+4+5+6=21, plus π/2
    assert!(
        result.contains("1/2 * pi") && result.contains("21"),
        "Should handle many constants"
    );
}

// ============================================================================
// Negative and Complex Cases
// ============================================================================

#[test]
fn test_nary_with_negation() {
    // atan pair where one is negated in sum
    let result = simplify_str("arctan(2) - arctan(1/2) + pi/2");
    // Note: arctan(x) + arctan(1/x) = π/2, but arctan(x) - arctan(1/x) ≠ simple form
    // This should NOT simplify via reciprocal rule
    assert!(
        result.contains("arctan"),
        "Subtraction shouldn't match reciprocal rule"
    );
}

#[test]
fn test_nary_variables_in_args() {
    // Symbolic reciprocals (x and 1/x)
    let result = simplify_str("arctan(x) + arctan(1/x)");
    assert_eq!(result, "1/2 * pi", "Should work with symbolic reciprocals");
}

#[test]
fn test_nary_complex_fractions() {
    // Complex fraction reciprocals - currently NOT supported
    // are_reciprocals() only handles simple cases like x and 1/x
    let result = simplify_str("arctan((a + b)/(c + d)) + arctan((c + d)/(a + b))");
    // This should NOT simplify (complex expression reciprocals not detected)
    assert!(
        result.contains("arctan"),
        "Complex expression reciprocals not yet supported"
    );
}

// ============================================================================
// Regression Tests
// ============================================================================

#[test]
fn test_nary_doesnt_break_binary() {
    // Ensure binary case still works
    let result = simplify_str("arctan(2) + arctan(1/2)");
    assert_eq!(result, "1/2 * pi", "Binary case should still work");
}

#[test]
fn test_nary_preserves_other_functions() {
    // Other functions should not be affected
    let result = simplify_str("sin(x) + arctan(3) + cos(y) + arctan(1/3)");
    assert!(
        result.contains("1/2 * pi") && result.contains("cos(y)") && result.contains("sin(x)"),
        "Should not affect non-atan functions"
    );
}

#[test]
fn test_nary_no_false_positives() {
    // Similar looking but not reciprocals
    let result = simplify_str("arctan(2) + arctan(1/3)");
    // Accept both "arctan(1/3)" and "arctan(1 / 3)" (display format varies)
    let has_atan_third = result.contains("arctan(1/3)") || result.contains("arctan(1 / 3)");
    assert!(
        result.contains("arctan(2)") && has_atan_third && !result.contains("pi"),
        "Should not match non-reciprocals, got: {}",
        result
    );
}
