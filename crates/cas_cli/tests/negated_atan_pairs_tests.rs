//! Test for negated atan pairs - BEATS SYMPY! 🏆

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

#[test]
fn test_negated_atan_pair_basic() {
    // -atan(x) - arctan(1/x) = -π/2
    let result = simplify_str("-atan(2) - arctan(1/2)");
    assert!(
        result.contains("1/2 * pi") && result.contains("-"),
        "Should simplify to -π/2, got: {}",
        result
    );
}

#[test]
fn test_negated_atan_pair_with_variable() {
    // The case that Sympy can't do!
    let result = simplify_str("-atan(1/2) - arctan(2) + x");
    assert!(
        result.contains("1/2 * pi") && result.contains("x") && !result.contains("arctan"),
        "Should simplify to x - π/2, got: {}",
        result
    );
}

#[test]
fn test_negated_atan_pair_different_order() {
    let result = simplify_str("-atan(3) + y - arctan(1/3)");
    assert!(
        result.contains("1/2 * pi") && !result.contains("arctan"),
        "Should find negated pair, got: {}",
        result
    );
}

#[test]
fn test_mixed_positive_and_negative_atan() {
    // Positive pair + separate negated terms
    let result = simplify_str("arctan(2) + arctan(1/2) - arctan(5) - arctan(1/5)");
    // First pair: π/2, second (negated) pair: -π/2 → π/2 - π/2 = 0
    assert_eq!(result, "0", "Should simplify to 0, got: {}", result);
}

#[test]
fn test_partially_negated_no_match() {
    // Only one negated, other positive - should NOT match
    let result = simplify_str("-atan(2) + arctan(1/2)");
    assert!(
        result.contains("arctan(2)") && result.contains("arctan(1/2)"),
        "Should NOT match partial negation, got: {}",
        result
    );
}
