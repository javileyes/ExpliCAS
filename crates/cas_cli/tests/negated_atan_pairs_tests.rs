#![allow(clippy::format_in_format_args)]
//! Test for negated atan pairs - BEATS SYMPY! 🏆

use cas_ast::Expr;
use cas_formatter::DisplayExpr;
use cas_solver::Simplifier;

fn simplify_str(input: &str) -> String {
    let mut simplifier = Simplifier::with_default_rules();
    let mut ctx = simplifier.context.clone();

    match cas_parser::parse(input, &mut ctx) {
        Ok(expr) => {
            simplifier.context = ctx;
            let (result, _) = simplifier.simplify(expr);
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: result,
                }
            )
        }
        Err(e) => panic!("Parse error: {:?}", e),
    }
}

/// Assert that `expr` simplifies to a value algebraically equivalent to `expected`.
fn assert_simplify_equiv(expr: &str, expected: &str, msg: &str) {
    let mut simplifier = Simplifier::with_default_rules();

    let e = cas_parser::parse(expr, &mut simplifier.context).expect("Failed to parse expr");
    let ex =
        cas_parser::parse(expected, &mut simplifier.context).expect("Failed to parse expected");

    let (se, _) = simplifier.simplify(e);
    let (sx, _) = simplifier.simplify(ex);

    let diff = simplifier.context.add(Expr::Sub(se, sx));
    let (diff_simplified, _) = simplifier.simplify(diff);

    let diff_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: diff_simplified,
        }
    );

    assert!(
        diff_str == "0",
        "{}\n  input: {}\n  expected equiv to: {}\n  got: {}\n  diff: {}",
        msg,
        expr,
        expected,
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: se
            }
        ),
        diff_str
    );
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
    // Semantic equivalence: x - π/2
    assert_simplify_equiv(
        "-atan(1/2) - arctan(2) + x",
        "x - pi/2",
        "Should simplify to x - π/2",
    );
}

#[test]
fn test_negated_atan_pair_different_order() {
    // Semantic equivalence: y - π/2
    assert_simplify_equiv(
        "-atan(3) + y - arctan(1/3)",
        "y - pi/2",
        "Should find negated pair",
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
    // Accept both "arctan(1/2)" and "arctan(1 / 2)" (display format varies)
    let has_atan_half = result.contains("arctan(1/2)") || result.contains("arctan(1 / 2)");
    assert!(
        result.contains("arctan(2)") && has_atan_half,
        "Should NOT match partial negation, got: {}",
        result
    );
}
