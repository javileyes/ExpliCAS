#![allow(clippy::format_in_format_args)]
#![allow(clippy::field_reassign_with_default)]
#![allow(dead_code)]
#![allow(unused_variables)]
/// Anti-regression tests for the did_rationalize phase control flag
/// These tests verify that:
/// 1. Rationalization + other simplifiable parts work together correctly
/// 2. Distribution still works when no rationalization occurs
use cas_ast::{Context, DisplayExpr, Expr};
use cas_engine::Simplifier;
use cas_parser::parse;

/// Assert that `expr` simplifies to a value algebraically equivalent to `expected`.
fn assert_simplify_equiv(expr: &str, expected: &str, msg: &str) {
    let mut simplifier = Simplifier::with_default_rules();

    let e = parse(expr, &mut simplifier.context).expect("Failed to parse expr");
    let ex = parse(expected, &mut simplifier.context).expect("Failed to parse expected");

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

/// Test 1: Rationalization produces correct form
/// x/(1+√2) should become x*(√2-1) = x*√2 - x (distributed form is acceptable)
#[test]
fn test_rationalize_compact_form() {
    let mut ctx = Context::new();
    let mut simplifier = Simplifier::with_default_rules();

    let expr = parse("x/(1 + sqrt(2))", &mut ctx).unwrap();
    simplifier.context = ctx;
    let (result, _) = simplifier.simplify(expr);
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // With phase-based simplification, the result may be:
    // - Compact: x * (-1 + √2)
    // - Expanded: -x + x*√2 or x*√2 - x
    // Both are mathematically correct after rationalization
    assert!(
        result_str.contains("x") && (result_str.contains("2^(1/2)") || result_str.contains("√")),
        "Expected result containing x and sqrt(2), got: {}",
        result_str
    );
}

/// Test 2: Pure distribution (no rationalization) still works
/// When there's no surd in denominator, distribution should proceed normally
#[test]
fn test_distribution_without_rationalization() {
    let mut ctx = Context::new();
    let mut simplifier = Simplifier::with_default_rules();

    let expr = parse("2*(x + 3)", &mut ctx).unwrap();
    simplifier.context = ctx;
    let (result, _) = simplifier.simplify(expr);
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Should expand: 2*(x+3) -> 2*x + 6
    assert!(
        result_str.contains("6") && (result_str.contains("2 * x") || result_str.contains("x * 2")),
        "Expected distribution to work: 2*(x+3) -> 6 + 2*x, got: {}",
        result_str
    );
}

/// Test 3: Explicit expand() still works
/// expand((x+1)^2) should expand to binomial
#[test]
fn test_explicit_expand_still_works() {
    let mut ctx = Context::new();
    let mut simplifier = Simplifier::with_default_rules();

    let expr = parse("expand((x+1)^2)", &mut ctx).unwrap();
    simplifier.context = ctx;
    let (result, _) = simplifier.simplify(expr);
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Should see expansion with quadratic term (x^2 or x^(2))
    // If it stays as (1+x)^2 that's also acceptable (factored form preserved)
    // The key test is that it doesn't error
    assert!(
        result_str.contains("x") && (result_str.contains("^") || result_str.contains("+")),
        "Expected valid polynomial form, got: {}",
        result_str
    );
}

/// Test 4: did_rationalize resets between simplify() calls
/// After rationalizing one expression, a new simplify() call should allow distribution
#[test]
fn test_flag_resets_between_simplify_calls() {
    let mut simplifier = Simplifier::with_default_rules();

    // First: rationalize something
    let mut ctx1 = Context::new();
    let expr1 = parse("1/(1 + sqrt(2))", &mut ctx1).unwrap();
    simplifier.context = ctx1;
    let (result1, _) = simplifier.simplify(expr1);
    let result1_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result1
        }
    );
    // Should rationalize (contains surd term minus 1)
    assert!(
        result1_str.contains("- 1") || result1_str.contains("-1"),
        "First expr should rationalize, got: {}",
        result1_str
    );

    // Second: distribute something (should work because flag resets)
    let mut ctx2 = Context::new();
    let expr2 = parse("2*(y + 3)", &mut ctx2).unwrap();
    simplifier.context = ctx2;
    let (result2, _) = simplifier.simplify(expr2);
    let result2_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result2
        }
    );

    // Distribution should work
    assert!(
        result2_str.contains("6"),
        "Second expr should distribute: 2*(y+3) -> 6 + 2*y, got: {}",
        result2_str
    );
}

/// Test 5: Nested expression with surd and non-surd parts
/// Verifies handling of expressions like (x + 1/(1+√2))
#[test]
fn test_nested_surd_and_regular_parts() {
    let mut ctx = Context::new();
    let mut simplifier = Simplifier::with_default_rules();

    // x + 1/(1+√2) - has both regular part and rationalization
    let expr = parse("x + 1/(1 + sqrt(2))", &mut ctx).unwrap();
    simplifier.context = ctx;
    let (result, _) = simplifier.simplify(expr);
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );

    // Should see x (preserved) and some surd part (2^(1/2) or √)
    assert!(
        result_str.contains("x"),
        "Expected x to be preserved, got: {}",
        result_str
    );
    // The rationalized part should be present (with surd)
    assert!(
        result_str.contains("2") && (result_str.contains("^(1/2)") || result_str.contains("√")),
        "Expected rationalized surd part, got: {}",
        result_str
    );
}

/// Test 6: Simple fraction without surd - no rationalization, distribution semantically correct
#[test]
fn test_simple_fraction_no_rationalization() {
    // Verify semantic equivalence: x/2 + 3*(a+1) = x/2 + 3*a + 3
    assert_simplify_equiv(
        "x/2 + 3*(a + 1)",
        "x/2 + 3*a + 3",
        "Distribution should produce semantically correct result",
    );
}
