//! Policy A+ Specification Tests
//!
//! These tests verify the contract between simplify() and expand():
//! - simplify() does NOT expand binomial×binomial products
//! - simplify() DOES apply structural reductions (e.g., Difference of Squares)
//! - expand() aggressively expands products
//!
//! See POLICY.md for full documentation.

use cas_ast::DisplayExpr;
use cas_engine::Simplifier;
use cas_parser::parse;

fn create_simplifier() -> Simplifier {
    let mut s = Simplifier::new();
    s.register_default_rules();
    s
}

/// Helper to check for power notation (handles both x^2 and x^(2))
fn contains_power(output: &str, base: &str, exp: &str) -> bool {
    output.contains(&format!("{}^{}", base, exp)) || output.contains(&format!("{}^({})", base, exp))
}

// ============================================================================
// SIMPLIFY: Binomial×Binomial Preservation
// ============================================================================

#[test]
fn test_simplify_preserves_binomial_product() {
    // (x+1)*(x+2) should NOT expand to x²+3x+2
    let mut s = create_simplifier();
    let expr = parse("(x+1)*(x+2)", &mut s.context).unwrap();
    let (result, _) = s.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: result
        }
    );

    // Should NOT contain x^2 (polynomial form)
    assert!(
        !contains_power(&output, "x", "2"),
        "simplify should preserve (x+1)(x+2), got: {}",
        output
    );
}

#[test]
fn test_simplify_preserves_binomial_difference() {
    // (x-1)*(x-2) should NOT expand
    let mut s = create_simplifier();
    let expr = parse("(x-1)*(x-2)", &mut s.context).unwrap();
    let (result, _) = s.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: result
        }
    );

    assert!(
        !contains_power(&output, "x", "2"),
        "simplify should preserve (x-1)(x-2), got: {}",
        output
    );
}

// ============================================================================
// SIMPLIFY: Structural Reductions DO Apply
// ============================================================================

#[test]
fn test_simplify_applies_difference_of_squares_xy() {
    // (x-y)*(x+y) → x² - y²
    let mut s = create_simplifier();
    let expr = parse("(x-y)*(x+y)", &mut s.context).unwrap();
    let (result, _) = s.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: result
        }
    );

    // Should be expanded via Difference of Squares
    assert!(
        contains_power(&output, "x", "2") || contains_power(&output, "y", "2"),
        "simplify should reduce (x-y)(x+y) to x²-y², got: {}",
        output
    );
    // Should NOT remain as product
    assert!(
        !output.contains("(x - y)") && !output.contains("(x + y)"),
        "should not preserve product form, got: {}",
        output
    );
}

#[test]
fn test_simplify_applies_difference_of_squares_numeric() {
    // (x-1)*(x+1) → x² - 1
    let mut s = create_simplifier();
    let expr = parse("(x-1)*(x+1)", &mut s.context).unwrap();
    let (result, _) = s.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: result
        }
    );

    assert!(
        contains_power(&output, "x", "2"),
        "simplify should reduce (x-1)(x+1), got: {}",
        output
    );
}

// ============================================================================
// EXPAND: Aggressive Expansion
// ============================================================================

#[test]
fn test_expand_expands_binomial_product() {
    // expand((x+1)*(x+2)) → x² + 3x + 2
    let mut s = create_simplifier();
    let expr = parse("expand((x+1)*(x+2))", &mut s.context).unwrap();
    let (result, _) = s.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: result
        }
    );

    assert!(
        contains_power(&output, "x", "2"),
        "expand should produce polynomial form, got: {}",
        output
    );
    // Should not remain as product
    assert!(
        !output.contains("(1 + x)") && !output.contains("(2 + x)"),
        "expand should not preserve product form, got: {}",
        output
    );
}

#[test]
fn test_expand_expands_difference_of_squares() {
    // expand((x-1)*(x+1)) → x² - 1 (same as simplify, but via expansion not DoS)
    let mut s = create_simplifier();
    let expr = parse("expand((x-1)*(x+1))", &mut s.context).unwrap();
    let (result, _) = s.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: result
        }
    );

    assert!(
        contains_power(&output, "x", "2"),
        "expand should produce polynomial form, got: {}",
        output
    );
}

#[test]
fn test_expand_binomial_power() {
    // expand((x+1)^3) → x³ + 3x² + 3x + 1
    let mut s = create_simplifier();
    let expr = parse("expand((x+1)^3)", &mut s.context).unwrap();
    let (result, _) = s.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: result
        }
    );

    assert!(
        contains_power(&output, "x", "3"),
        "expand should produce polynomial form with x³, got: {}",
        output
    );
}

// ============================================================================
// IDEMPOTENCE
// ============================================================================

#[test]
fn test_simplify_idempotence() {
    // simplify(simplify(expr)) == simplify(expr)
    let mut s = create_simplifier();
    let expr = parse("(x+1)*(x+2) + 3*x", &mut s.context).unwrap();

    let (once, _) = s.simplify(expr);
    let (twice, _) = s.simplify(once);

    assert_eq!(once, twice, "simplify should be idempotent");
}

#[test]
fn test_expand_idempotence() {
    // expand(expand(expr)) == expand(expr)
    let mut s = create_simplifier();
    let expr = parse("expand((x+1)*(x+2))", &mut s.context).unwrap();

    let (once, _) = s.simplify(expr);
    // Wrap in expand again
    let expr2 = s
        .context
        .add(cas_ast::Expr::Function("expand".to_string(), vec![once]));
    let (twice, _) = s.simplify(expr2);

    assert_eq!(once, twice, "expand should be idempotent");
}
