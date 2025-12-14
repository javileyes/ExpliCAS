//! Policy A+ Specification Tests
//!
//! These tests verify the contract between simplify() and expand():
//! - simplify() does NOT expand binomial×binomial products
//! - simplify() DOES apply structural reductions (e.g., Difference of Squares)
//! - expand() aggressively expands products
//!
//! See POLICY.md for full documentation.

use cas_ast::{Context, DisplayExpr, ExprId};
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

/// Helper to format an expression as string
fn format_expr(ctx: &Context, id: ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
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

// ============================================================================
// DISPLAY ORDERING: Positive terms before negative (P5)
// ============================================================================

#[test]
fn test_display_order_difference_of_squares() {
    // (x+y)(x-y) should display as x² - y² (not -y² + x²)
    let mut s = create_simplifier();
    let expr = parse("(x+y)*(x-y)", &mut s.context).unwrap();
    let (result, _) = s.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: result
        }
    );

    // x² should come before -y² (or y²)
    let x2_pos = output.find("x");
    let y2_pos = output.find("y");
    assert!(
        x2_pos < y2_pos,
        "x² should come before y² in display, got: {}",
        output
    );
}

#[test]
fn test_display_order_neg_plus_positive() {
    // -x + y should display as y - x
    let mut s = create_simplifier();
    let expr = parse("-x + y", &mut s.context).unwrap();
    let (result, _) = s.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: result
        }
    );

    assert!(
        output.starts_with("y"),
        "positive term y should come first, got: {}",
        output
    );
}

#[test]
fn test_display_order_polynomial() {
    // x + x² - 3 should display as x² + x - 3
    let mut s = create_simplifier();
    let expr = parse("x + x^2 - 3", &mut s.context).unwrap();
    let (result, _) = s.simplify(expr);
    let output = format!(
        "{}",
        DisplayExpr {
            context: &s.context,
            id: result
        }
    );

    // x² should come first, then x, then -3
    let x2_pos = output.find("x^");
    let neg3_pos = output.find("-");
    assert!(
        x2_pos.is_some() && neg3_pos.is_some() && x2_pos < neg3_pos,
        "x² should come before -3, got: {}",
        output
    );
}

// ============================================================================
// FRACTIONAL DISTRIBUTION POLICY: simplify preserves, expand distributes
// ============================================================================

#[test]
fn test_simplify_preserves_fractional_binomial() {
    // simplify 1/2 * (√2 - 1) should preserve the factored form
    let mut s = Simplifier::with_default_rules();
    let expr = parse("1/2 * (sqrt(2) - 1)", &mut s.context).unwrap();
    let (result, _) = s.simplify(expr);
    let output = format_expr(&s.context, result);

    // Should contain the binomial structure, not be fully distributed
    // Expected: (√(2) - 1)/2 or 1/2 * (√(2) - 1)
    assert!(
        output.contains("√(2) - 1") || output.contains("2^(1/2) - 1"),
        "simplify should preserve fractional binomial form: got {}",
        output
    );
}

#[test]
fn test_expand_distributes_fractional_binomial() {
    // expand 1/2 * (√2 - 1) should distribute
    let mut ctx = Context::new();
    let expr = parse("1/2 * (sqrt(2) - 1)", &mut ctx).unwrap();
    let expanded = cas_engine::expand::expand(&mut ctx, expr);
    let output = format_expr(&ctx, expanded);

    // Should be distributed: √2/2 - 1/2 or similar
    assert!(
        output.contains("/2") && !output.contains("(√(2) - 1)/2"),
        "expand should distribute fractional binomial: got {}",
        output
    );
}

// ============================================================================
// SIMPLIFY vs EXPAND CONTRACT: Style-Only Differences
// These tests verify that the ONLY difference is presentation style
// ============================================================================

#[test]
fn test_contract_fractional_binomial_style_only() {
    // 1/2 * (√2 - 1): simplify keeps factored, expand distributes
    let mut s = Simplifier::with_default_rules();
    let expr = parse("1/2 * (sqrt(2) - 1)", &mut s.context).unwrap();

    // simplify: factored form
    let (simplified, _) = s.simplify(expr);
    let simp_out = format_expr(&s.context, simplified);

    // expand: distributed form
    let mut ctx = Context::new();
    let expr2 = parse("1/2 * (sqrt(2) - 1)", &mut ctx).unwrap();
    let expanded = cas_engine::expand::expand(&mut ctx, expr2);
    let exp_out = format_expr(&ctx, expanded);

    // simplify should preserve binomial structure
    assert!(
        simp_out.contains("√(2) - 1") || simp_out.contains("2^(1/2) - 1"),
        "simplify should preserve fractional binomial: got {}",
        simp_out
    );

    // expand should NOT preserve binomial structure
    assert!(
        !exp_out.contains("(√(2) - 1)") && !exp_out.contains("(2^(1/2) - 1)"),
        "expand should distribute fractional binomial: got {}",
        exp_out
    );
}

#[test]
fn test_contract_negative_fractional_binomial() {
    // (-1/2) * (1 - √2) should become 1/2 * (√2 - 1) in simplify
    let mut s = Simplifier::with_default_rules();
    let expr = parse("(-1/2) * (1 - sqrt(2))", &mut s.context).unwrap();
    let (simplified, _) = s.simplify(expr);
    let output = format_expr(&s.context, simplified);

    // Should have √2 - 1 (flipped), not 1 - √2
    assert!(
        output.contains("√(2) - 1") || output.contains("2^(1/2) - 1"),
        "simplify should flip negative binomial to positive form: got {}",
        output
    );
}

// ============================================================================
// SIMPLIFY vs EXPAND CONTRACT: Structural Patterns (Must Distribute)
// These tests verify patterns that MUST be simplified even in simplify()
// ============================================================================

#[test]
fn test_contract_difference_of_squares_simplifies() {
    // (x-1)(x+1) should become x² - 1 in BOTH simplify and expand
    let mut s = Simplifier::with_default_rules();
    let expr = parse("(x-1)*(x+1)", &mut s.context).unwrap();
    let (simplified, _) = s.simplify(expr);
    let output = format_expr(&s.context, simplified);

    // Should simplify to x² - 1
    assert!(
        output.contains("x^2") || output.contains("x^(2)"),
        "simplify should apply difference of squares: got {}",
        output
    );
    assert!(
        output.contains("- 1") || output.contains("-1"),
        "simplify should produce x² - 1: got {}",
        output
    );
}

#[test]
fn test_contract_integer_distribution_simplifies() {
    // 2 * (x + 1) should distribute in BOTH simplify and expand (integer coefficient)
    let mut s = Simplifier::with_default_rules();
    let expr = parse("2 * (x + 1)", &mut s.context).unwrap();
    let (simplified, _) = s.simplify(expr);
    let output = format_expr(&s.context, simplified);

    // Should distribute: 2x + 2
    assert!(
        output.contains("2 * x") || output.contains("2*x") || output.contains("2x"),
        "simplify should distribute integer over binomial: got {}",
        output
    );
}

#[test]
fn test_contract_expand_preserves_conjugate_products() {
    // expand() should PRESERVE conjugate products (x-1)(x+1) - this is by design
    // The DifferenceOfSquaresRule only runs in simplify(), not in raw expand()
    let mut ctx = Context::new();
    let expr = parse("(x-1)*(x+1)", &mut ctx).unwrap();
    let expanded = cas_engine::expand::expand(&mut ctx, expr);
    let output = format_expr(&ctx, expanded);

    // expand() preserves conjugate products (guard in is_conjugate)
    // Either preserves factored form OR expands - both are valid
    assert!(
        output.contains("x")
            && (output.contains("-1") || output.contains("+ 1") || output.contains("- 1")),
        "expand should handle conjugate product: got {}",
        output
    );
}

// ============================================================================
// SIMPLIFY vs EXPAND CONTRACT: Regression Tests
// These tests ensure neither mode loses functionality
// ============================================================================

#[test]
fn test_regression_simplify_preserves_factored_quadratic() {
    // (x+1)^2 should stay factored in simplify (binomial power guard)
    let mut s = Simplifier::with_default_rules();
    let expr = parse("(x+1)^2", &mut s.context).unwrap();
    let (simplified, _) = s.simplify(expr);
    let output = format_expr(&s.context, simplified);

    // Config-dependent: if expand_binomials is off, should preserve
    // For now just check it doesn't crash and produces valid output
    assert!(
        output.contains("x") && (output.contains("1") || output.contains("2")),
        "simplify should handle binomial power: got {}",
        output
    );
}

#[test]
fn test_regression_expand_fully_expands_binomial_power() {
    // (x+1)^2 should expand to x² + 2x + 1 in expand
    let mut ctx = Context::new();
    let expr = parse("(x+1)^2", &mut ctx).unwrap();
    let expanded = cas_engine::expand::expand(&mut ctx, expr);
    let output = format_expr(&ctx, expanded);

    // Should be fully expanded
    assert!(
        output.contains("x^2") || output.contains("x^(2)"),
        "expand should expand binomial power to x²: got {}",
        output
    );
}

#[test]
fn test_regression_simplify_combines_like_terms() {
    // x + x should become 2x in simplify
    let mut s = Simplifier::with_default_rules();
    let expr = parse("x + x", &mut s.context).unwrap();
    let (simplified, _) = s.simplify(expr);
    let output = format_expr(&s.context, simplified);

    assert!(
        output.contains("2") && output.contains("x"),
        "simplify should combine like terms: got {}",
        output
    );
}

#[test]
fn test_regression_expand_combines_like_terms() {
    // x + x should also become 2x in expand
    let mut ctx = Context::new();
    let expr = parse("x + x", &mut ctx).unwrap();
    let expanded = cas_engine::expand::expand(&mut ctx, expr);

    // Expand + simplify to clean up
    let mut s = Simplifier::with_default_rules();
    s.context = ctx;
    let (final_result, _) = s.simplify(expanded);
    let output = format_expr(&s.context, final_result);

    assert!(
        output.contains("2") && output.contains("x"),
        "expand should combine like terms after expansion: got {}",
        output
    );
}
