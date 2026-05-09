//! Non-regression tests for Neg(Add/Sub) parenthesization in display
//!
//! These tests verify that expressions like `a - (b + c)` are rendered correctly
//! with parentheses around the subtracted sum, preventing sign errors.
//!
//! Bug fix: Ensures -(a + b) is rendered as "-(a + b)" not "-a + b"

use cas_ast::{hold, Context, Expr};
use cas_formatter::DisplayExpr;
use cas_parser::parse;

fn render(input: &str) -> String {
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("parse expression");
    format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr
        }
    )
}

// ============================================================================
// CLI Display Tests (DisplayExpr)
// ============================================================================

/// Test that Neg(Add) is displayed with parentheses: `a - (b + c)` not `a - b + c`
#[test]
fn test_neg_add_has_parentheses_displayexpr() {
    let mut ctx = Context::new();

    // Build: x + 1 - (3x + 2) as Add(Add(x, 1), Neg(Add(3x, 2)))
    let x = ctx.var("x");
    let one = ctx.num(1);
    let two = ctx.num(2);
    let three = ctx.num(3);

    let x_plus_1 = ctx.add(Expr::Add(x, one));
    let three_x = ctx.add(Expr::Mul(three, x));
    let three_x_plus_2 = ctx.add(Expr::Add(three_x, two));
    let neg_sum = ctx.add(Expr::Neg(three_x_plus_2));
    let expr = ctx.add(Expr::Add(x_plus_1, neg_sum));

    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr
        }
    );

    println!("Display result: {}", display);

    // Must have parentheses around the subtracted sum
    assert!(
        display.contains("(") && display.contains(")"),
        "Expected parentheses in '{}'. Should be 'x + 1 - (3 * x + 2)' or similar",
        display
    );

    // The subtracted sum should be in parentheses: look for "- (" pattern
    // which indicates the subtraction is followed by a parenthesized group
    assert!(
        display.contains("- ("),
        "Expected '- (' indicating parenthesized subtraction. Got: '{}'",
        display
    );
}

/// Test that Neg(Add) is displayed with parentheses: simple case
#[test]
fn test_neg_add_has_parentheses_simple() {
    let mut ctx = Context::new();

    // Build: a - (b + c)
    let a = ctx.var("a");
    let b = ctx.var("b");
    let c = ctx.var("c");

    let b_plus_c = ctx.add(Expr::Add(b, c));
    let neg_sum = ctx.add(Expr::Neg(b_plus_c));
    let expr = ctx.add(Expr::Add(a, neg_sum));

    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr
        }
    );

    println!("Simple case display: {}", display);

    // Must have parentheses
    assert!(
        display.contains("(b + c)") || display.contains("(c + b)"),
        "Expected '(b + c)' or '(c + b)' in '{}'. Neg(Add) must have parens.",
        display
    );
}

/// Test unary negation of sum: `-(a + b)` should have parentheses
#[test]
fn test_unary_neg_add_has_parentheses() {
    let mut ctx = Context::new();

    // Build: -(a + b)
    let a = ctx.var("a");
    let b = ctx.var("b");

    let a_plus_b = ctx.add(Expr::Add(a, b));
    let neg_sum = ctx.add(Expr::Neg(a_plus_b));

    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: neg_sum
        }
    );

    println!("Unary neg display: {}", display);

    // Must start with "-(" and include the sum
    assert!(
        display.starts_with("-("),
        "Unary neg of sum should start with '-('. Got: '{}'",
        display
    );
}

/// Internal hold barriers are transparent for display grouping decisions.
#[test]
fn test_unary_neg_internal_hold_add_has_parentheses() {
    let mut ctx = Context::new();

    let a = ctx.var("a");
    let b = ctx.var("b");

    let a_plus_b = ctx.add(Expr::Add(a, b));
    let held_sum = hold::wrap_hold(&mut ctx, a_plus_b);
    let neg_sum = ctx.add(Expr::Neg(held_sum));

    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: neg_sum
        }
    );

    assert!(
        display.starts_with("-("),
        "Internal hold around a negated sum must keep grouping. Got: '{}'",
        display
    );
    assert!(
        display.contains("a + b") || display.contains("b + a"),
        "Expected held sum inside parentheses. Got: '{}'",
        display
    );
}

#[test]
fn test_negative_unit_times_function_omits_unit_factor() {
    let mut ctx = Context::new();

    let neg_one = ctx.num(-1);
    let x = ctx.var("x");
    let two = ctx.num(2);
    let x_squared = ctx.add(Expr::Pow(x, two));
    let atanh = ctx.call("atanh", vec![x_squared]);
    let expr = ctx.add(Expr::Mul(neg_one, atanh));

    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr
        }
    );

    assert_eq!(display, "-atanh(x^2)");
}

#[test]
fn test_fraction_display_extracts_negative_numeric_factor_from_numerator() {
    let mut ctx = Context::new();

    let x = ctx.var("x");
    let neg_two = ctx.num(-2);
    let one = ctx.num(1);
    let four = ctx.num(4);
    let x_fourth = ctx.add(Expr::Pow(x, four));
    let numerator = ctx.add(Expr::Mul(x, neg_two));
    let denominator = ctx.add(Expr::Sub(one, x_fourth));
    let expr = ctx.add(Expr::Div(numerator, denominator));

    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr
        }
    );

    assert_eq!(display, "-2 * x / (1 - x^4)");
}

#[test]
fn test_negative_division_display_avoids_outer_parentheses() {
    let mut ctx = Context::new();

    let x = ctx.var("x");
    let two = ctx.num(2);
    let one = ctx.num(1);
    let four = ctx.num(4);
    let x_fourth = ctx.add(Expr::Pow(x, four));
    let numerator = ctx.add(Expr::Mul(x, two));
    let denominator = ctx.add(Expr::Sub(one, x_fourth));
    let fraction = ctx.add(Expr::Div(numerator, denominator));
    let expr = ctx.add(Expr::Neg(fraction));

    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr
        }
    );

    assert_eq!(display, "-2 * x / (1 - x^4)");
}

#[test]
fn test_negative_reciprocal_product_display_avoids_outer_parentheses() {
    let mut ctx = Context::new();

    let x = ctx.var("x");
    let two = ctx.num(2);
    let one = ctx.num(1);
    let neg_one = ctx.num(-1);
    let four = ctx.num(4);
    let x_fourth = ctx.add(Expr::Pow(x, four));
    let positive_numerator = ctx.add(Expr::Mul(x, two));
    let numerator = ctx.add(Expr::Neg(positive_numerator));
    let denominator = ctx.add(Expr::Sub(one, x_fourth));
    let reciprocal_denominator = ctx.add(Expr::Pow(denominator, neg_one));
    let expr = ctx.add(Expr::Mul(numerator, reciprocal_denominator));

    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr
        }
    );

    assert_eq!(display, "-2 * x/(1 - x^4)");
}

#[test]
fn test_fraction_display_avoids_double_parens_for_single_additive_denominator() {
    let mut ctx = Context::new();

    let x = ctx.var("x");
    let two = ctx.num(2);
    let one = ctx.num(1);
    let neg_one = ctx.num(-1);
    let four = ctx.num(4);
    let x_fourth = ctx.add(Expr::Pow(x, four));
    let numerator = ctx.add(Expr::Mul(x, two));
    let denominator = ctx.add(Expr::Sub(one, x_fourth));
    let reciprocal_denominator = ctx.add(Expr::Pow(denominator, neg_one));
    let expr = ctx.add(Expr::Mul(numerator, reciprocal_denominator));

    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr
        }
    );

    assert_eq!(display, "(x * 2)/(1 - x^4)");
}

#[test]
fn test_fraction_display_extracts_negative_numeric_denominator_sign() {
    let mut ctx = Context::new();

    let x = ctx.var("x");
    let neg_two = ctx.num(-2);
    let expr = ctx.add(Expr::Div(x, neg_two));

    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr
        }
    );

    assert_eq!(display, "-x / 2");
}

#[test]
fn test_fraction_display_preserves_negative_sign_for_inverse_trig_like_quotient() {
    let display = render("-2*(2*x^3+3*x^2+3*x+1)*sqrt(3)/sqrt(2-3*(x^2+x+1)^4)");

    assert_eq!(
        display,
        "-2 * sqrt(3) * (2 * x^3 + 3 * x^2 + 3 * x + 1) / sqrt(2 - 3 * (x^2 + x + 1)^4)"
    );
}

/// Internal hold barriers on the RHS of subtraction still need parentheses.
#[test]
fn test_sub_internal_hold_add_has_parentheses() {
    let mut ctx = Context::new();

    let a = ctx.var("a");
    let b = ctx.var("b");
    let c = ctx.var("c");

    let a_plus_b = ctx.add(Expr::Add(a, b));
    let held_sum = hold::wrap_hold(&mut ctx, a_plus_b);
    let expr = ctx.add(Expr::Sub(c, held_sum));

    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr
        }
    );

    assert!(
        display.contains("- ("),
        "Subtraction from an internal held sum must keep grouping. Got: '{}'",
        display
    );
    assert!(
        display.contains("a + b") || display.contains("b + a"),
        "Expected held sum inside subtraction parentheses. Got: '{}'",
        display
    );
}

/// Test that Neg(Sub) also gets parentheses: `c - (a - b)`
#[test]
fn test_neg_sub_has_parentheses() {
    let mut ctx = Context::new();

    // Build: c - (a - b)
    let a = ctx.var("a");
    let b = ctx.var("b");
    let c = ctx.var("c");

    let a_minus_b = ctx.add(Expr::Sub(a, b));
    let neg_diff = ctx.add(Expr::Neg(a_minus_b));
    let expr = ctx.add(Expr::Add(c, neg_diff));

    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: expr
        }
    );

    println!("Neg(Sub) display: {}", display);

    // Must have parentheses around the subtracted difference
    assert!(
        display.contains("(a - b)"),
        "Expected parentheses around difference in '{}'. Neg(Sub) must have parens.",
        display
    );
}

/// Regression test: the exact expression from the original bug report
#[test]
fn test_fraction_combination_display_regression() {
    let mut ctx = Context::new();

    // Build the numerator: x + 1 - (3x + 2) as the rule produces it
    let x = ctx.var("x");
    let one = ctx.num(1);
    let two = ctx.num(2);
    let three = ctx.num(3);

    // (x + 1)
    let x_plus_1 = ctx.add(Expr::Add(x, one));

    // (3x + 2)
    let three_x = ctx.add(Expr::Mul(three, x));
    let three_x_plus_2 = ctx.add(Expr::Add(three_x, two));

    // Neg(3x + 2)
    let neg_sum = ctx.add(Expr::Neg(three_x_plus_2));

    // (x + 1) + Neg(3x + 2)
    let numerator = ctx.add(Expr::Add(x_plus_1, neg_sum));

    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: numerator
        }
    );

    println!("Numerator display: {}", display);

    // The critical check: the subtracted sum should be in parentheses
    // Look for "- (" pattern which indicates proper parenthesization
    assert!(
        display.contains("- ("),
        "REGRESSION: Expected '- (' indicating parenthesized subtraction. \
         Expected something like 'x + 1 - (3 * x + 2)'. Got: '{}'",
        display
    );
}
