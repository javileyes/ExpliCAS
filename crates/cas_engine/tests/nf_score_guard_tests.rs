// Guard tests for nf_score policy - ensures canonicalizing rewrites work correctly
//
// These tests guard against regression in the semantic equality wrapper policy.
// The issue was that Div(1,2) → Number(1/2) was being rejected because they're
// semantically equal. The fix uses nf_score to accept canonicalizing rewrites.

use cas_ast::{Context, DisplayExpr, Expr};
use cas_engine::Simplifier;
use cas_parser::parse;

/// Verify that 4^(1/2) simplifies to 2, even when exponent is Div form
#[test]
fn test_sqrt_4_with_div_exponent() {
    let mut ctx = Context::new();
    let four = ctx.num(4);
    let one = ctx.num(1);
    let two = ctx.num(2);

    // Create 4^(1/2) using Div structure
    let half_div = ctx.add(Expr::Div(one, two));
    let pow_expr = ctx.add(Expr::Pow(four, half_div));

    let mut simplifier = Simplifier::with_default_rules();
    simplifier.context = ctx;

    let (result, _) = simplifier.simplify(pow_expr);

    let display = DisplayExpr {
        context: &simplifier.context,
        id: result,
    };

    assert_eq!(
        display.to_string(),
        "2",
        "4^(Div(1,2)) should simplify to 2"
    );
}

/// Verify that -4^(1/2) simplifies to -2
#[test]
fn test_neg_sqrt_4_with_div_exponent() {
    let mut ctx = Context::new();
    let four = ctx.num(4);
    let one = ctx.num(1);
    let two = ctx.num(2);

    // Create -(4^(1/2)) using Div structure
    let half_div = ctx.add(Expr::Div(one, two));
    let pow_expr = ctx.add(Expr::Pow(four, half_div));
    let neg_expr = ctx.add(Expr::Neg(pow_expr));

    let mut simplifier = Simplifier::with_default_rules();
    simplifier.context = ctx;

    let (result, _) = simplifier.simplify(neg_expr);

    let display = DisplayExpr {
        context: &simplifier.context,
        id: result,
    };

    assert_eq!(
        display.to_string(),
        "-2",
        "-(4^(Div(1,2))) should simplify to -2"
    );
}

/// Verify that 4^(1/2) with Number exponent also works (control test)
#[test]
fn test_sqrt_4_with_number_exponent() {
    let mut ctx = Context::new();
    let four = ctx.num(4);
    let half = ctx.rational(1, 2);

    let pow_expr = ctx.add(Expr::Pow(four, half));

    let mut simplifier = Simplifier::with_default_rules();
    simplifier.context = ctx;

    let (result, _) = simplifier.simplify(pow_expr);

    let display = DisplayExpr {
        context: &simplifier.context,
        id: result,
    };

    assert_eq!(
        display.to_string(),
        "2",
        "4^(Number(1/2)) should simplify to 2"
    );
}

/// Verify that parsing "4^(1/2)" works correctly
#[test]
fn test_sqrt_4_parsed() {
    let mut ctx = Context::new();
    let expr = parse("4^(1/2)", &mut ctx).expect("should parse");

    let mut simplifier = Simplifier::with_default_rules();
    simplifier.context = ctx;

    let (result, _) = simplifier.simplify(expr);

    let display = DisplayExpr {
        context: &simplifier.context,
        id: result,
    };

    assert_eq!(display.to_string(), "2", "4^(1/2) should simplify to 2");
}
