use super::*;
use cas_ast::{Constant, Expr};
use cas_parser::parse;
use num_traits::Zero;

fn parse_expr(ctx: &mut Context, s: &str) -> ExprId {
    parse(s, ctx).expect("parse failed")
}

// Contractual Test 1: lim_{x→∞} 1/x = 0
#[test]
fn test_limit_one_over_x_contract() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "1/x");
    let x = parse_expr(&mut ctx, "x");
    let mut budget = Budget::new();

    let result = limit(
        &mut ctx,
        expr,
        x,
        Approach::PosInfinity,
        &LimitOptions::default(),
        &mut budget,
    );
    assert!(result.is_ok());

    let r = result.unwrap();
    assert!(r.warning.is_none(), "Should resolve successfully");

    if let Expr::Number(n) = ctx.get(r.expr) {
        assert!(n.is_zero(), "Result should be 0");
    } else {
        panic!("Expected Number(0)");
    }
}

// Contractual Test 2: lim_{x→∞} 5/x^3 = 0
#[test]
fn test_limit_five_over_x_cubed_contract() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "5/x^3");
    let x = parse_expr(&mut ctx, "x");
    let mut budget = Budget::new();

    let result = limit(
        &mut ctx,
        expr,
        x,
        Approach::PosInfinity,
        &LimitOptions::default(),
        &mut budget,
    );
    assert!(result.is_ok());

    let r = result.unwrap();
    if let Expr::Number(n) = ctx.get(r.expr) {
        assert!(n.is_zero(), "Result should be 0");
    } else {
        panic!("Expected Number(0)");
    }
}

// Contractual Test 3: lim_{x→∞} x = ∞
#[test]
fn test_limit_x_pos_inf_contract() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let mut budget = Budget::new();

    let result = limit(
        &mut ctx,
        x,
        x,
        Approach::PosInfinity,
        &LimitOptions::default(),
        &mut budget,
    );
    assert!(result.is_ok());

    let r = result.unwrap();
    assert!(matches!(
        ctx.get(r.expr),
        Expr::Constant(Constant::Infinity)
    ));
}

// Contractual Test 4: lim_{x→-∞} x = -∞
#[test]
fn test_limit_x_neg_inf_contract() {
    let mut ctx = Context::new();
    let x = parse_expr(&mut ctx, "x");
    let mut budget = Budget::new();

    let result = limit(
        &mut ctx,
        x,
        x,
        Approach::NegInfinity,
        &LimitOptions::default(),
        &mut budget,
    );
    assert!(result.is_ok());

    let r = result.unwrap();
    assert!(matches!(ctx.get(r.expr), Expr::Neg(_)));
}

// Contractual Test 5: lim_{x→∞} x^2 = ∞
#[test]
fn test_limit_x_squared_pos_inf_contract() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x^2");
    let x = parse_expr(&mut ctx, "x");
    let mut budget = Budget::new();

    let result = limit(
        &mut ctx,
        expr,
        x,
        Approach::PosInfinity,
        &LimitOptions::default(),
        &mut budget,
    );
    assert!(result.is_ok());

    let r = result.unwrap();
    assert!(matches!(
        ctx.get(r.expr),
        Expr::Constant(Constant::Infinity)
    ));
}

// Contractual Test 6: lim_{x→-∞} x^3 = -∞
#[test]
fn test_limit_x_cubed_neg_inf_contract() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "x^3");
    let x = parse_expr(&mut ctx, "x");
    let mut budget = Budget::new();

    let result = limit(
        &mut ctx,
        expr,
        x,
        Approach::NegInfinity,
        &LimitOptions::default(),
        &mut budget,
    );
    assert!(result.is_ok());

    let r = result.unwrap();
    assert!(matches!(ctx.get(r.expr), Expr::Neg(_)));
}

// Test: Unresolved limit returns residual
#[test]
fn test_limit_unresolved_returns_residual() {
    let mut ctx = Context::new();
    let expr = parse_expr(&mut ctx, "sin(x)"); // Cannot resolve in V1
    let x = parse_expr(&mut ctx, "x");
    let mut budget = Budget::new();

    let result = limit(
        &mut ctx,
        expr,
        x,
        Approach::PosInfinity,
        &LimitOptions::default(),
        &mut budget,
    );
    assert!(result.is_ok());

    let r = result.unwrap();
    assert!(
        r.warning.is_some(),
        "Should have warning for unresolved limit"
    );

    // Should be residual Function("limit", ...)
    if let Expr::Function(fn_id, _) = ctx.get(r.expr) {
        assert_eq!(ctx.sym_name(*fn_id), "limit");
    } else {
        panic!("Expected residual limit function");
    }
}
