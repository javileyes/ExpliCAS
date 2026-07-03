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

// ── Bilateral combiner (frontier-audit limits gap, auto-mejora cycle) ──

fn bilateral(ctx: &mut Context, expr_s: &str, point_s: &str) -> LimitResult {
    let expr = parse_expr(ctx, expr_s);
    let x = parse_expr(ctx, "x");
    let point = parse_expr(ctx, point_s);
    let mut budget = Budget::new();
    limit(
        ctx,
        expr,
        x,
        Approach::Finite(point),
        &LimitOptions::default(),
        &mut budget,
    )
    .expect("limit should not error")
}

fn is_undefined(ctx: &Context, e: ExprId) -> bool {
    matches!(ctx.get(e), Expr::Constant(Constant::Undefined))
}

fn is_pos_infinity(ctx: &Context, e: ExprId) -> bool {
    matches!(ctx.get(e), Expr::Constant(Constant::Infinity))
}

fn is_neg_infinity(ctx: &Context, e: ExprId) -> bool {
    if let Expr::Neg(inner) = ctx.get(e) {
        return matches!(ctx.get(*inner), Expr::Constant(Constant::Infinity));
    }
    false
}

fn is_residual_limit(ctx: &Context, e: ExprId) -> bool {
    matches!(ctx.get(e), Expr::Function(f, _) if ctx.sym_name(*f) == "limit")
}

#[test]
fn bilateral_dne_when_laterals_diverge_with_opposite_signs() {
    let mut ctx = Context::new();
    let r = bilateral(&mut ctx, "1/x", "0");
    assert!(
        is_undefined(&ctx, r.expr),
        "1/x at 0 must be DNE (undefined)"
    );
    let w = r.warning.expect("DNE must explain the laterals");
    assert!(
        w.contains("does not exist") && w.contains("−∞") && w.contains("+∞"),
        "{w}"
    );

    let r3 = bilateral(&mut ctx, "1/x^3", "0");
    assert!(is_undefined(&ctx, r3.expr));
    let shifted = bilateral(&mut ctx, "1/(x-2)", "2");
    assert!(is_undefined(&ctx, shifted.expr));
}

#[test]
fn bilateral_dne_when_finite_laterals_differ() {
    let mut ctx = Context::new();
    let r = bilateral(&mut ctx, "abs(x)/x", "0");
    assert!(is_undefined(&ctx, r.expr), "sign function jump is DNE");
    let w = r.warning.expect("warning");
    assert!(w.contains("-1") && w.contains("1"), "laterals quoted: {w}");
}

#[test]
fn bilateral_dne_when_one_lateral_finite_and_other_infinite() {
    let mut ctx = Context::new();
    // e^(1/x): x→0⁻ gives e^(−∞) = 0, x→0⁺ gives +∞.
    let r = bilateral(&mut ctx, "e^(1/x)", "0");
    assert!(is_undefined(&ctx, r.expr));
}

#[test]
fn bilateral_agreeing_negative_infinity_combines() {
    let mut ctx = Context::new();
    let r = bilateral(&mut ctx, "-1/x^2", "0");
    assert!(
        is_neg_infinity(&ctx, r.expr),
        "-1/x^2 at 0 is −∞ from both sides"
    );
    assert!(r.warning.is_none());
    // The even-power positive twin stays owned by the direct rule.
    let p = bilateral(&mut ctx, "1/x^2", "0");
    assert!(is_pos_infinity(&ctx, p.expr));
}

#[test]
fn bilateral_combiner_never_touches_domain_boundaries_or_oscillation() {
    let mut ctx = Context::new();
    // Domain only extends to one side: stays the honest residual.
    let sqrt = bilateral(&mut ctx, "sqrt(x)", "0");
    assert!(
        is_residual_limit(&ctx, sqrt.expr),
        "sqrt at 0 must stay residual"
    );
    let ln = bilateral(&mut ctx, "ln(x)", "0");
    assert!(is_residual_limit(&ctx, ln.expr));
    // Oscillation: neither lateral exists.
    let osc = bilateral(&mut ctx, "sin(1/x)", "0");
    assert!(is_residual_limit(&ctx, osc.expr));
}
