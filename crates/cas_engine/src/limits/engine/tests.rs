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
    let Expr::Function(f, _) = ctx.get(e) else {
        return false;
    };
    let name = ctx.sym_name(*f);
    name == "limit"
}

#[test]
fn infinity_minus_infinity_combines_over_common_denominator() {
    // ∞ − ∞ at a finite point: the operand-wise split cannot subtract two
    // divergences, but combining `lhs − rhs` into one fraction lets the
    // engine's removable-hole / Taylor rules finish the job. The fix reaches
    // the combine fallback even when an operand has NO rule-level limit
    // (1/sin(x) at 0 is bilaterally undefined).
    let mut ctx = Context::new();
    for (src, expected) in [
        ("1/sin(x) - 1/x", 0i64),
        ("1/tan(x) - 1/x", 0),
        ("csc(x) - cot(x)", 0),
    ] {
        let r = bilateral(&mut ctx, src, "0");
        match ctx.get(r.expr) {
            Expr::Number(n) => assert!(
                *n == num_rational::BigRational::from_integer(expected.into()),
                "{src} at 0 should be {expected}, got {n}"
            ),
            other => panic!("{src} at 0 should resolve to {expected}, got {other:?}"),
        }
    }
    // A non-zero finite value through the same path: 1/(x-1) - 2/(x²-1) = 1/(x+1) -> 1/2.
    let half = bilateral(&mut ctx, "1/(x-1) - 2/(x^2-1)", "1");
    let one_half = num_rational::BigRational::new(1.into(), 2.into());
    assert!(
        matches!(ctx.get(half.expr), Expr::Number(n) if *n == one_half),
        "1/(x-1) - 2/(x^2-1) at 1 should be 1/2, got {:?}",
        ctx.get(half.expr)
    );
}

#[test]
fn infinity_minus_infinity_combines_in_the_add_arm_too() {
    // The `+` companion: `x/(x-1) + 1/(1-x) = 1` even though each operand
    // diverges at x = 1 (the two denominators are negatives of each other).
    let mut ctx = Context::new();
    let one_r = bilateral(&mut ctx, "x/(x-1) + 1/(1-x)", "1");
    assert!(
        matches!(ctx.get(one_r.expr), Expr::Number(n) if *n == num_rational::BigRational::from_integer(1.into())),
        "x/(x-1) + 1/(1-x) at 1 should be 1, got {:?}",
        ctx.get(one_r.expr)
    );
    let zero = bilateral(&mut ctx, "1/x + 1/(-x)", "0");
    assert!(matches!(ctx.get(zero.expr), Expr::Number(n) if n.is_zero()));
    // Same-sign ∞ + ∞ folds to a single ±∞, not a literal Add(∞, ∞).
    let pos = bilateral(&mut ctx, "1/x^2 + 1/x^2", "0");
    assert!(matches!(
        ctx.get(pos.expr),
        Expr::Constant(Constant::Infinity)
    ));
}

#[test]
fn infinity_minus_infinity_stays_sound_on_degenerate_and_divergent_forms() {
    let mut ctx = Context::new();
    // f - f with f -> ∞ is 0 (exact cancellation), not indeterminate garbage.
    let zero = bilateral(&mut ctx, "1/x^2 - 1/x^2", "0");
    assert!(matches!(ctx.get(zero.expr), Expr::Number(n) if n.is_zero()));
    // exp(x) - x diverges at +∞ — must not be mis-combined into a finite value.
    let x = parse_expr(&mut ctx, "x");
    let inf_call = {
        let expr = parse_expr(&mut ctx, "exp(x) - x");
        let mut budget = Budget::new();
        limit(
            &mut ctx,
            expr,
            x,
            Approach::PosInfinity,
            &LimitOptions::default(),
            &mut budget,
        )
        .expect("limit")
    };
    assert!(matches!(
        ctx.get(inf_call.expr),
        Expr::Constant(Constant::Infinity)
    ));
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
