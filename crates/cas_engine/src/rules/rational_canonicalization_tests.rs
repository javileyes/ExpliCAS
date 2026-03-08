use super::rational_canonicalization::{CanonicalizeNestedPowRule, CanonicalizeRationalDivRule};
use crate::rule::Rule;
use cas_ast::{Context, Expr};

#[test]
fn test_rational_div_canonicalization() {
    let mut ctx = Context::new();
    let rule = CanonicalizeRationalDivRule;
    let five = ctx.num(5);
    let six = ctx.num(6);
    let expr = ctx.add(Expr::Div(five, six));
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    if let Expr::Number(n) = ctx.get(rewrite.new_expr) {
        assert_eq!(n.to_string(), "5/6");
    } else {
        panic!("Expected Number, got {:?}", ctx.get(rewrite.new_expr));
    }
}

#[test]
fn test_rational_div_zero_denominator() {
    let mut ctx = Context::new();
    let rule = CanonicalizeRationalDivRule;
    let five = ctx.num(5);
    let zero = ctx.num(0);
    let expr = ctx.add(Expr::Div(five, zero));
    // Should NOT rewrite — division by zero
    assert!(rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .is_none());
}

#[test]
fn test_nested_pow_safe_odd_k() {
    // (x^3)^(1/2) → x^(3/2) — safe because k=3 is odd
    let mut ctx = Context::new();
    let rule = CanonicalizeNestedPowRule;
    let x = ctx.var("x");
    let three = ctx.num(3);
    let inner = ctx.add(Expr::Pow(x, three));
    let half = ctx.rational(1, 2);
    let expr = ctx.add(Expr::Pow(inner, half));
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    // Should be Pow(x, 3/2)
    if let Expr::Pow(base, exp) = ctx.get(rewrite.new_expr) {
        assert!(matches!(ctx.get(*base), Expr::Variable(_)));
        if let Expr::Number(n) = ctx.get(*exp) {
            assert_eq!(n.to_string(), "3/2");
        } else {
            panic!("Expected Number exponent");
        }
    } else {
        panic!("Expected Pow");
    }
}

#[test]
fn test_nested_pow_unsafe_even_k_even_q() {
    // (x^2)^(1/2) — MUST NOT rewrite (would give x instead of |x|)
    let mut ctx = Context::new();
    let rule = CanonicalizeNestedPowRule;
    let x = ctx.var("x");
    let two = ctx.num(2);
    let inner = ctx.add(Expr::Pow(x, two));
    let half = ctx.rational(1, 2);
    let expr = ctx.add(Expr::Pow(inner, half));
    assert!(rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .is_none());
}

#[test]
fn test_nested_pow_integer_exponent() {
    // (x^3)^2 → x^6 — safe because outer exponent is integer
    let mut ctx = Context::new();
    let rule = CanonicalizeNestedPowRule;
    let x = ctx.var("x");
    let three = ctx.num(3);
    let inner = ctx.add(Expr::Pow(x, three));
    let two = ctx.num(2);
    let expr = ctx.add(Expr::Pow(inner, two));
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    if let Expr::Pow(_, exp) = ctx.get(rewrite.new_expr) {
        if let Expr::Number(n) = ctx.get(*exp) {
            assert_eq!(n.to_string(), "6");
        } else {
            panic!("Expected Number exponent");
        }
    } else {
        panic!("Expected Pow");
    }
}
