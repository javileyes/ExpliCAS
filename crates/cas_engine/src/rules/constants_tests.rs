use super::constants::{PhiReciprocalRule, PhiSquaredRule, RecognizePhiRule};
use crate::rule::Rule;
use cas_ast::{Constant, Context, Expr};
use cas_formatter::DisplayExpr;

#[test]
fn test_recognize_phi_div_form() {
    // (1 + sqrt(5)) / 2 -> phi
    let mut ctx = Context::new();
    let one = ctx.num(1);
    let five = ctx.num(5);
    let half_exp = ctx.rational(1, 2);
    let sqrt5 = ctx.add(Expr::Pow(five, half_exp));
    let sum = ctx.add(Expr::Add(one, sqrt5));
    let two = ctx.num(2);
    let expr = ctx.add(Expr::Div(sum, two));

    let rule = RecognizePhiRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_some(), "Should recognize (1+√5)/2 as phi");
    assert!(matches!(
        ctx.get(rewrite.unwrap().new_expr),
        Expr::Constant(Constant::Phi)
    ));
}

#[test]
fn test_recognize_phi_mul_form() {
    // 1/2 * (1 + sqrt(5)) -> phi
    let mut ctx = Context::new();
    let one = ctx.num(1);
    let five = ctx.num(5);
    let half_exp = ctx.rational(1, 2);
    let sqrt5 = ctx.add(Expr::Pow(five, half_exp));
    let sum = ctx.add(Expr::Add(one, sqrt5));
    let half = ctx.rational(1, 2);
    let expr = ctx.add(Expr::Mul(half, sum));

    let rule = RecognizePhiRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_some(), "Should recognize 1/2*(1+√5) as phi");
}

#[test]
fn test_phi_squared() {
    // phi^2 -> phi + 1
    let mut ctx = Context::new();
    let phi = ctx.add(Expr::Constant(Constant::Phi));
    let two = ctx.num(2);
    let expr = ctx.add(Expr::Pow(phi, two));

    let rule = PhiSquaredRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_some(), "Should simplify phi^2");
    let result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.unwrap().new_expr
        }
    );
    assert!(
        result.contains("phi") && result.contains("1"),
        "Result should be phi + 1: {}",
        result
    );
}

#[test]
fn test_phi_reciprocal() {
    // 1/phi -> phi - 1
    let mut ctx = Context::new();
    let one = ctx.num(1);
    let phi = ctx.add(Expr::Constant(Constant::Phi));
    let expr = ctx.add(Expr::Div(one, phi));

    let rule = PhiReciprocalRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_some(), "Should simplify 1/phi");
    let result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.unwrap().new_expr
        }
    );
    assert!(
        result.contains("phi"),
        "Result should contain phi: {}",
        result
    );
}

#[test]
fn test_phi_stays_phi() {
    // phi should remain phi (no inverse rule)
    let mut ctx = Context::new();
    let phi = ctx.add(Expr::Constant(Constant::Phi));

    let rule1 = RecognizePhiRule;
    let rule2 = PhiSquaredRule;
    let rule3 = PhiReciprocalRule;

    assert!(rule1
        .apply(&mut ctx, phi, &crate::parent_context::ParentContext::root())
        .is_none());
    assert!(rule2
        .apply(&mut ctx, phi, &crate::parent_context::ParentContext::root())
        .is_none());
    assert!(rule3
        .apply(&mut ctx, phi, &crate::parent_context::ParentContext::root())
        .is_none());
}
