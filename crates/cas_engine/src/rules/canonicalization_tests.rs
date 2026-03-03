use super::canonicalization::{
    CancelFractionSignsRule, CanonicalizeNegationRule, CanonicalizeRootRule,
};
use crate::rule::Rule;
use cas_ast::{Context, Expr};
use cas_formatter::DisplayExpr;

#[test]
fn test_canonicalize_negation() {
    let mut ctx = Context::new();
    let rule = CanonicalizeNegationRule;
    // -5 -> -5 (Number)
    // Use add_raw to bypass Context::add's canonicalization which already converts Neg(Number(n)) -> Number(-n)
    let five = ctx.num(5);
    let expr = ctx.add_raw(Expr::Neg(five));
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    // The display might look the same "-5", but the structure is different.
    // Let's check if it's a Number.
    if let Expr::Number(n) = ctx.get(rewrite.new_expr) {
        assert_eq!(format!("{}", n), "-5");
    } else {
        panic!("Expected Number, got {:?}", ctx.get(rewrite.new_expr));
    }
}

#[test]
fn test_canonicalize_sqrt() {
    let mut ctx = Context::new();
    let rule = CanonicalizeRootRule;
    // sqrt(x)
    let x = ctx.var("x");
    let expr = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x]);
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    // Should be x^(1/2)
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "x^(1/2)"
    );
}

#[test]
fn test_canonicalize_nth_root() {
    let mut ctx = Context::new();
    let rule = CanonicalizeRootRule;

    // sqrt(x, 3) -> x^(1/3)
    let x = ctx.var("x");
    let three = ctx.num(3);
    let expr = ctx.call_builtin(cas_ast::BuiltinFn::Sqrt, vec![x, three]);
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "x^(1 / 3)"
    );

    // root(x, 4) -> x^(1/4)
    let four = ctx.num(4);
    let expr2 = ctx.call_builtin(cas_ast::BuiltinFn::Root, vec![x, four]);
    let rewrite2 = rule
        .apply(
            &mut ctx,
            expr2,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite2.new_expr
            }
        ),
        "x^(1 / 4)"
    );
}

#[test]
fn test_cancel_fraction_signs_explicit_neg() {
    // (-a)/(-b) -> a/b
    let mut ctx = Context::new();
    let a = ctx.var("a");
    let b = ctx.var("b");
    let neg_a = ctx.add(Expr::Neg(a));
    let neg_b = ctx.add(Expr::Neg(b));
    let expr = ctx.add(Expr::Div(neg_a, neg_b));

    let rule = CancelFractionSignsRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_some(), "Should apply to (-a)/(-b)");
    let result = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.unwrap().new_expr
        }
    );
    assert_eq!(result, "a / b");
}

#[test]
fn test_cancel_fraction_signs_sub_implicit() {
    // (1-x)/(1-y) -> (x-1)/(y-1) because 1 < x and 1 < y canonically
    let mut ctx = Context::new();
    let one = ctx.num(1);
    let x = ctx.var("x");
    let y = ctx.var("y");
    // Build Sub(1, x) and Sub(1, y)
    let num = ctx.add(Expr::Sub(one, x));
    let den = ctx.add(Expr::Sub(one, y));
    let expr = ctx.add(Expr::Div(num, den));

    let rule = CancelFractionSignsRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_some(), "Should apply to (1-x)/(1-y)");
}

#[test]
fn test_cancel_fraction_signs_single_neg_unchanged() {
    // (-a)/b should NOT be changed by this rule (only one is negative)
    let mut ctx = Context::new();
    let a = ctx.var("a");
    let b = ctx.var("b");
    let neg_a = ctx.add(Expr::Neg(a));
    let expr = ctx.add(Expr::Div(neg_a, b));

    let rule = CancelFractionSignsRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_none(), "Should NOT apply to (-a)/b");
}

#[test]
fn test_cancel_fraction_signs_single_neg_den_unchanged() {
    // a/(-b) should NOT be changed by this rule (only one is negative)
    let mut ctx = Context::new();
    let a = ctx.var("a");
    let b = ctx.var("b");
    let neg_b = ctx.add(Expr::Neg(b));
    let expr = ctx.add(Expr::Div(a, neg_b));

    let rule = CancelFractionSignsRule;
    let rewrite = rule.apply(
        &mut ctx,
        expr,
        &crate::parent_context::ParentContext::root(),
    );

    assert!(rewrite.is_none(), "Should NOT apply to a/(-b)");
}
