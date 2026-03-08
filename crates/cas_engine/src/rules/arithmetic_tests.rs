use super::arithmetic::{AddZeroRule, CombineConstantsRule, MulOneRule};
use crate::rule::{Rule, SimpleRule};
use crate::step::ImportanceLevel;
use cas_ast::{Context, Expr};
use cas_formatter::DisplayExpr;

#[test]
fn test_add_zero() {
    let mut ctx = Context::new();
    let rule = AddZeroRule;
    let x = ctx.var("x");
    let zero = ctx.num(0);
    let expr = ctx.add(Expr::Add(x, zero));
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
        "x"
    );
}

#[test]
fn test_mul_one() {
    let mut ctx = Context::new();
    let rule = MulOneRule;
    let one = ctx.num(1);
    let y = ctx.var("y");
    let expr = ctx.add(Expr::Mul(one, y));
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
        "y"
    );
}

#[test]
fn test_combine_constants() {
    let mut ctx = Context::new();
    let rule = CombineConstantsRule;
    let two = ctx.num(2);
    let three = ctx.num(3);
    let expr = ctx.add(Expr::Add(two, three));
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
        "5"
    );
}

#[test]
fn test_mul_one_rule_importance() {
    let rule = MulOneRule;
    assert_eq!(
        SimpleRule::importance(&rule),
        ImportanceLevel::Low,
        "MulOneRule should have Low importance (hidden in normal mode)"
    );
}

#[test]
fn test_add_zero_rule_importance() {
    let rule = AddZeroRule;
    assert_eq!(
        SimpleRule::importance(&rule),
        ImportanceLevel::Low,
        "AddZeroRule should have Low importance (hidden in normal mode)"
    );
}
