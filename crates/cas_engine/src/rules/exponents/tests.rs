use super::*;
use crate::rule::Rule;
use cas_ast::{Context, Expr};
use cas_formatter::DisplayExpr;

#[test]
fn test_product_power() {
    let mut ctx = Context::new();
    let rule = ProductPowerRule;

    // x^2 * x^3 -> x^(2+3)
    let x = ctx.var("x");
    let two = ctx.num(2);
    let three = ctx.num(3);
    let x2 = ctx.add(Expr::Pow(x, two));
    let x3 = ctx.add(Expr::Pow(x, three));
    let expr = ctx.add(Expr::Mul(x2, x3));

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
        "x^5"
    );

    // x * x -> x^2
    let expr2 = ctx.add(Expr::Mul(x, x));
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
        "x^2"
    );
}

#[test]
fn test_power_power() {
    let mut ctx = Context::new();
    let rule = PowerPowerRule;

    // (x^2)^3 -> x^(2*3)
    let x = ctx.var("x");
    let two = ctx.num(2);
    let three = ctx.num(3);
    let x2 = ctx.add(Expr::Pow(x, two));
    let expr = ctx.add(Expr::Pow(x2, three));

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
        "x^6"
    );
}

#[test]
fn test_zero_one_power() {
    let mut ctx = Context::new();
    let rule = IdentityPowerRule;

    // x^0 -> 1
    let x = ctx.var("x");
    let zero = ctx.num(0);
    let expr = ctx.add(Expr::Pow(x, zero));
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
        "1"
    );

    // x^1 -> x
    let one = ctx.num(1);
    let expr2 = ctx.add(Expr::Pow(x, one));
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
        "x"
    );
}
