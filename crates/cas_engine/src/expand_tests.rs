use crate::expand::expand;
use cas_ast::{Context, ExprId};
use cas_formatter::DisplayExpr;
use cas_parser::parse;

fn s(ctx: &Context, id: ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

#[test]
fn test_expand_mul_distribute() {
    let mut ctx = Context::new();
    let expr = parse("2 * (x + 3)", &mut ctx).expect("parse should work");
    let res = expand(&mut ctx, expr);
    assert_eq!(s(&ctx, res), "2 * x + 2 * 3");
}

#[test]
fn test_expand_mul_nested() {
    let mut ctx = Context::new();
    let expr = parse("a * (b + c + d)", &mut ctx).expect("parse should work");
    let res = expand(&mut ctx, expr);
    let str_res = s(&ctx, res);
    assert!(str_res.contains("a * b"));
    assert!(str_res.contains("a * c"));
    assert!(str_res.contains("a * d"));
}

#[test]
fn test_expand_pow_binomial() {
    let mut ctx = Context::new();
    let expr = parse("(x + 1)^2", &mut ctx).expect("parse should work");
    let res = expand(&mut ctx, expr);
    let str_res = s(&ctx, res);
    assert!(str_res.contains("x^2"));
}
