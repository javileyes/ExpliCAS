use crate::collect::{collect, simplify_numeric_exponents};
use cas_ast::{Context, Expr, ExprId};
use cas_formatter::DisplayExpr;
use cas_parser::parse;

fn s(ctx: &Context, id: ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

#[test]
fn test_collect_integers() {
    let mut ctx = Context::new();
    let expr = parse("1 + 2", &mut ctx).expect("parse should work");
    let res = collect(&mut ctx, expr);
    assert_eq!(s(&ctx, res), "3");
}

#[test]
fn test_collect_variables() {
    let mut ctx = Context::new();
    let expr = parse("x + x", &mut ctx).expect("parse should work");
    let res = collect(&mut ctx, expr);
    assert_eq!(s(&ctx, res), "2 * x");
}

#[test]
fn test_collect_mixed() {
    let mut ctx = Context::new();
    let expr = parse("2*x + 3*y + 4*x", &mut ctx).expect("parse should work");
    let res = collect(&mut ctx, expr);
    // Order depends on implementation, but should have 6*x and 3*y
    let res_str = s(&ctx, res);
    assert!(res_str.contains("6 * x"));
    assert!(res_str.contains("3 * y"));
}

#[test]
fn test_collect_cancel() {
    let mut ctx = Context::new();
    let expr = parse("x - x", &mut ctx).expect("parse should work");
    let res = collect(&mut ctx, expr);
    assert_eq!(s(&ctx, res), "0");
}

#[test]
fn test_collect_powers() {
    let mut ctx = Context::new();
    let expr = parse("x^2 + 2*x^2", &mut ctx).expect("parse should work");
    let res = collect(&mut ctx, expr);
    assert_eq!(s(&ctx, res), "3 * x^2");
}

#[test]
fn test_simplify_numeric_exponents() {
    let mut ctx = Context::new();
    // x^(1/2 + 1/3) should become x^(5/6)
    let expr = parse("x^(1/2 + 1/3)", &mut ctx).expect("parse should work");
    let res = simplify_numeric_exponents(&mut ctx, expr);
    assert_ne!(res, expr, "Expression should be simplified");
    assert_eq!(s(&ctx, res), "x^(5/6)");
}

#[test]
fn test_collect_double_negation() {
    let mut ctx = Context::new();
    let x = parse("x", &mut ctx).expect("parse should work");
    let neg_x = ctx.add(Expr::Neg(x));
    let neg_neg_x = ctx.add(Expr::Neg(neg_x));
    let res = collect(&mut ctx, neg_neg_x);
    assert_eq!(s(&ctx, res), "x");
}

#[test]
fn test_collect_sub_neg() {
    let mut ctx = Context::new();
    let expr = parse("a - (-b)", &mut ctx).expect("parse should work");
    let res = collect(&mut ctx, expr);
    let res_str = s(&ctx, res);
    assert!(res_str == "a + b" || res_str == "b + a");
}

#[test]
fn test_collect_nested_neg_add() {
    let mut ctx = Context::new();
    let expr = parse("a + -(-b)", &mut ctx).expect("parse should work");
    let res = collect(&mut ctx, expr);
    let res_str = s(&ctx, res);
    assert!(res_str == "a + b" || res_str == "b + a");
}

#[test]
fn test_collect_neg_neg_cos() {
    let mut ctx = Context::new();
    let expr = parse("-(-cos(x))", &mut ctx).expect("parse should work");
    let res = collect(&mut ctx, expr);
    assert_eq!(s(&ctx, res), "cos(x)");
}

#[test]
fn test_collect_sub_neg_cos() {
    let mut ctx = Context::new();
    let expr = parse("-3 - (-cos(x))", &mut ctx).expect("parse should work");
    let res = collect(&mut ctx, expr);
    let res_str = s(&ctx, res);
    assert!(res_str.contains("cos(x)"));
    assert!(!res_str.contains("- -"));
    assert!(!res_str.contains("- (-"));
}

#[test]
fn test_collect_user_repro() {
    let mut ctx = Context::new();
    let expr = parse("8 * sin(x)^4 - (3 - 4 * cos(2 * x) + cos(4 * x))", &mut ctx)
        .expect("parse should work");
    let res = collect(&mut ctx, expr);
    let res_str = s(&ctx, res);
    assert!(!res_str.contains("- -cos"));
    assert!(!res_str.contains("- (-cos"));
    assert!(res_str.contains("cos(4 * x)"));
    assert!(res_str.contains("3"));
}
