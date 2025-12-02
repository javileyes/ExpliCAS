use cas_ast::{Context, Expr, DisplayExpr};
use cas_engine::collect::collect;
use cas_parser::parse;

fn s(ctx: &Context, id: cas_ast::ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

#[test]
fn test_collect_double_negation() {
    let mut ctx = Context::new();
    // -(-x) -> x
    let x = parse("x", &mut ctx).unwrap();
    let neg_x = ctx.add(Expr::Neg(x));
    let neg_neg_x = ctx.add(Expr::Neg(neg_x));
    
    let res = collect(&mut ctx, neg_neg_x);
    assert_eq!(s(&ctx, res), "x");
}

#[test]
fn test_collect_sub_neg() {
    let mut ctx = Context::new();
    // a - (-b) -> a + b
    let expr = parse("a - (-b)", &mut ctx).unwrap();
    let res = collect(&mut ctx, expr);
    // Should be "a + b" or "b + a" (sorted)
    let res_str = s(&ctx, res);
    assert!(res_str == "a + b" || res_str == "b + a");
}

#[test]
fn test_collect_nested_neg_add() {
    let mut ctx = Context::new();
    // a + -(-b) -> a + b
    let expr = parse("a + -(-b)", &mut ctx).unwrap();
    let res = collect(&mut ctx, expr);
    let res_str = s(&ctx, res);
    assert!(res_str == "a + b" || res_str == "b + a");
}

#[test]
fn test_collect_neg_neg_cos() {
    let mut ctx = Context::new();
    // -(-cos(x)) -> cos(x)
    let expr = parse("-(-cos(x))", &mut ctx).unwrap();
    let res = collect(&mut ctx, expr);
    assert_eq!(s(&ctx, res), "cos(x)");
}

#[test]
fn test_collect_sub_neg_cos() {
    let mut ctx = Context::new();
    // -3 - (-cos(x)) -> -3 + cos(x)
    // Note: parser might parse -3 as Neg(3) or Number(-3).
    // " - (-cos(x)) " is Sub(..., Neg(cos(x))).
    let expr = parse("-3 - (-cos(x))", &mut ctx).unwrap();
    let res = collect(&mut ctx, expr);
    // Expected: -3 + cos(x) (or sorted)
    // Sorted: -3 comes first? Number vs Function.
    // Number usually first.
    let res_str = s(&ctx, res);
    println!("Result: {}", res_str);
    assert!(res_str.contains("cos(x)"));
    assert!(!res_str.contains("- -")); // Should not have double negation
    assert!(!res_str.contains("- (-"));
}

#[test]
fn test_collect_user_repro() {
    let mut ctx = Context::new();
    // 8 * sin(x)^4 - (3 - 4 * cos(2 * x) + cos(4 * x))
    let expr = parse("8 * sin(x)^4 - (3 - 4 * cos(2 * x) + cos(4 * x))", &mut ctx).unwrap();
    let res = collect(&mut ctx, expr);
    let res_str = s(&ctx, res);
    println!("User Repro Result: {}", res_str);
    
    // Should be: -3 + cos(4x) + 4cos(2x) + 8sin^4(x)
    // Or similar. Definitely NOT "- -cos"
    assert!(!res_str.contains("- -cos"));
    assert!(!res_str.contains("- (-cos"));
    // Check for presence of terms
    assert!(res_str.contains("cos(4 * x)"));
    assert!(res_str.contains("3"));
}
