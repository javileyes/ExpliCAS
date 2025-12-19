#[test]
fn test_hold_transparency_from_engine() {
    use cas_ast::{Context, Expr, DisplayExpr};
    let mut ctx = Context::new();
    let x = ctx.add(Expr::Variable("x".to_string()));
    let held = ctx.add(Expr::Function("__hold".to_string(), vec![x]));
    let display = format!("{}", DisplayExpr { context: &ctx, id: held });
    assert_eq!(display, "x", "Expected x but got {}", display);
}
