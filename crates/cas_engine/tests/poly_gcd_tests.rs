#[test]
fn test_hold_transparency_from_engine() {
    use cas_ast::{Context, DisplayExpr, Expr};
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let held = ctx.call("__hold", vec![x]);
    let display = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: held
        }
    );
    assert_eq!(display, "x", "Expected x but got {}", display);
}
