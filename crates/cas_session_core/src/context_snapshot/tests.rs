use super::ContextSnapshot;

#[test]
fn context_snapshot_roundtrip_preserves_node_count() {
    let mut ctx = cas_ast::Context::new();
    let x = ctx.var("x");
    let two = ctx.num(2);
    let _expr = ctx.add(cas_ast::Expr::Mul(x, two));

    let snapshot = ContextSnapshot::from_context(&ctx);
    let restored = snapshot.into_context();
    assert_eq!(ctx.nodes.len(), restored.nodes.len());
}
