use super::super::super::super::*;
use cas_ast::{Context, Expr};

#[test]
fn render_simplify_timeline_helper_produces_document() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");
    let add = ctx.add(Expr::Add(x, y));
    let steps = vec![];
    let html = render_simplify_timeline_html(
        &mut ctx,
        &steps,
        add,
        Some(add),
        VerbosityLevel::Normal,
        Some("x+y"),
    );
    assert!(html.contains("<!DOCTYPE html"));
}
