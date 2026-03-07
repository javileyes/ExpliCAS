use super::super::*;
use cas_ast::{Context, Expr};
use cas_solver::Step;

#[test]
fn test_html_generation() {
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let three = ctx.num(3);
    let add_expr = ctx.add(Expr::Add(two, three));
    let five = ctx.num(5);

    let steps = vec![Step::new(
        "2 + 3 = 5",
        "Combine Constants",
        add_expr,
        five,
        vec![],
        Some(&ctx),
    )];

    let mut timeline = TimelineHtml::new(&mut ctx, &steps, add_expr, VerbosityLevel::Verbose);
    let html = timeline.to_html();

    assert!(html.contains("<!DOCTYPE html"));
    assert!(html.contains("timeline"));
    assert!(html.contains("CAS Simplification"));
    assert!(html.contains("Combine Constants"));
}

#[test]
fn test_html_escape() {
    assert_eq!(html_escape("<script>"), "&lt;script&gt;");
    assert_eq!(html_escape("x & y"), "x &amp; y");
}

#[test]
fn render_simplify_timeline_helper_produces_document() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");
    let add = ctx.add(cas_ast::Expr::Add(x, y));
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
