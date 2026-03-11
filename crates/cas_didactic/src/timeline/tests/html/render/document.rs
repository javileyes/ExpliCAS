use super::super::super::super::*;
use crate::runtime::Step;
use cas_ast::{Context, Expr};

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
