use super::{
    try_plan_collect_rule_expr, try_rewrite_collect_like_terms_expr,
    try_rewrite_collect_like_terms_identity_expr,
};
use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_math::collect_semantics_support::CollectSemanticsMode;
use cas_parser::parse;

#[test]
fn builds_plan_for_real_combination() {
    let mut ctx = Context::new();
    let expr = parse("x + x", &mut ctx).expect("parse");
    let plan = try_plan_collect_rule_expr(&mut ctx, expr, CollectSemanticsMode::Generic, false)
        .expect("plan");
    assert_ne!(plan.new_expr, expr);
    assert!(plan.local_before.is_some());
    assert!(plan.local_after.is_some());
    assert_eq!(plan.description, "Combine like terms");
}

#[test]
fn returns_none_when_strict_blocks() {
    let mut ctx = Context::new();
    let expr = parse("x/(x+1) - x/(x+1)", &mut ctx).expect("parse");
    let plan = try_plan_collect_rule_expr(&mut ctx, expr, CollectSemanticsMode::Strict, true);
    assert!(plan.is_none());
}

#[test]
fn rewrite_collect_like_terms_add() {
    let mut ctx = Context::new();
    let expr = parse("x + x", &mut ctx).expect("parse");
    let rewritten = try_rewrite_collect_like_terms_expr(&mut ctx, expr).expect("rewrite");
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewritten
            }
        ),
        "2 * x"
    );
}

#[test]
fn rewrite_collect_like_terms_identity_has_desc() {
    let mut ctx = Context::new();
    let expr = parse("x + x", &mut ctx).expect("parse");
    let rewrite = try_rewrite_collect_like_terms_identity_expr(&mut ctx, expr).expect("rewrite");
    assert_eq!(rewrite.desc, "Collect like terms");
}
