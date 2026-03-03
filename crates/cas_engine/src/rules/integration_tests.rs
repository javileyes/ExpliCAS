use super::integration::{CosProductTelescopingRule, ProductToSumRule};
use crate::parent_context::ParentContext;
use crate::rule::Rule;
use cas_ast::Context;
use cas_parser::parse;

#[test]
fn test_product_to_sum_sin_cos() {
    let mut ctx = Context::new();
    let expr = parse("2*sin(x)*cos(y)", &mut ctx).unwrap();

    let rule = ProductToSumRule;
    let result = rule.apply(&mut ctx, expr, &ParentContext::root());

    assert!(
        result.is_some(),
        "ProductToSum should match 2*sin(x)*cos(y)"
    );
}

#[test]
fn test_cos_product_telescoping() {
    let mut ctx = Context::new();
    // cos(x) * cos(2*x) * cos(4*x) -> sin(8*x) / (8*sin(x))
    let expr = parse("cos(x)*cos(2*x)*cos(4*x)", &mut ctx).unwrap();

    let rule = CosProductTelescopingRule;
    let result = rule.apply(&mut ctx, expr, &ParentContext::root());

    assert!(
        result.is_some(),
        "CosProductTelescoping should match cos(x)*cos(2x)*cos(4x)"
    );

    let rewrite = result.unwrap();
    // CosProductTelescopingRule now uses structured assumption_events
    assert!(
        !rewrite.assumption_events.is_empty(),
        "Should have assumption_events for sin(u) ≠ 0 warning"
    );
}
