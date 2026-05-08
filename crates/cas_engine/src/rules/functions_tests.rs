use super::functions::{AbsDomainAddSubCancellationRule, EvaluateAbsRule};
use crate::rule::Rule;
use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

#[test]
fn test_evaluate_abs() {
    let mut ctx = Context::new();
    let rule = EvaluateAbsRule;

    // abs(-5) -> 5
    // Note: Parser might produce Number(-5) or Neg(Number(5)).
    // Our parser likely produces Number(-5) for literals.
    let expr1 = parse("abs(-5)", &mut ctx).expect("Failed to parse abs(-5)");
    let rewrite1 = rule
        .apply(
            &mut ctx,
            expr1,
            &crate::parent_context::ParentContext::root(),
        )
        .expect("Rule failed to apply");
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite1.new_expr
            }
        ),
        "5"
    );

    // abs(5) -> 5
    let expr2 = parse("abs(5)", &mut ctx).expect("Failed to parse abs(5)");
    let rewrite2 = rule
        .apply(
            &mut ctx,
            expr2,
            &crate::parent_context::ParentContext::root(),
        )
        .expect("Rule failed to apply");
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite2.new_expr
            }
        ),
        "5"
    );

    // abs(-x) -> abs(x)
    let expr3 = parse("abs(-x)", &mut ctx).expect("Failed to parse abs(-x)");
    let rewrite3 = rule
        .apply(
            &mut ctx,
            expr3,
            &crate::parent_context::ParentContext::root(),
        )
        .expect("Rule failed to apply");
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite3.new_expr
            }
        ),
        "|x|"
    );
}

#[test]
fn test_abs_domain_add_sub_cancellation_uses_root_lower_bound_for_product_reciprocal() {
    let mut ctx = Context::new();
    let expr = parse(
        "abs(1/((x-1)*(x-2)))-1/((x-1)*(x-2))+acosh(3-2*x)-acosh(3-2*x)",
        &mut ctx,
    )
    .expect("parse product reciprocal abs cancellation");
    let parent_ctx = crate::parent_context::ParentContext::root().with_root_expr(&ctx, expr);
    let rule = AbsDomainAddSubCancellationRule;

    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .expect("domain add/sub cancellation should apply");

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
}

#[test]
fn test_abs_domain_add_sub_cancellation_uses_root_lower_bound_for_mixed_power_product_reciprocal() {
    let mut ctx = Context::new();
    let root = parse(
        "abs(1/((x-1)^2*(x-2)))+1/((x-1)^2*(x-2))+acosh(3-2*x)-acosh(3-2*x)",
        &mut ctx,
    )
    .expect("parse mixed power product reciprocal abs cancellation");
    let expr = parse("abs(1/((x-1)^2*(x-2)))+1/((x-1)^2*(x-2))", &mut ctx)
        .expect("parse mixed power product residual");
    let parent_ctx = crate::parent_context::ParentContext::root().with_root_expr(&ctx, root);
    let rule = AbsDomainAddSubCancellationRule;

    let root_rewrite = rule
        .apply(&mut ctx, root, &parent_ctx)
        .expect("domain add/sub cancellation should apply at the root");
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: root_rewrite.new_expr
            }
        ),
        "0"
    );

    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .expect("domain add/sub cancellation should apply");

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
}
