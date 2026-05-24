use super::functions::{
    AbsDomainAddSubCancellationRule, AbsExpRule, AbsIdempotentRule, AbsOfEvenPowerRule,
    AbsSqrtRule, AbsSumOfSquaresRule, EvaluateAbsRule, SimplifySqrtSquareRule,
};
use crate::rule::Rule;
use cas_ast::{target_kind::TargetKind, Context};
use cas_formatter::DisplayExpr;
use cas_parser::parse;

#[test]
fn evaluate_abs_rule_targets_function_only() {
    let targets = EvaluateAbsRule
        .target_types()
        .expect("EvaluateAbsRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Function));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Sub));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Div));
    assert!(!targets.contains(TargetKind::Pow));
}

#[test]
fn simplify_sqrt_square_rule_targets_function_and_pow_only() {
    let targets = SimplifySqrtSquareRule
        .target_types()
        .expect("SimplifySqrtSquareRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Function));
    assert!(targets.contains(TargetKind::Pow));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Sub));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Div));
}

#[test]
fn abs_idempotent_rule_targets_function_only() {
    let targets = AbsIdempotentRule
        .target_types()
        .expect("AbsIdempotentRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Function));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Sub));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Div));
    assert!(!targets.contains(TargetKind::Pow));
}

#[test]
fn abs_of_even_power_rule_targets_function_only() {
    let targets = AbsOfEvenPowerRule
        .target_types()
        .expect("AbsOfEvenPowerRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Function));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Sub));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Div));
    assert!(!targets.contains(TargetKind::Pow));
}

#[test]
fn abs_sqrt_rule_targets_function_only() {
    let targets = AbsSqrtRule
        .target_types()
        .expect("AbsSqrtRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Function));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Sub));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Div));
    assert!(!targets.contains(TargetKind::Pow));
}

#[test]
fn abs_exp_rule_targets_function_only() {
    let targets = AbsExpRule
        .target_types()
        .expect("AbsExpRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Function));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Sub));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Div));
    assert!(!targets.contains(TargetKind::Pow));
}

#[test]
fn abs_sum_of_squares_rule_targets_function_only() {
    let targets = AbsSumOfSquaresRule
        .target_types()
        .expect("AbsSumOfSquaresRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Function));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Sub));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Div));
    assert!(!targets.contains(TargetKind::Pow));
}

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
