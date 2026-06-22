use super::*;
use crate::rule::Rule;
use cas_ast::{target_kind::TargetKind, Context, Expr};
use cas_formatter::DisplayExpr;
use cas_parser::parse;

#[test]
fn exp_quotient_rule_targets_div_only() {
    let targets = ExpQuotientRule
        .target_types()
        .expect("ExpQuotientRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Div));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Pow));
    assert!(!targets.contains(TargetKind::Function));
}

#[test]
fn test_exp_quotient_direct() {
    let mut ctx = Context::new();
    let rule = ExpQuotientRule;
    let expr = parse("e^a / e^b", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "e^(a - b)"
    );
}

fn apply_exp_quotient(src: &str) -> String {
    let mut ctx = Context::new();
    let expr = parse(src, &mut ctx).unwrap();
    let rewrite = ExpQuotientRule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap_or_else(|| panic!("ExpQuotientRule should fire on `{src}`"));
    format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    )
}

#[test]
fn test_exp_quotient_combines_through_a_numerator_product() {
    // A co-factor in the numerator no longer blocks the e^a/e^b combination.
    assert_eq!(apply_exp_quotient("x * e^a / e^b"), "x * e^(a - b)");
    assert_eq!(apply_exp_quotient("2 * x * e^a / e^b"), "2 * x * e^(a - b)");
}

#[test]
fn test_exp_quotient_combines_with_denominator_and_both_side_products() {
    assert_eq!(apply_exp_quotient("e^a / (y * e^b)"), "e^(a - b) / y");
    assert_eq!(
        apply_exp_quotient("x * e^a / (y * e^b)"),
        "x * e^(a - b) / y"
    );
}

#[test]
fn test_exp_quotient_declines_without_e_on_both_sides() {
    // No `e` power in the denominator → not a quotient combination → decline.
    let mut ctx = Context::new();
    let expr = parse("x * e^a / y", &mut ctx).unwrap();
    assert!(ExpQuotientRule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root()
        )
        .is_none());
}

#[test]
fn product_power_rule_targets_mul_only() {
    let targets = ProductPowerRule
        .target_types()
        .expect("ProductPowerRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Div));
    assert!(!targets.contains(TargetKind::Pow));
    assert!(!targets.contains(TargetKind::Function));
}

#[test]
fn product_same_exponent_rule_targets_mul_only() {
    let targets = ProductSameExponentRule
        .target_types()
        .expect("ProductSameExponentRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Div));
    assert!(!targets.contains(TargetKind::Pow));
}

#[test]
fn power_product_rule_targets_pow_only() {
    let targets = PowerProductRule
        .target_types()
        .expect("PowerProductRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Pow));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Div));
}

#[test]
fn power_quotient_rule_targets_pow_only() {
    let targets = PowerQuotientRule
        .target_types()
        .expect("PowerQuotientRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Pow));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Div));
}

#[test]
fn evaluate_power_rule_targets_pow_only() {
    let targets = EvaluatePowerRule
        .target_types()
        .expect("EvaluatePowerRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Pow));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Function));
}

#[test]
fn negative_exponent_normalization_rule_targets_pow_only() {
    let targets = NegativeExponentNormalizationRule
        .target_types()
        .expect("NegativeExponentNormalizationRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Pow));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Function));
}

#[test]
fn negative_base_power_rule_targets_pow_only() {
    let targets = NegativeBasePowerRule
        .target_types()
        .expect("NegativeBasePowerRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Pow));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Function));
}

#[test]
fn even_pow_sub_swap_rule_targets_pow_only() {
    let targets = EvenPowSubSwapRule
        .target_types()
        .expect("EvenPowSubSwapRule should be structurally targeted");

    assert!(targets.contains(TargetKind::Pow));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Function));
}

#[test]
fn test_product_power() {
    let mut ctx = Context::new();
    let rule = ProductPowerRule;

    // x^2 * x^3 -> x^(2+3)
    let x = ctx.var("x");
    let two = ctx.num(2);
    let three = ctx.num(3);
    let x2 = ctx.add(Expr::Pow(x, two));
    let x3 = ctx.add(Expr::Pow(x, three));
    let expr = ctx.add(Expr::Mul(x2, x3));

    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "x^5"
    );

    // x * x -> x^2
    let expr2 = ctx.add(Expr::Mul(x, x));
    let rewrite2 = rule
        .apply(
            &mut ctx,
            expr2,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite2.new_expr
            }
        ),
        "x^2"
    );
}

#[test]
fn test_power_power() {
    let mut ctx = Context::new();
    let rule = PowerPowerRule;

    // (x^2)^3 -> x^(2*3)
    let x = ctx.var("x");
    let two = ctx.num(2);
    let three = ctx.num(3);
    let x2 = ctx.add(Expr::Pow(x, two));
    let expr = ctx.add(Expr::Pow(x2, three));

    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "x^6"
    );
}

#[test]
fn test_zero_one_power() {
    let mut ctx = Context::new();
    let rule = IdentityPowerRule;

    // x^0 -> 1
    let x = ctx.var("x");
    let zero = ctx.num(0);
    let expr = ctx.add(Expr::Pow(x, zero));
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "1"
    );

    // x^1 -> x
    let one = ctx.num(1);
    let expr2 = ctx.add(Expr::Pow(x, one));
    let rewrite2 = rule
        .apply(
            &mut ctx,
            expr2,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite2.new_expr
            }
        ),
        "x"
    );
}
