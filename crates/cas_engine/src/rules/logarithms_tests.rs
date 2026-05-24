use super::logarithms::*;
use crate::rule::Rule;
use cas_ast::{target_kind::TargetKind, Context};
use cas_formatter::DisplayExpr;
use cas_parser::parse;

#[test]
fn evaluate_log_rule_targets_function_only() {
    let targets = EvaluateLogRule
        .target_types()
        .expect("EvaluateLogRule should target function calls");

    assert!(targets.contains(TargetKind::Function));
    assert!(!targets.contains(TargetKind::Add));
    assert!(!targets.contains(TargetKind::Mul));
    assert!(!targets.contains(TargetKind::Pow));
}

#[test]
fn ln_e_rules_target_function_only() {
    for (rule_name, targets) in [
        (
            "LnEProductRule",
            LnEProductRule
                .target_types()
                .expect("LnEProductRule should target function calls"),
        ),
        (
            "LnEDivRule",
            LnEDivRule
                .target_types()
                .expect("LnEDivRule should target function calls"),
        ),
    ] {
        assert!(
            targets.contains(TargetKind::Function),
            "{rule_name} should run on ln(...) calls"
        );
        assert!(
            !targets.contains(TargetKind::Add),
            "{rule_name} should not run on additive roots"
        );
        assert!(
            !targets.contains(TargetKind::Mul),
            "{rule_name} should not run on multiplicative roots"
        );
        assert!(
            !targets.contains(TargetKind::Div),
            "{rule_name} should not run on division roots"
        );
    }
}

#[test]
fn test_log_one() {
    let mut ctx = Context::new();
    let rule = EvaluateLogRule;
    // log(x, 1) -> 0
    let expr = parse("log(x, 1)", &mut ctx).unwrap();
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
        "0"
    );
}

#[test]
fn test_ln_e_product_direct() {
    let mut ctx = Context::new();
    let rule = LnEProductRule;
    let expr = parse("ln(e*x)", &mut ctx).unwrap();
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
        "ln(x) + 1"
    );
}

#[test]
fn test_ln_e_div_direct() {
    let mut ctx = Context::new();
    let rule = LnEDivRule;

    let x_over_e = parse("ln(x/e)", &mut ctx).unwrap();
    let x_over_e_rewrite = rule
        .apply(
            &mut ctx,
            x_over_e,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: x_over_e_rewrite.new_expr
            }
        ),
        "ln(x) - 1"
    );

    let e_over_x = parse("ln(e/x)", &mut ctx).unwrap();
    let e_over_x_rewrite = rule
        .apply(
            &mut ctx,
            e_over_x,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: e_over_x_rewrite.new_expr
            }
        ),
        "1 - ln(x)"
    );
}

#[test]
fn test_log_base_base() {
    let mut ctx = Context::new();
    let rule = EvaluateLogRule;
    // log(x, x) -> 1
    let expr = parse("log(x, x)", &mut ctx).unwrap();
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
}

#[test]
fn test_log_inverse() {
    let mut ctx = Context::new();
    let rule = EvaluateLogRule;
    // log(x, x^2) -> 2
    let expr = parse("log(x, x^2)", &mut ctx).unwrap();
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
        "2"
    );
}

#[test]
fn test_log_expansion() {
    let mut ctx = Context::new();
    let rule = EvaluateLogRule;
    // log(b, x^y) -> y * log(b, x)
    let expr = parse("log(2, x^3)", &mut ctx).unwrap();
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
        "3 * log(2, x)"
    );
}

#[test]
fn test_log_product() {
    let mut ctx = Context::new();
    let rule = LogExpansionRule;
    // log(b, x*y) -> log(b, x) + log(b, y) (requires Assume mode for variables)
    let expr = parse("log(2, x * y)", &mut ctx).unwrap();
    // Create parent context with Assume mode (allows expansion with warning)
    let parent_ctx =
        crate::parent_context::ParentContext::root().with_domain_mode(crate::DomainMode::Assume);
    let rewrite = rule.apply(&mut ctx, expr, &parent_ctx).unwrap();
    let res = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    );
    assert!(res.contains("log(2, x)"));
    assert!(res.contains("log(2, y)"));
    assert!(res.contains("+"));
}

#[test]
fn test_log_quotient() {
    let mut ctx = Context::new();
    let rule = LogExpansionRule;
    // log(b, x/y) -> log(b, x) - log(b, y) (requires Assume mode for variables)
    let expr = parse("log(2, x / y)", &mut ctx).unwrap();
    // Create parent context with Assume mode (allows expansion with warning)
    let parent_ctx =
        crate::parent_context::ParentContext::root().with_domain_mode(crate::DomainMode::Assume);
    let rewrite = rule.apply(&mut ctx, expr, &parent_ctx).unwrap();
    let res = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    );
    assert!(res.contains("log(2, x)"));
    assert!(res.contains("log(2, y)"));
    assert!(res.contains("-"));
}

#[test]
fn test_ln_e() {
    let mut ctx = Context::new();
    let rule = EvaluateLogRule;
    // ln(e) -> 1
    // Note: ln(e) parses to log(e, e)
    let expr = parse("ln(e)", &mut ctx).unwrap();
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
}
