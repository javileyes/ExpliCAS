use super::expansion_rules::*;
use crate::rule::Rule;
use crate::rules::trigonometry::identities::{
    AngleIdentityRule, EvaluateTrigRule, TanToSinCosRule,
};
use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

#[test]
fn test_evaluate_trig_zero() {
    let mut ctx = Context::new();
    let rule = EvaluateTrigRule;

    // sin(0) -> 0
    let expr = parse("sin(0)", &mut ctx).unwrap();
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

    // cos(0) -> 1
    let expr = parse("cos(0)", &mut ctx).unwrap();
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

    // tan(0) -> 0
    let expr = parse("tan(0)", &mut ctx).unwrap();
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
fn test_evaluate_trig_identities() {
    let mut ctx = Context::new();
    let rule = EvaluateTrigRule;

    // sin(-x) -> -sin(x)
    let expr = parse("sin(-x)", &mut ctx).unwrap();
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
        "-sin(x)"
    );

    // cos(-x) -> cos(x)
    let expr = parse("cos(-x)", &mut ctx).unwrap();
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
        "cos(x)"
    );

    // tan(-x) -> -tan(x)
    let expr = parse("tan(-x)", &mut ctx).unwrap();
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
        "-tan(x)"
    );
}

#[test]
fn test_trig_identities() {
    let mut ctx = Context::new();
    let rule = AngleIdentityRule;
    // AngleIdentityRule is now gated behind expand_mode (Ticket 6c)
    let expand_ctx = crate::parent_context::ParentContext::root().with_expand_mode_flag(true);

    // sin(x + y)
    let expr = parse("sin(x + y)", &mut ctx).unwrap();
    let rewrite = rule.apply(&mut ctx, expr, &expand_ctx).unwrap();
    assert!(format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    )
    .contains("sin(x)"));

    // cos(x + y) -> cos(x)cos(y) - sin(x)sin(y)
    let expr = parse("cos(x + y)", &mut ctx).unwrap();
    let rewrite = rule.apply(&mut ctx, expr, &expand_ctx).unwrap();
    let res = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    );
    assert!(res.contains("cos(x)"));
    assert!(res.contains("-"));

    // sin(x - y)
    let expr = parse("sin(x - y)", &mut ctx).unwrap();
    let rewrite = rule.apply(&mut ctx, expr, &expand_ctx).unwrap();
    assert!(format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    )
    .contains("-"));
}

#[test]
fn test_tan_to_sin_cos() {
    let mut ctx = Context::new();
    let rule = TanToSinCosRule;
    let expr = parse("tan(x)", &mut ctx).unwrap();
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
        "sin(x) / cos(x)"
    );
}

#[test]
fn test_double_angle() {
    let mut ctx = Context::new();
    let rule = DoubleAngleRule;
    // DoubleAngleRule is now gated behind expand_mode to prevent oscillation
    // with DoubleAngleContractionRule during default simplification.
    let expand_ctx = crate::parent_context::ParentContext::root().with_expand_mode_flag(true);

    // sin(2x)
    let expr = parse("sin(2 * x)", &mut ctx).unwrap();
    let rewrite = rule.apply(&mut ctx, expr, &expand_ctx).unwrap();
    let result_str = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    );
    // Check that result contains the key components, regardless of order
    assert!(
        result_str.contains("sin(x)"),
        "Result should contain sin(x), got: {}",
        result_str
    );
    assert!(
        result_str.contains("cos(x)"),
        "Result should contain cos(x), got: {}",
        result_str
    );
    assert!(
        result_str.contains("2") || result_str.contains("* 2") || result_str.contains("2 *"),
        "Result should contain 2, got: {}",
        result_str
    );

    // cos(2x)
    let expr = parse("cos(2 * x)", &mut ctx).unwrap();
    let rewrite = rule.apply(&mut ctx, expr, &expand_ctx).unwrap();
    assert!(format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    )
    .contains("cos(x)^2 - sin(x)^2"));
}

#[test]
fn test_evaluate_inverse_trig() {
    let mut ctx = Context::new();
    let rule = EvaluateTrigRule;

    // arcsin(0) -> 0
    let expr = parse("arcsin(0)", &mut ctx).unwrap();
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

    // arccos(1) -> 0
    let expr = parse("arccos(1)", &mut ctx).unwrap();
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

    // arcsin(1) -> pi/2
    // Note: pi/2 might be formatted as "pi / 2" or similar depending on Display impl
    let expr = parse("arcsin(1)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert!(format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    )
    .contains("pi"));
    assert!(format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    )
    .contains("2"));

    // arccos(0) -> pi/2
    let expr = parse("arccos(0)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert!(format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    )
    .contains("pi"));
    assert!(format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    )
    .contains("2"));
}
