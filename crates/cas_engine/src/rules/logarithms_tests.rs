use super::logarithms::*;
use crate::rule::Rule;
use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

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
    let parent_ctx = crate::parent_context::ParentContext::root()
        .with_domain_mode(crate::domain::DomainMode::Assume);
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
    let parent_ctx = crate::parent_context::ParentContext::root()
        .with_domain_mode(crate::domain::DomainMode::Assume);
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
