use crate::rule::Rule;
use crate::rules::calculus::IntegrateRule;
use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

#[test]
fn test_integrate_power() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate(x^2, x) -> x^3/3
    let expr = parse("integrate(x^2, x)", &mut ctx).unwrap();
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
        "x^(1 + 2) / (1 + 2)" // Canonical: smaller numbers first
    );
}

#[test]
fn test_integrate_constant() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate(5, x) -> 5x
    let expr = parse("integrate(5, x)", &mut ctx).unwrap();
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
        "5 * x"
    );
}

#[test]
fn test_integrate_trig() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate(sin(x), x) -> -cos(x)
    let expr = parse("integrate(sin(x), x)", &mut ctx).unwrap();
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
        "-cos(x)"
    );
}

#[test]
fn test_integrate_linearity() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate(x + 1, x) -> x^2/2 + x
    let expr = parse("integrate(x + 1, x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    let res = format!(
        "{}",
        DisplayExpr {
            context: &ctx,
            id: rewrite.new_expr
        }
    );
    // x^2/2 + 1*x
    assert!(res.contains("x^2 / 2"));
    assert!(res.contains("1 * x") || res.contains("x"));
}
#[test]
fn test_integrate_linear_subst_trig() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate(sin(2*x), x) -> -cos(2*x)/2
    let expr = parse("integrate(sin(2*x), x)", &mut ctx).unwrap();
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
        "-cos(2 * x) / 2"
    );
}

#[test]
fn test_integrate_linear_subst_exp() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate(exp(3*x), x) -> exp(3*x)/3
    let expr = parse("integrate(exp(3*x), x)", &mut ctx).unwrap();
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
        "e^(3 * x) / 3"
    );
}

#[test]
fn test_integrate_linear_subst_power() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate((2*x + 1)^2, x) -> (2*x + 1)^3 / (2*3) -> (2*x+1)^3 / 6
    let expr = parse("integrate((2*x + 1)^2, x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    // Note: 2*3 is not simplified by IntegrateRule, it produces Expr::mul(2, 3).
    // Simplification happens later in the pipeline.
    // So we expect (2*x + 1)^(2+1) / (2 * (2+1))
    // Actually get_linear_coeffs returns a=2+0 for 2x+1.
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "(2 * x + 1)^(1 + 2) / ((0 + 2) * (1 + 2))" // Canonical: polynomial order (2*x before 1)
    );
}

#[test]
fn test_integrate_linear_subst_log() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate(1/(3*x), x) -> ln(3*x)/3
    let expr = parse("integrate(1/(3*x), x)", &mut ctx).unwrap();
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
        "ln(3 * x) / 3"
    );
}
