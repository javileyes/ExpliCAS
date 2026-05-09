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
        "x^3 / 3"
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
fn test_integrate_scaled_denominator_square_carries_required_domain() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    let expr = parse("integrate((2*x+1)/(3*(x^2+x-1)^2), x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();

    let required: Vec<_> = rewrite
        .required_conditions
        .iter()
        .map(|condition| condition.display(&ctx))
        .collect();
    assert_eq!(required, vec!["x^2 + x - 1 ≠ 0".to_string()]);
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
        "-1/2 * cos(2 * x)"
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
        "1/3 * e^(3 * x)"
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
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "(2 * x + 1)^3 / (2 * 3)"
    );
}

#[test]
fn test_integrate_linear_subst_log() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    // integrate(1/(3*x), x) -> ln(abs(3*x))/3
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
        "ln(|3 * x|) / 3"
    );
}

#[test]
fn test_integrate_inverse_function_kernels() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;

    let arctan_expr = parse("integrate(1/(x^2+1), x)", &mut ctx).unwrap();
    let arctan_rewrite = rule
        .apply(
            &mut ctx,
            arctan_expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: arctan_rewrite.new_expr
            }
        ),
        "arctan(x)"
    );

    let asinh_expr = parse("integrate((x^2+1)^(-1/2), x)", &mut ctx).unwrap();
    let asinh_rewrite = rule
        .apply(
            &mut ctx,
            asinh_expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: asinh_rewrite.new_expr
            }
        ),
        "asinh(x)"
    );
}

#[test]
fn test_integrate_atanh_surd_open_interval_condition_compacts_to_denominator() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    let expr = parse("integrate(2*x/(3-x^4), x)", &mut ctx).unwrap();
    let rewrite = rule
        .apply(
            &mut ctx,
            expr,
            &crate::parent_context::ParentContext::root(),
        )
        .unwrap();

    assert_eq!(rewrite.required_conditions.len(), 1);
    assert_eq!(rewrite.required_conditions[0].display(&ctx), "3 - x^4 > 0");
}

#[test]
fn test_integrate_secant_squared_kernel() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    let expr = parse("integrate(1/cos(x)^2, x)", &mut ctx).unwrap();
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
    assert_eq!(rewrite.required_conditions.len(), 1);
    assert_eq!(rewrite.required_conditions[0].display(&ctx), "cos(x) ≠ 0");
}

#[test]
fn test_integrate_cosecant_squared_kernel() {
    let mut ctx = Context::new();
    let rule = IntegrateRule;
    let expr = parse("integrate(1/sin(x)^2, x)", &mut ctx).unwrap();
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
        "-cos(x) / sin(x)"
    );
    assert_eq!(rewrite.required_conditions.len(), 1);
    assert_eq!(rewrite.required_conditions[0].display(&ctx), "sin(x) ≠ 0");
}
