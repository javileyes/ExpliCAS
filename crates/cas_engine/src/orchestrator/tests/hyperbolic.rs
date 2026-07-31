//! Tests del orquestador: familia `hyperbolic` (troceo P1).

use super::*;

#[test]
fn simplify_pipeline_handles_nested_additive_hyperbolic_cubic_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_nested_additive_hyperbolic_cubic_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) - (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_nested_additive_hyperbolic_cubic_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_handles_atanh_ln_definition_gap_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "ln(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn child_isolated_exact_zero_handles_hyperbolic_pythagorean_residual_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(child_isolated_exact_zero(
        &options,
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn collapsed_fraction_hyperbolic_half_angle_factor_shortcut_matches_regression() {
    let mut ctx = Context::new();
    let expr = parse("((1/x + 1/(x+1)) * (sinh(x/2)^2))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_collapsed_fraction_hyperbolic_half_angle_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "collapsed-fraction hyperbolic half-angle shortcut should match"
    );
}
#[test]
fn simplify_pipeline_handles_collapsed_fraction_times_hyperbolic_half_angle_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let raw = parse("((1/x + 1/(x+1)) * (sinh(x/2)^2))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (raw_result, _steps, _stats) = orchestrator.simplify_pipeline(raw, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, raw_result),
        "((2 * x + 1) * (cosh(x) - 1))/(x * (x + 1) * 2)"
    );
}
#[test]
fn detects_direct_hyperbolic_sinh_sum_to_product_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("sinh(x) + sinh(y)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("2*sinh((x+y)/2)*cosh((x-y)/2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_hyperbolic_sinh_sum_to_product_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}
#[test]
fn simplify_pipeline_handles_hyperbolic_sinh_sum_to_product_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sinh(x)+sinh(y)) + m) - ((2*sinh((x+y)/2)*cosh((x-y)/2)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_hyperbolic_sinh_sum_to_product_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sinh(x)+sinh(y)) + 1)/((2*sinh((x+y)/2)*cosh((x-y)/2)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn detects_direct_hyperbolic_cosh_sum_to_product_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("cosh(x) + cosh(y)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("2*cosh((x+y)/2)*cosh((x-y)/2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_hyperbolic_cosh_sum_to_product_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}
#[test]
fn simplify_pipeline_handles_hyperbolic_cosh_sum_to_product_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cosh(x)+cosh(y)) + m) - ((2*cosh((x+y)/2)*cosh((x-y)/2)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_hyperbolic_cosh_sum_to_product_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cosh(x)+cosh(y)) + 1)/((2*cosh((x+y)/2)*cosh((x-y)/2)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn detects_direct_hyperbolic_cosh_difference_to_product_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("cosh(x) - cosh(y)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("2*sinh((x+y)/2)*sinh((x-y)/2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        matches_direct_hyperbolic_cosh_difference_to_product_pair_root(
            &mut simplifier.context,
            lhs,
            rhs
        )
    );
}
#[test]
fn simplify_pipeline_handles_hyperbolic_cosh_difference_to_product_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cosh(x)-cosh(y)) + m) - ((2*sinh((x+y)/2)*sinh((x-y)/2)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_direct_recursive_hyperbolic_sinh_sum_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("sinh(6*x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse(
        "sinh(5*x)*cosh(x)+cosh(5*x)*sinh(x)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_recursive_hyperbolic_sinh_sum_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}
#[test]
fn simplify_pipeline_handles_recursive_hyperbolic_sinh_sum_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sinh(6*x)) + m) - ((sinh(5*x)*cosh(x)+cosh(5*x)*sinh(x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_recursive_hyperbolic_sinh_sum_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sinh(6*x)) + 1)/((sinh(5*x)*cosh(x)+cosh(5*x)*sinh(x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn detects_direct_recursive_hyperbolic_cosh_sum_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("cosh(6*x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse(
        "cosh(5*x)*cosh(x)+sinh(5*x)*sinh(x)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_recursive_hyperbolic_cosh_sum_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}
#[test]
fn simplify_pipeline_handles_recursive_hyperbolic_cosh_sum_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cosh(6*x)) + m) - ((cosh(5*x)*cosh(x)+sinh(5*x)*sinh(x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_recursive_hyperbolic_cosh_sum_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cosh(6*x)) + 1)/((cosh(5*x)*cosh(x)+sinh(5*x)*sinh(x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_handles_hyperbolic_cosh_cubic_passthrough_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sinh(2*x)*sinh(x)+a) + 1)/((4*cosh(x)^3-4*cosh(x)+a) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn perfect_square_trinomial_factor_shortcut_matches_tanh_partner_regression() {
    let mut ctx = Context::new();
    let expr = parse("(9*x^2 - 6*x + 1) * tanh(u)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let Some((result, _)) = super::try_standard_perfect_square_trinomial_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    ) else {
        panic!("perfect-square factor shortcut should match tanh partners");
    };

    let rendered = render_expr(&ctx, result);
    assert!(rendered.contains("(3 * x - 1)^2") || rendered.contains("(1 - 3 * x)^2"));
    assert!(
        rendered.contains("tanh(u)")
            || rendered.contains("sinh(u)")
            || rendered.contains("cosh(u)")
    );
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_hyperbolic_exp_ratio_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "(cot(5*pi/12)) * ((exp(x)-exp(-x))/(exp(x)+exp(-x)))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_special_angle_exact_value_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "special-angle exact-value factor shortcut should match hyperbolic exp-ratio products"
    );
}
#[test]
fn detects_direct_hyperbolic_exp_sum_pair_regression() {
    let mut ctx = Context::new();
    let lhs =
        parse("cosh(u) - sinh(u)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("exp(-u)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_hyperbolic_exp_sum_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_hyperbolic_half_angle_square_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("sinh(x/2)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("(cosh(x)-1)/2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_hyperbolic_half_angle_square_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_two_factor_product_pair_zero_difference_hyperbolic_half_angle_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((1/x + 1/(x+1)) * (sinh(x/2)^2)) - (((2*x+1)/(x*(x+1))) * ((cosh(x)-1)/2))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let view = AddView::from_expr(&ctx, expr);
    let lhs_factor_ids = flatten_mul_chain(&mut ctx, view.terms[0].0);
    let rhs_factor_ids = flatten_mul_chain(&mut ctx, view.terms[1].0);
    let lhs_factors = lhs_factor_ids
        .iter()
        .copied()
        .map(|factor| render(&ctx, factor))
        .collect::<Vec<_>>();
    let rhs_factors = rhs_factor_ids
        .iter()
        .copied()
        .map(|factor| render(&ctx, factor))
        .collect::<Vec<_>>();
    assert!(
        matches_direct_two_factor_product_pair_zero_difference_root(&mut ctx, expr),
        "lhs factors = {:?}, rhs factors = {:?}",
        lhs_factors,
        rhs_factors
    );
}
#[test]
fn detects_direct_hyperbolic_pythagorean_pair_regression() {
    let mut ctx = Context::new();
    let lhs =
        parse("cosh(z)^2 - sinh(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_hyperbolic_pythagorean_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_hyperbolic_triple_angle_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("sinh(3*z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("3*sinh(z) + 4*sinh(z)^3", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_hyperbolic_triple_angle_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_nested_zero_hyperbolic_triple_angle_residual_regression() {
    let mut ctx = Context::new();
    let expr = parse("4*cosh(x)^3 - 3*cosh(x) - cosh(3*x)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_nested_zero_hyperbolic_triple_angle_residual_pair_root(
            &mut ctx, expr
        )
    );
}
#[test]
fn rejects_direct_nested_zero_hyperbolic_triple_angle_residual_on_pure_trig_ratio_regression() {
    let mut ctx = Context::new();
    let expr = parse("sin(2*x)/cos(2*x) - tan(2*x)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        !super::matches_direct_nested_zero_hyperbolic_triple_angle_residual_pair_root(
            &mut ctx, expr
        )
    );
}
#[test]
fn rejects_direct_nested_zero_hyperbolic_triple_angle_residual_on_double_angle_regression() {
    let mut ctx = Context::new();
    let expr = parse("cosh(2*x) - (2*cosh(x)^2 - 1)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        !super::matches_direct_nested_zero_hyperbolic_triple_angle_residual_pair_root(
            &mut ctx, expr
        )
    );
}
#[test]
fn detects_direct_hyperbolic_from_exp_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("(exp(z)-exp(-z))/(exp(z)+exp(-z))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("tanh(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_hyperbolic_from_exp_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_tanh_to_sinh_cosh_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("tanh(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("sinh(z)/cosh(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_tanh_to_sinh_cosh_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_hyperbolic_double_angle_sum_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("cosh(2*z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs =
        parse("cosh(z)^2 + sinh(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_hyperbolic_double_angle_sum_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_tanh_pythagorean_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("1 - tanh(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("1/cosh(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_tanh_pythagorean_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_two_factor_product_pair_zero_difference_special_angle_hyperbolic_exp_ratio_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((cot(5*pi/12)) * ((exp(x)-exp(-x))/(exp(x)+exp(-x)))) - (((2 - 3^(1/2))) * tanh(x))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn detects_two_factor_product_pair_zero_difference_special_angle_hyperbolic_double_angle_regression(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "((cot(5*pi/12)) * (cosh(2*x))) - (((2 - 3^(1/2))) * (cosh(x)^2 + sinh(x)^2))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn detects_two_factor_product_pair_zero_difference_special_angle_hyperbolic_triple_angle_sqrt_form_regression(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "((cot(5*pi/12)) * (sinh(3*x))) - (((2 - sqrt(3))) * (3*sinh(x) + 4*sinh(x)^3))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn simplify_pipeline_handles_special_angle_cot_times_hyperbolic_triple_angle_sqrt_form_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cot(5*pi/12)) * (sinh(3*x))) - (((2 - sqrt(3))) * (3*sinh(x) + 4*sinh(x)^3))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_hyperbolic_anchor_times_geometric_sum_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((exp(x) - exp(-x))/2) * (u^3 + u^2 + u + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("(sinh(x)) * ((u+1)*(u^2+1))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
}
#[test]
fn detects_two_factor_product_pair_zero_difference_special_angle_tanh_pythagorean_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((cot(5*pi/12)) * (1/cosh(x)^2)) - (((2 - 3^(1/2))) * (1 - tanh(x)^2))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn detects_two_factor_product_pair_zero_difference_perfect_square_tanh_fraction_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((9*x^2 - 6*x + 1) * tanh(u)) - (((3*x - 1)^2) * (sinh(u)/cosh(u)))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn simplify_pipeline_handles_quartic_gcf_times_hyperbolic_triple_angle_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((z^4 - z^2) * (sinh(3*u))) - (((z^2*(z-1)*(z+1))) * (3*sinh(u) + 4*sinh(u)^3))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_direct_quotient_pair_zero_difference_tanh_half_angle_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "(tanh(2*x))/(abs(sin(x/2))) - ((2*tanh(x)/(1+tanh(x)^2))/(sqrt((1-cos(x))/2)) )",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_quotient_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn simplify_pipeline_handles_half_angle_against_hyperbolic_sinh_cubic_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(x)^2 - (1 - cos(2*x))/2) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_binomial_square_against_exp_hyperbolic_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(cosh(x) + sinh(x) - e^x) + ((sin(x) + cos(x))^2 - (1 + sin(2*x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_binomial_square_against_exp_hyperbolic_product_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(cosh(x) + sinh(x) - e^x) * ((sin(x) + cos(x))^2 - (1 + sin(2*x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_binomial_square_against_exp_hyperbolic_shifted_quotient_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cosh(x) + sinh(x) - e^x) + 1)/(((sin(x) + cos(x))^2 - (1 + sin(2*x))) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_handles_exp_hyperbolic_against_hyperbolic_sinh_cubic_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(cosh(x) + sinh(x) - e^x) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_exp_hyperbolic_against_hyperbolic_sinh_cubic_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(cosh(x) + sinh(x) - e^x) - (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_phase_shift_against_hyperbolic_cosh_cubic_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_phase_shift_against_hyperbolic_cosh_cubic_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))) - (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_hyperbolic_sum_against_hyperbolic_cosh_cubic_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))) + (sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_hyperbolic_sum_against_hyperbolic_cosh_cubic_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))) - (sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_hyperbolic_sum_against_reciprocal_trig_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))) + 1)/((tan(x) + cot(x) - sec(x)*csc(x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_handles_hyperbolic_cosh_cubic_against_telescoping_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_hyperbolic_sum_against_telescoping_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_binomial_square_against_hyperbolic_pythagorean_product_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x) + cos(x))^2 - (1 + sin(2*x))) * (cosh(x)^2 - sinh(x)^2 - 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_binomial_square_against_hyperbolic_pythagorean_shifted_quotient_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((sin(x) + cos(x))^2 - (1 + sin(2*x))) + 1)/((cosh(x)^2 - sinh(x)^2 - 1) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_handles_trig_binomial_square_against_exp_cosh_product_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x) + cos(x))^2 - (1 + sin(2*x))) * (exp(x) + exp(-x) - 2*cosh(x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_binomial_square_against_exp_cosh_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((sin(x) + cos(x))^2 - (1 + sin(2*x))) + 1)/((exp(x) + exp(-x) - 2*cosh(x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn nested_fraction_zero_hyperbolic_identity_pair_shortcut_handles_pressure_hotspot_regressions() {
    for hyperbolic_side in [
        "sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))",
        "cosh(x) + sinh(x) - e^x",
        "exp(x) - exp(-x) - 2*sinh(x)",
        "exp(x) + exp(-x) - 2*cosh(x)",
        "cosh(x)^2 - sinh(x)^2 - 1",
        "2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))",
    ] {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let input =
            format!("(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + ({hyperbolic_side})");
        let expr = parse(&input, &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (rewritten, steps) =
            super::try_standard_nested_fraction_zero_hyperbolic_identity_pair_shortcut(
                &mut simplifier.context,
                expr,
                false,
            )
            .unwrap_or_else(|| {
                panic!("expected nested-fraction hyperbolic identity shortcut for {input}")
            });

        assert_eq!(
            render(&simplifier.context, rewritten),
            "0",
            "input: {input}"
        );
        assert!(steps.is_empty());
    }
}
#[test]
fn log_zero_hyperbolic_cosh_cubic_pair_shortcut_handles_pressure_hotspot_regressions() {
    for log_side in [
        "ln((x*y)^2) - ln(x^2) - ln(y^2)",
        "ln(x^3) + ln(y^2) - ln(x^3 * y^2)",
    ] {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let input = format!("({log_side}) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))");
        let expr = parse(&input, &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let (rewritten, steps) = super::try_standard_log_zero_hyperbolic_cosh_cubic_pair_shortcut(
            &mut simplifier.context,
            expr,
            false,
        )
        .unwrap_or_else(|| panic!("expected log-zero hyperbolic cosh-cubic shortcut for {input}"));

        assert_eq!(
            render(&simplifier.context, rewritten),
            "0",
            "input: {input}"
        );
        assert!(steps.is_empty());
    }
}
#[test]
fn targeted_direct_small_zero_additive_combination_collapses_log_square_hyperbolic_cubic_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, steps) =
        super::try_standard_targeted_direct_small_zero_additive_combination_shortcut(
            &mut simplifier,
            expr,
            false,
        )
        .unwrap_or_else(|| panic!("expected targeted direct small-zero shortcut"));

    assert_eq!(render(&simplifier.context, rewritten), "0");
    assert!(steps.is_empty());
}
#[test]
fn detects_direct_hyperbolic_cosh_cubic_zero_identity_base_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_small_zero_or_known_pair_base_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn simplify_pipeline_handles_small_mixed_trig_hyperbolic_zero_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_small_mixed_trig_hyperbolic_zero_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))) - (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn small_trig_zero_pair_shortcut_handles_triple_sine_quotient_against_hyperbolic_pythagorean_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) =
        try_standard_small_trig_zero_pair_shortcut(&options, &mut simplifier.context, expr, false)
            .unwrap_or_else(|| panic!("expected small trig zero pair shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn small_trig_zero_pair_shortcut_handles_triple_sine_plus_rational_against_hyperbolic_pythagorean_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) =
        try_standard_small_trig_zero_pair_shortcut(&options, &mut simplifier.context, expr, false)
            .unwrap_or_else(|| panic!("expected small trig zero pair shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_triple_sine_quotient_against_hyperbolic_pythagorean_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn nested_exact_zero_child_shortcut_handles_triple_sine_plus_rational_against_hyperbolic_pythagorean_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_nested_exact_zero_child_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected nested exact-zero child shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_rational_against_hyperbolic_pythagorean_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn supported_nested_zero_partner_rewrites_to_zero_handles_atanh_grouped_log_plus_exp_log_partner_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (log((x*y)^2) - 2*log(x) - 2*log(y)) + (exp(z*log(w)) - w^z)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(supported_nested_zero_partner_rewrites_to_zero(
        &options,
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn supported_nested_zero_partner_rewrites_to_zero_handles_atanh_grouped_log_plus_exp_log_plus_fraction_partner_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(atanh((u^2 - 1)/(u^2 + 1)) - log(u)) + (log((p*q)^2) - 2*log(p) - 2*log(q)) + (exp(r*log(s)) - s^r) + (2/(t^2 - 1) - 1/(t-1) + 1/(t+1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(supported_nested_zero_partner_rewrites_to_zero(
        &options,
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn supported_nested_zero_partner_rewrites_to_zero_handles_root_denesting_plus_atanh_grouped_log_exp_fraction_partner_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(m + sqrt(m^2 - n^2)) - (sqrt(m+n) + sqrt(m-n))/sqrt(2)) + (atanh((u^2 - 1)/(u^2 + 1)) - log(u)) + (log((p*q)^2) - 2*log(p) - 2*log(q)) + (exp(r*log(s)) - s^r) + (2/(t^2 - 1) - 1/(t-1) + 1/(t+1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(supported_nested_zero_partner_rewrites_to_zero(
        &options,
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn classify_multiterm_trig_numeric_subset_status_is_none_on_raw_polynomial_hyperbolic_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert_eq!(
        classify_root_exact_zero_multiterm_trig_numeric_subset_status(
            &options,
            &mut simplifier.context,
            expr
        ),
        Some("candidate_ready")
    );
}
#[test]
fn simplify_pipeline_handles_polynomial_against_hyperbolic_pythagorean_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn engine_simplify_steps_off_handles_triple_sine_plus_rational_against_hyperbolic_pythagorean_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    simplifier.set_collect_steps(false);
    let expr = parse(
        "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut options = SimplifyOptions::default();
    options.shared.context_mode = crate::options::ContextMode::Standard;
    options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps) = simplifier.simplify_with_options(expr, options);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn child_isolated_exact_zero_handles_original_hyperbolic_pythagorean_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(is_potential_small_trig_zero_identity_root(
        &simplifier.context,
        expr
    ));
    assert!(child_isolated_exact_zero(
        &options,
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn child_isolated_exact_zero_handles_rational_against_hyperbolic_pythagorean_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(child_isolated_exact_zero(
        &options,
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn child_isolated_exact_zero_handles_triple_sine_against_hyperbolic_pythagorean_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(child_isolated_exact_zero(
        &options,
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn classify_multiterm_trig_numeric_subset_status_is_candidate_ready_on_triple_sine_against_polynomial_plus_hyperbolic_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut options = SimplifyOptions::default();
    options.shared.context_mode = crate::options::ContextMode::Standard;
    options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    assert_eq!(
        classify_root_exact_zero_multiterm_trig_numeric_subset_status(
            &options,
            &mut simplifier.context,
            expr
        ),
        Some("candidate_ready")
    );
}
#[test]
fn multiterm_trig_numeric_subset_zero_shortcut_handles_triple_sine_against_polynomial_plus_hyperbolic_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut options = SimplifyOptions::default();
    options.shared.context_mode = crate::options::ContextMode::Standard;
    options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps) = try_standard_multiterm_trig_numeric_subset_zero_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected multiterm trig-numeric subset shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_triple_sine_against_polynomial_plus_hyperbolic_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn classify_multiterm_trig_numeric_subset_status_is_candidate_ready_on_triple_sine_against_log_plus_hyperbolic_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (ln(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y))) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut options = SimplifyOptions::default();
    options.shared.context_mode = crate::options::ContextMode::Standard;
    options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    assert_eq!(
        classify_root_exact_zero_multiterm_trig_numeric_subset_status(
            &options,
            &mut simplifier.context,
            expr
        ),
        Some("candidate_ready")
    );
}
#[test]
fn multiterm_trig_numeric_subset_zero_shortcut_handles_triple_sine_against_log_plus_hyperbolic_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (ln(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y))) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut options = SimplifyOptions::default();
    options.shared.context_mode = crate::options::ContextMode::Standard;
    options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps) = try_standard_multiterm_trig_numeric_subset_zero_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected multiterm trig-numeric subset shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_triple_sine_against_log_plus_hyperbolic_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (ln(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y))) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn multiterm_trig_numeric_subset_zero_shortcut_handles_triple_sine_against_atanh_grouped_log_plus_exp_log_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (log((x*y)^2) - 2*log(x) - 2*log(y)) + (exp(z*log(w)) - w^z) + (sin(a) + sin(b) - 2*sin((a+b)/2)*cos((a-b)/2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut options = SimplifyOptions::default();
    options.shared.context_mode = crate::options::ContextMode::Standard;
    options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps) = try_standard_multiterm_trig_numeric_subset_zero_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected multiterm trig-numeric subset shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_triple_sine_against_atanh_grouped_log_plus_exp_log_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (log((x*y)^2) - 2*log(x) - 2*log(y)) + (exp(z*log(w)) - w^z) + (sin(a) + sin(b) - 2*sin((a+b)/2)*cos((a-b)/2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn multiterm_trig_numeric_subset_zero_shortcut_handles_triple_sine_against_atanh_grouped_log_exp_fraction_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(atanh((u^2 - 1)/(u^2 + 1)) - log(u)) + (log((p*q)^2) - 2*log(p) - 2*log(q)) + (exp(r*log(s)) - s^r) + (sin(a) + sin(b) - 2*sin((a+b)/2)*cos((a-b)/2)) + (2/(t^2 - 1) - 1/(t-1) + 1/(t+1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut options = SimplifyOptions::default();
    options.shared.context_mode = crate::options::ContextMode::Standard;
    options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps) = try_standard_multiterm_trig_numeric_subset_zero_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected multiterm trig-numeric subset shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn matches_direct_small_zero_identity_root_handles_atanh_square_ratio_log_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "atanh((x^2 - 1)/(x^2 + 1)) - log(x)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn direct_small_zero_identity_shortcut_handles_atanh_square_ratio_log_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "atanh((x^2 - 1)/(x^2 + 1)) - log(x)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, steps) = try_standard_direct_small_zero_identity_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        expr,
        true,
    )
    .unwrap_or_else(|| panic!("expected direct small-zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0].rule_name,
        "Collapse Exact Zero Additive Subexpression"
    );
}
#[test]
fn atanh_square_ratio_log_zero_shortcut_handles_two_term_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "atanh((x^2 - 1)/(x^2 + 1)) - log(x)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, steps) =
        try_standard_atanh_square_ratio_log_zero_shortcut(&mut simplifier.context, expr, true)
            .unwrap_or_else(|| panic!("expected direct atanh-log zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
    assert_eq!(steps.len(), 1);
    assert_eq!(
        steps[0].rule_name,
        "Collapse Exact Zero Additive Subexpression"
    );
}
#[test]
fn atanh_square_ratio_log_subset_zero_shortcut_handles_atanh_square_ratio_plus_grouped_log_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (log((x*y)^2) - 2*log(x) - 2*log(y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_atanh_square_ratio_log_subset_zero_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected atanh-log subset zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_atanh_square_ratio_plus_grouped_log_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (log((x*y)^2) - 2*log(x) - 2*log(y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn symbolic_root_denesting_subset_zero_shortcut_handles_root_denesting_against_atanh_plus_grouped_log_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(x + sqrt(x^2 - y^2)) - (sqrt(x+y) + sqrt(x-y))/sqrt(2)) + (atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (log((x*y)^2) - 2*log(x) - 2*log(y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_symbolic_root_denesting_subset_zero_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected symbolic root denesting subset zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_root_denesting_against_atanh_plus_grouped_log_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(x + sqrt(x^2 - y^2)) - (sqrt(x+y) + sqrt(x-y))/sqrt(2)) + (atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (log((x*y)^2) - 2*log(x) - 2*log(y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn symbolic_root_denesting_subset_zero_shortcut_handles_root_denesting_against_atanh_grouped_log_exp_fraction_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(m + sqrt(m^2 - n^2)) - (sqrt(m+n) + sqrt(m-n))/sqrt(2)) + (atanh((u^2 - 1)/(u^2 + 1)) - log(u)) + (log((p*q)^2) - 2*log(p) - 2*log(q)) + (exp(r*log(s)) - s^r) + (2/(t^2 - 1) - 1/(t-1) + 1/(t+1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_symbolic_root_denesting_subset_zero_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected symbolic root denesting subset zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_root_denesting_against_atanh_grouped_log_exp_fraction_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(m + sqrt(m^2 - n^2)) - (sqrt(m+n) + sqrt(m-n))/sqrt(2)) + (atanh((u^2 - 1)/(u^2 + 1)) - log(u)) + (log((p*q)^2) - 2*log(p) - 2*log(q)) + (exp(r*log(s)) - s^r) + (2/(t^2 - 1) - 1/(t-1) + 1/(t+1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn sqrt_perfect_square_abs_subset_zero_shortcut_handles_shifted_root_square_against_atanh_plus_grouped_log_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(x + 2*sqrt(x-1)) - sqrt(x-1) - 1) + (atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (log((x*y)^2) - 2*log(x) - 2*log(y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_sqrt_perfect_square_abs_subset_zero_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected sqrt perfect-square abs subset zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_shifted_root_square_against_atanh_plus_grouped_log_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(x + 2*sqrt(x-1)) - sqrt(x-1) - 1) + (atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (log((x*y)^2) - 2*log(x) - 2*log(y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn inverse_trig_composition_subset_zero_shortcut_handles_arcsin_atan_against_atanh_plus_grouped_log_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(asin(x/sqrt(x^2 + 1)) - atan(x)) + (atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (log((x*y)^2) - 2*log(x) - 2*log(y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_inverse_trig_composition_subset_zero_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected inverse trig composition subset zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn inverse_trig_plus_sqrt_subset_zero_shortcut_handles_shifted_root_square_against_atanh_plus_grouped_log_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(x + 2*sqrt(x-1)) - sqrt(x-1) - 1) + (asin(x/sqrt(x^2 + 1)) - atan(x)) + (atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (log((x*y)^2) - 2*log(x) - 2*log(y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_inverse_trig_plus_sqrt_subset_zero_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected inverse trig + sqrt subset zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_shifted_root_square_inverse_trig_against_atanh_plus_grouped_log_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(x + 2*sqrt(x-1)) - sqrt(x-1) - 1) + (asin(x/sqrt(x^2 + 1)) - atan(x)) + (atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (log((x*y)^2) - 2*log(x) - 2*log(y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_shifted_root_square_inverse_trig_triple_sine_exp_log_against_atanh_plus_grouped_log_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(x + 2*sqrt(x-1)) - sqrt(x-1) - 1) + (asin(x/sqrt(x^2 + 1)) - atan(x)) + (atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (log((x*y)^2) - 2*log(x) - 2*log(y)) + (exp(z*log(w)) - w^z) + (sin(a) + sin(b) - 2*sin((a+b)/2)*cos((a-b)/2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn shifted_quotient_shortcut_keeps_tanh_ratio_exact_one_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((tanh(x)) + 1)/((((e^x - e^(-x))/(e^x + e^(-x)))) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_shifted_quotient_exact_one_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected shifted quotient exact-one shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn shifted_quotient_shortcut_handles_trig_mixed_against_exp_sinh_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))) + 1)/((exp(x) - exp(-x) - 2*sinh(x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_shifted_quotient_nested_zero_core_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected shifted quotient shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn shifted_quotient_shortcut_handles_hyperbolic_pythagorean_residual_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cosh(x)^2 - 1) + 1)/((sinh(x)^2) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_shifted_quotient_nested_zero_core_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected shifted quotient shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn shifted_quotient_direct_small_zero_hot_gate_matches_nested_fraction_vs_hyperbolic_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (numerator, denominator) = match simplifier.context.get(expr).clone() {
        Expr::Div(numerator, denominator) => (numerator, denominator),
        _ => panic!("expected division root"),
    };
    let numerator_core = strip_positive_one_passthrough_root(&mut simplifier.context, numerator)
        .unwrap_or_else(|| panic!("expected numerator core"));
    let denominator_core =
        strip_positive_one_passthrough_root(&mut simplifier.context, denominator)
            .unwrap_or_else(|| panic!("expected denominator core"));

    assert!(matches_shifted_quotient_direct_small_zero_hot_gate_root(
        &mut simplifier.context,
        numerator_core,
        denominator_core,
    ));
}
#[test]
fn shifted_quotient_shortcut_handles_hyperbolic_sinh_double_angle_residual_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sinh(x)*cosh(x) + a) + 1)/((sinh(2*x) + a) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_shifted_quotient_nested_zero_core_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected shifted quotient shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn shifted_quotient_shortcut_handles_hyperbolic_cosh_double_angle_square_residual_difference_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cosh(2*x)) + 1)/(((2*cosh(x)^2 - 1)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_shifted_quotient_nested_zero_core_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected shifted quotient shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn shifted_quotient_shortcut_handles_hyperbolic_sinh_angle_difference_residual_difference_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sinh(x-y)) + 1)/((sinh(x)*cosh(y) - sinh(y)*cosh(x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_shifted_quotient_nested_zero_core_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected shifted quotient shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn shifted_quotient_shortcut_handles_hyperbolic_cosh_triple_angle_residual_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((4*cosh(x)^3 - 3*cosh(x)) + 1)/((cosh(3*x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_shifted_quotient_nested_zero_core_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected shifted quotient shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn shifted_quotient_shortcut_handles_hyperbolic_sinh_sum_residual_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sinh(x+y)) + 1)/((sinh(x)*cosh(y) + cosh(x)*sinh(y)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_shifted_quotient_nested_zero_core_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected shifted quotient shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn shifted_quotient_shortcut_handles_hyperbolic_tanh_double_angle_residual_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*tanh(x)/(1+tanh(x)^2)) + 1)/((tanh(2*x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_shifted_quotient_nested_zero_core_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected shifted quotient shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_handles_contextual_tanh_square_composition_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((9*u^2 - 6*u + 1) + tanh(2*v)) - (((3*u - 1)^2) + (2*tanh(v)/(1 + tanh(v)^2)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_contextual_multivariate_tanh_composition_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((x^2 + y^2)*(a^2 + b^2)) + tanh(2*u)) - (((x*a + y*b)^2 + (x*b - y*a)^2) + (2*tanh(u)/(1 + tanh(u)^2)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_direct_tanh_double_angle_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("tanh(2*v)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("2*tanh(v)/(1 + tanh(v)^2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_tanh_double_angle_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}
#[test]
fn detects_composed_small_additive_tanh_square_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("(9*u^2 - 6*u + 1) + tanh(2*v)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse(
        "((3*u - 1)^2) + (2*tanh(v)/(1 + tanh(v)^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_composed_small_additive_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}
#[test]
fn small_composed_additive_pair_shortcut_handles_contextual_tanh_square_composition_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((9*u^2 - 6*u + 1) + tanh(2*v)) - (((3*u - 1)^2) + (2*tanh(v)/(1 + tanh(v)^2)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _steps) = super::try_standard_small_composed_additive_pair_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected small composed additive shortcut"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_cubic_against_hyperbolic_cubic_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(2*sin(2*x)*sin(x) - (4*cos(x) - 4*cos(x)^3)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_root_shortcut_collapses_affine_hyperbolic_pythagorean_factor_residuals() {
    for input in [
        "sinh(2*x+1)*(cosh(2*x+1)^2 - 1) - sinh(2*x+1)^3",
        "cosh(2*x+1)*(1 + sinh(2*x+1)^2) - cosh(2*x+1)^3",
    ] {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr =
            parse(input, &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        orchestrator.options.collect_steps = false;
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(
            render(&simplifier.context, rewritten),
            "0",
            "input: {input}"
        );
    }
}
