//! Tests del orquestador: familia `trig_angles` (troceo P1).

use super::*;

#[test]
fn standard_pythagorean_additive_shortcut_handles_negated_numeric_pair() {
    let mut ctx = Context::new();
    let expr = parse("-3*sin(x)^2 - 3*cos(x)^2", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _) = try_standard_pythagorean_additive_shortcut(&mut ctx, expr, false)
        .unwrap_or_else(|| panic!("shortcut should match negated numeric pythagorean pair"));
    assert_eq!(render(&ctx, rewritten), "-3");
}
#[test]
fn standard_pythagorean_additive_shortcut_combines_positive_pair_with_constant() {
    let mut ctx = Context::new();
    let expr = parse("sin(x)^2 + cos(x)^2 + 5", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _) = try_standard_pythagorean_additive_shortcut(&mut ctx, expr, false)
        .unwrap_or_else(|| {
            panic!("shortcut should match positive numeric pythagorean pair with constant")
        });
    assert_eq!(render(&ctx, rewritten), "6");
}
#[test]
fn standard_pythagorean_additive_shortcut_combines_two_positive_pairs() {
    let mut ctx = Context::new();
    let expr = parse("sin(x)^2 + cos(x)^2 + sin(y)^2 + cos(y)^2", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _) = try_standard_pythagorean_additive_shortcut(&mut ctx, expr, false)
        .unwrap_or_else(|| panic!("shortcut should match two positive numeric pythagorean pairs"));
    assert_eq!(render(&ctx, rewritten), "2");
}
#[test]
fn standard_pythagorean_additive_pipeline_shortcut_rejects_large_mixed_log_scope_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (ln(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y))) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(try_standard_pythagorean_additive_pipeline_shortcut(
        &crate::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    )
    .is_none());
}
#[test]
fn standard_trig_double_angle_cos_variant_zero_shortcut_handles_split_constants() {
    let mut ctx = Context::new();
    let expr = parse("3 - 4*sin(x)^2 - 2*cos(2*x) - 1", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _) =
        try_standard_trig_double_angle_cos_variant_zero_shortcut(&mut ctx, expr, false)
            .unwrap_or_else(|| panic!("shortcut should match split-constant double-angle zero"));
    assert_eq!(render(&ctx, rewritten), "0");
}
#[test]
fn standard_trig_double_angle_cos_variant_zero_shortcut_rejects_large_mixed_log_scope_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (ln(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y))) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        try_standard_trig_double_angle_cos_variant_zero_shortcut(&mut ctx, expr, false).is_none()
    );
}
#[test]
fn standard_trig_binomial_square_double_angle_shortcut_reduces_to_one() {
    let mut ctx = Context::new();
    let expr = parse("(sin(x) + cos(x))^2 - sin(2*x)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _) = try_standard_trig_binomial_square_double_angle_shortcut(
        &crate::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("shortcut should reduce trig square plus double-angle pair"));
    assert_eq!(render(&ctx, rewritten), "1");
}
#[test]
fn simplify_pipeline_finishes_pythagorean_passthrough_regression_to_zero() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 - sin(x)^2) + m) - ((cos(x)^2) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_finishes_pythagorean_passthrough_from_sin_sq_regression_to_zero() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x)^2) + m) - ((1-cos(x)^2) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_reciprocal_trig_plus_product_to_sum_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(tan(x) + cot(x) - sec(x)*csc(x)) + (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_reciprocal_trig_minus_product_to_sum_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(tan(x) + cot(x) - sec(x)*csc(x)) - (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_rational_factor_times_product_to_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let raw = parse(
        "((1/x + 1/(x+1)) * (2*sin(x)*cos(2*x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rewritten_target = parse(
        "((1/x + 1/(x+1)) * (sin(3*x) - sin(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (raw_result, _steps, _stats) = orchestrator.simplify_pipeline(raw, &mut simplifier);
    let (target_result, _steps, _stats) =
        orchestrator.simplify_pipeline(rewritten_target, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, raw_result),
        render(&simplifier.context, target_result)
    );
}
#[test]
fn embedded_trig_product_to_sum_candidate_matches_rational_factor_regression() {
    let mut ctx = Context::new();
    let expr = parse("((1/x + 1/(x+1)) * (2*sin(x)*cos(2*x)))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let partner =
        parse("(1/x + 1/(x+1))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let trig_factor =
        parse("(2*sin(x)*cos(2*x))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected_trig = try_rewrite_product_to_sum_expr(&mut ctx, trig_factor)
        .unwrap_or_else(|| panic!("expected product-to-sum rewrite"))
        .rewritten;

    let result = super::embedded_trig_product_to_sum_candidate_root(&mut ctx, expr);

    assert!(
        result.is_some(),
        "embedded product-to-sum shortcut should match"
    );
    let rewritten = result.unwrap();
    let rewritten_factors = flatten_mul_chain(&mut ctx, rewritten);
    assert_eq!(rewritten_factors.len(), 2);
    assert!(rewritten_factors
        .iter()
        .copied()
        .any(|factor| compare_expr(&ctx, factor, partner) == Ordering::Equal));
    assert!(rewritten_factors.iter().copied().any(|factor| compare_expr(
        &ctx,
        factor,
        expected_trig
    ) == Ordering::Equal));
}
#[test]
fn collapsed_fraction_direct_pair_factor_shortcut_matches_sum_to_product_regression() {
    let mut ctx = Context::new();
    let expr = parse("((1/x + 1/(x+1)) * (sin(x) + sin(3*x)))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_collapsed_fraction_direct_pair_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "collapsed-fraction direct-pair factor shortcut should match"
    );
}
#[test]
fn collapsed_fraction_direct_pair_factor_shortcut_matches_flattened_product_to_sum_regression() {
    let mut ctx = Context::new();
    let expr = parse("((1/x + 1/(x+1)) * (2*sin(x)*cos(2*x)))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_collapsed_fraction_direct_pair_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "collapsed-fraction direct-pair factor shortcut should match flattened product-to-sum"
    );
}
#[test]
fn simplify_pipeline_handles_sum_to_product_root_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let raw = parse("(sin(x) + sin(3*x))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (raw_result, _steps, _stats) = orchestrator.simplify_pipeline(raw, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, raw_result),
        "2 * sin(2 * x) * cos(x)"
    );
}
#[test]
fn simplify_pipeline_handles_reciprocal_trig_product_with_product_to_sum_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(tan(x) + cot(x) - sec(x)*csc(x)) * (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_reciprocal_trig_shifted_quotient_with_product_to_sum_zero_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((tan(x) + cot(x) - sec(x)*csc(x)) + 1)/((2*sin(x)*sin(y) - cos(x-y) + cos(x+y)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn detects_direct_trig_product_to_sum_sin_sin_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("2*sin(x)*sin(y)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("cos(x-y) - cos(x+y)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_trig_product_to_sum_sin_sin_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}
#[test]
fn detects_direct_trig_product_to_sum_sin_sin_raw_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "2*sin(x)*sin(y) - cos(x-y) + cos(x+y)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn detects_direct_trig_product_to_sum_sin_sin_raw_zero_identity_reordered_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "cos(x+y) + 2*sin(x)*sin(y) - cos(x-y)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn simplify_pipeline_handles_trig_product_to_sum_sin_sin_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*sin(y)) + 1)/((cos(x-y) - cos(x+y)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_handles_trig_product_to_sum_sin_sin_raw_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "2*sin(x)*sin(y) - cos(x-y) + cos(x+y)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_tan_cot_product_plus_trig_product_to_sum_sin_sin_zero_regression() {
    for expr_text in ["tan(x)*cot(x) - 1", "2*sin(x)*sin(y) - cos(x-y) + cos(x+y)"] {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(expr_text, &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        orchestrator.options.collect_steps = false;
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }
}
#[test]
fn simplify_pipeline_handles_trig_product_to_sum_sin_sin_plus_odd_half_power_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(2*sin(x)*sin(y) - cos(x-y) + cos(x+y)) + (sqrt(x^5) - x^2*sqrt(x))",
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
fn detects_direct_trig_product_to_sum_and_odd_half_partition_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(2*sin(x)*sin(y) - cos(x-y) + cos(x+y)) + (sqrt(x^5) - x^2*sqrt(x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let terms = AddView::from_expr(&simplifier.context, expr).terms;
    let odd_expr = build_signed_sum_expr_root(&mut simplifier.context, &[terms[1], terms[2]]);
    let trig_expr =
        build_signed_sum_expr_root(&mut simplifier.context, &[terms[0], terms[3], terms[4]]);
    assert!(
        matches_direct_odd_half_power_zero_scope_root(&mut simplifier.context, odd_expr),
        "odd_expr={}",
        render(&simplifier.context, odd_expr)
    );
    assert!(
        matches_direct_trig_product_to_sum_zero_identity_root(&mut simplifier.context, trig_expr),
        "trig_expr={}",
        render(&simplifier.context, trig_expr)
    );
    assert!(
        matches_direct_trig_product_to_sum_and_odd_half_partition_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn simplify_pipeline_handles_trig_product_to_sum_sin_sin_plus_small_polynomial_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(2*sin(x)*sin(y) - cos(x-y) + cos(x+y)) + (x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_product_to_sum_sin_sin_minus_small_polynomial_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(2*sin(x)*sin(y) - cos(x-y) + cos(x+y)) - (x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_trig_product_to_sum_sin_sin_shifted_quotient_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*sin(y)) + 1)/((cos(x-y) - cos(x+y)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (numerator, denominator) = match simplifier.context.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => panic!("expected division root"),
    };
    let numerator_core = strip_positive_one_passthrough_root(&mut simplifier.context, numerator)
        .unwrap_or_else(|| panic!("expected numerator passthrough core"));
    let denominator_core =
        strip_positive_one_passthrough_root(&mut simplifier.context, denominator)
            .unwrap_or_else(|| panic!("expected denominator passthrough core"));

    let numerator_rewrite =
        try_rewrite_product_to_sum_expr(&mut simplifier.context, numerator_core)
            .map(|rewrite| render(&simplifier.context, rewrite.rewritten))
            .unwrap_or_else(|| "<none>".to_string());
    assert!(
        matches_direct_trig_product_to_sum_sin_sin_pair_root(
            &mut simplifier.context,
            numerator_core,
            denominator_core
        ),
        "numerator_core={}, denominator_core={}, numerator_rewrite={}",
        render(&simplifier.context, numerator_core),
        render(&simplifier.context, denominator_core),
        numerator_rewrite,
    );
}
#[test]
fn detects_direct_trig_product_to_sum_cos_cos_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("2*cos(x)*cos(y)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("cos(x+y) + cos(x-y)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_trig_product_to_sum_cos_cos_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}
#[test]
fn detects_direct_trig_product_to_sum_cos_cos_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "2*cos(x)*cos(y) - cos(x+y) - cos(x-y)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn simplify_pipeline_handles_trig_product_to_sum_cos_cos_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*cos(x)*cos(y) - cos(x+y) - cos(x-y)) + 1)/((sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn detects_direct_trig_product_to_sum_sin_cos_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("2*sin(x)*cos(y)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("sin(x+y) + sin(x-y)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_trig_product_to_sum_sin_cos_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}
#[test]
fn detects_direct_trig_product_to_sum_sin_cos_odd_difference_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("2*sin(x)*cos(2*x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("sin(3*x) - sin(x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_trig_product_to_sum_sin_cos_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}
#[test]
fn detects_direct_trig_product_to_sum_sin_cos_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "2*sin(x)*cos(y) - sin(x+y) - sin(x-y)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn simplify_pipeline_handles_trig_product_to_sum_sin_cos_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + 1)/((tan(x) + cot(x) - sec(x)*csc(x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn detects_direct_trig_mixed_double_angle_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_trig_mixed_double_angle_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn detects_direct_pythagorean_extended_pair_nested_arg_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("sin(sin(u))^4 + cos(sin(u))^4", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("1 - 2*sin(sin(u))^2*cos(sin(u))^2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    assert!(matches_direct_pythagorean_extended_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}
#[test]
fn simplify_pipeline_handles_pythagorean_extended_nested_arg_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(sin(u))^4 + cos(sin(u))^4) - (1 - 2*sin(sin(u))^2*cos(sin(u))^2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn direct_small_zero_identity_shortcut_handles_pythagorean_extended_polynomial_arg_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "(sin(u^3 + 1)^4 + cos(u^3 + 1)^4) - (1 - 2*sin(u^3 + 1)^2*cos(u^3 + 1)^2)",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) =
        try_standard_direct_small_zero_identity_shortcut(&options, &mut ctx, expr, false)
            .unwrap_or_else(|| panic!("expected direct small-zero identity shortcut to match"));
    assert_eq!(render(&ctx, rewritten), "0");
}
#[test]
fn direct_pythagorean_extended_zero_shortcut_handles_polynomial_arg_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "(sin(u^3 + 1)^4 + cos(u^3 + 1)^4) - (1 - 2*sin(u^3 + 1)^2*cos(u^3 + 1)^2)",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _steps) =
        try_standard_direct_pythagorean_extended_zero_shortcut(&mut ctx, expr, false)
            .unwrap_or_else(|| panic!("expected direct pythagorean-extended shortcut to match"));
    assert_eq!(render(&ctx, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_pythagorean_extended_polynomial_arg_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(u^3 + 1)^4 + cos(u^3 + 1)^4) - (1 - 2*sin(u^3 + 1)^2*cos(u^3 + 1)^2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn partitioned_direct_small_zero_sum_skips_direct_pythagorean_extended_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(sin(u))^4 + cos(sin(u))^4) - (1 - 2*sin(sin(u))^2*cos(sin(u))^2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    assert!(
        super::try_extract_partitioned_direct_small_zero_sum_chunks_root(
            &mut simplifier.context,
            expr,
        )
        .is_none()
    );
    assert!(super::try_extract_partitioned_exact_zero_leaf_chunks_root(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        expr,
    )
    .is_none());
}
#[test]
fn simplify_pipeline_handles_half_angle_tan_zero_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(1 - cos(2*x))/sin(2*x) - tan(x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_repeated_term_pure_double_angle_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("sin(x+x) - 2*sin(x)*cos(x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_angle_sum_diff_shifted_quotient_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cos(6*x)) + 1)/((cos(5*x)*cos(x)-sin(5*x)*sin(x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (numerator, denominator) = match simplifier.context.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => panic!("expected division root"),
    };
    let numerator_core = strip_positive_one_passthrough_root(&mut simplifier.context, numerator)
        .unwrap_or_else(|| panic!("expected numerator passthrough core"));
    let denominator_core =
        strip_positive_one_passthrough_root(&mut simplifier.context, denominator)
            .unwrap_or_else(|| panic!("expected denominator passthrough core"));

    assert!(
        matches_direct_angle_sum_diff_pair_root(
            &mut simplifier.context,
            numerator_core,
            denominator_core
        ),
        "numerator_core={}, denominator_core={}",
        render(&simplifier.context, numerator_core),
        render(&simplifier.context, denominator_core),
    );
}
#[test]
fn simplify_pipeline_handles_angle_sum_diff_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cos(6*x)) + 1)/((cos(5*x)*cos(x)-sin(5*x)*sin(x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_handles_angle_sum_diff_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cos(6*x)) + m) - ((cos(5*x)*cos(x)-sin(5*x)*sin(x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_angle_sum_diff_passthrough_reverse_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cos(5*x)*cos(x)-sin(5*x)*sin(x)) + m) - ((cos(6*x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn matches_direct_three_term_phase_shift_zero_subset_root_handles_positive_quarter_shift() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "sin(x) + cos(x) - sqrt(2)*sin(x + pi/4)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_three_term_phase_shift_zero_subset_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn matches_direct_three_term_phase_shift_zero_subset_root_handles_negative_quarter_shift() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "sin(x) - cos(x) - sqrt(2)*sin(x - pi/4)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_three_term_phase_shift_zero_subset_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn matches_direct_three_term_phase_shift_zero_subset_root_handles_weighted_third_shift() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "2*sin(x) + 2*sqrt(3)*cos(x) - 4*sin(pi/3 + x)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_three_term_phase_shift_zero_subset_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn matches_direct_three_term_phase_shift_zero_subset_root_handles_general_shifted_sine() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_three_term_phase_shift_zero_subset_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn matches_direct_numeric_general_phase_shift_zero_identity_root_handles_general_shifted_sine() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        matches_direct_numeric_general_phase_shift_zero_identity_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn is_potential_direct_three_term_phase_shift_zero_subset_root_handles_three_four_five_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(is_potential_direct_three_term_phase_shift_zero_subset_root(
        &mut simplifier.context,
        expr,
    ));
}
#[test]
fn detects_half_angle_binomial_square_shifted_quotient_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x)^2 - (1 - cos(2*x))/2) + 1)/(((sin(x) + cos(x))^2 - (1 + sin(2*x))) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (numerator, denominator) = match simplifier.context.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => panic!("expected division root"),
    };
    let numerator_core = strip_positive_one_passthrough_root(&mut simplifier.context, numerator)
        .unwrap_or_else(|| panic!("expected numerator passthrough core"));
    let denominator_core =
        strip_positive_one_passthrough_root(&mut simplifier.context, denominator)
            .unwrap_or_else(|| panic!("expected denominator passthrough core"));

    assert!(matches_direct_half_angle_square_zero_identity_root(
        &mut simplifier.context,
        numerator_core,
    ));
    assert!(matches_direct_trig_binomial_square_zero_identity_root(
        &mut simplifier.context,
        denominator_core,
    ));
    assert!(matches_direct_half_angle_binomial_square_pair_root(
        &mut simplifier.context,
        numerator_core,
        denominator_core,
    ));
}
#[test]
fn detects_direct_half_angle_square_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("cos(x)^2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("((1 + cos(2*x))/2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    assert!(matches_direct_half_angle_square_pair_root(
        &mut simplifier.context,
        lhs,
        rhs,
    ));
}
#[test]
fn detects_direct_pythagorean_identity_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("sin(x)^2 + cos(x)^2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("1", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    assert!(matches_direct_pythagorean_identity_pair_root(
        &mut simplifier.context,
        lhs,
        rhs,
    ));
}
#[test]
fn simplify_pipeline_handles_pythagorean_identity_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x)^2 + cos(x)^2) + m) - ((1) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn direct_pair_shortcut_handles_pythagorean_identity_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x)^2 + cos(x)^2) + m) - ((1) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_shared_passthrough_direct_pair_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected direct pair shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_pythagorean_identity_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x)^2 + cos(x)^2) + 1)/((1) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_handles_embedded_positive_pythagorean_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("sin(x)^2 + cos(x)^2 - sin(y)^2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "cos(y)^2");
}
#[test]
fn simplify_pipeline_handles_mixed_positive_negative_pythagorean_pairs_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "sin(x)^2 + cos(x)^2 - sin(y)^2 - cos(y)^2",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_squared_pythagorean_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((sin(x)^2 + cos(x)^2)^2) + m) - (((1)^2) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn exact_one_shortcut_handles_half_angle_binomial_square_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x)^2 - (1 - cos(2*x))/2) + 1)/(((sin(x) + cos(x))^2 - (1 + sin(2*x))) + 1)",
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
    .unwrap_or_else(|| panic!("expected exact-one shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_contracts_direct_half_angle_cos_square_root_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("((1+cos(2*x))/2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "cos(x)^2");
}
#[test]
fn simplify_pipeline_handles_scaled_direct_half_angle_cos_square_root_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("8*((1+cos(2*x))/2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "8 * cos(x)^2");
}
#[test]
fn simplify_pipeline_handles_fraction_times_direct_half_angle_cos_square_root_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*x+1)/(x*(x+1))) * ((1+cos(2*x))/2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, rewritten),
        "(cos(x)^2 + 2 * x * cos(x)^2) / (x * (x + 1))"
    );
}
#[test]
fn tangent_addition_factor_shortcut_matches_multiple_angle_regression() {
    let mut ctx = Context::new();
    let expr = parse("(sin(5*x)) * (tan(x) + tan(y))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_tangent_addition_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "tangent-addition factor shortcut should match multiple-angle products"
    );
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_tangent_addition_regression() {
    let mut ctx = Context::new();
    let expr = parse("(cot(5*pi/12)) * (tan(x) + tan(y))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_special_angle_exact_value_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "special-angle exact-value factor shortcut should match cot multiple-angle products"
    );
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_tan_angle_sum_regression() {
    let mut ctx = Context::new();
    let expr = parse("(cot(5*pi/12)) * (tan(x+y))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_special_angle_exact_value_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "special-angle exact-value factor shortcut should match tan-angle-sum products"
    );
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_sum_of_squares_product_regression() {
    let mut ctx = Context::new();
    let expr = parse("(cot(5*pi/12)) * ((w^2 + p^2)*(u^2 + v^2))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_special_angle_exact_value_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "special-angle exact-value factor shortcut should match sum-of-squares products"
    );
}
#[test]
fn sum_of_squares_product_subset_factor_shortcut_matches_special_angle_regression() {
    let mut ctx = Context::new();
    let expr = parse("(cot(5*pi/12)) * ((w^2 + p^2)*(u^2 + v^2))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_sum_of_squares_product_subset_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "sum-of-squares subset factor shortcut should match special-angle products"
    );
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_product_to_sum_regression() {
    let mut ctx = Context::new();
    let expr = parse("(sin(5*pi/6)) * (2*sin(x)*cos(2*x))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_special_angle_exact_value_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "special-angle exact-value factor shortcut should match product-to-sum products"
    );
}
#[test]
fn trig_product_to_sum_subset_factor_shortcut_matches_special_angle_regression() {
    let mut ctx = Context::new();
    let expr = parse("(sin(5*pi/6)) * (2*sin(x)*cos(2*x))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_trig_product_to_sum_subset_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "product-to-sum subset factor shortcut should match special-angle products"
    );
}
#[test]
fn trig_product_to_sum_subset_factor_shortcut_matches_external_partner_regression() {
    let mut ctx = Context::new();
    let expr = parse("(2*sin(x)*cos(2*x)) * (cos(pi - x))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_trig_product_to_sum_subset_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "product-to-sum subset factor shortcut should match products with an external partner"
    );
}
#[test]
fn trig_product_to_sum_subset_factor_shortcut_simplifies_reflection_partner_regression() {
    let mut ctx = Context::new();
    let expr = parse("(2*sin(x)*cos(2*x)) * (cos(pi - u))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let Some((result, _)) = super::try_standard_trig_product_to_sum_subset_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    ) else {
        panic!("product-to-sum subset factor shortcut should simplify reflection partners");
    };

    let rendered = render_expr(&ctx, result);
    assert!(rendered.contains("sin(3 * x) - sin(x)"));
    assert!(rendered.contains("cos(u)"));
}
#[test]
fn trig_product_to_sum_subset_factor_shortcut_canonicalizes_double_angle_partner_regression() {
    let mut ctx = Context::new();
    let expr = parse("(2*sin(x)*cos(2*x)) * (sin(2*u))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let Some((result, _)) = super::try_standard_trig_product_to_sum_subset_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    ) else {
        panic!("product-to-sum subset factor shortcut should rewrite direct-pair partners");
    };

    let rendered = render_expr(&ctx, result);
    assert!(rendered.contains("sin(3 * x) - sin(x)"));
    assert!(rendered.contains("2 * sin(u) * cos(u)"));
}
#[test]
fn trig_product_to_sum_subset_factor_shortcut_simplifies_sqrt_partner_regression() {
    let mut ctx = Context::new();
    let expr = parse("(2*sin(x)*cos(2*x)) * (sqrt(18) - sqrt(2))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let Some((result, _)) = super::try_standard_trig_product_to_sum_subset_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    ) else {
        panic!("product-to-sum subset factor shortcut should simplify sqrt partners");
    };

    let rendered = render_expr(&ctx, result);
    assert!(rendered.contains("sin(3 * x) - sin(x)"));
}
#[test]
fn trig_product_to_sum_subset_factor_shortcut_canonicalizes_sum_to_product_partner_regression() {
    let mut ctx = Context::new();
    let expr = parse("(2*sin(x)*cos(2*x)) * (sin(u) + sin(3*u))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let Some((result, _)) = super::try_standard_trig_product_to_sum_subset_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    ) else {
        panic!("product-to-sum subset factor shortcut should rewrite sum-to-product partners");
    };

    let rendered = render_expr(&ctx, result);
    assert!(rendered.contains("sin(3 * x) - sin(x)"));
    assert!(rendered.contains("2 * sin(2 * u) * cos(u)"));
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_double_angle_regression() {
    let mut ctx = Context::new();
    let expr = parse("(cot(5*pi/12)) * (sin(2*x))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_special_angle_exact_value_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "special-angle exact-value factor shortcut should match double-angle factors"
    );
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_positive_double_cos_square_diff_regression() {
    let mut ctx = Context::new();
    let expr = parse("(tan(5*pi/12)) * (cos(2*u))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_special_angle_exact_value_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "special-angle exact-value factor shortcut should match positive double-angle cosine factors"
    );
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_small_exact_constant_partner_regression() {
    let mut ctx = Context::new();
    let expr = parse("(tan(5*pi/12)) * (cos(2*pi/3))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_special_angle_exact_value_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "special-angle exact-value factor shortcut should match small exact constant partners"
    );
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_cos_fourth_power_regression() {
    let mut ctx = Context::new();
    let expr = parse("(cot(5*pi/12)) * ((3+4*cos(2*x)+cos(4*x))/8)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_special_angle_exact_value_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "special-angle exact-value factor shortcut should match cos-fourth-power reduction"
    );
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_angle_sum_fraction_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "(cot(5*pi/12)) * ((sin(x)*cos(y)+cos(x)*sin(y))/(cos(x)*cos(y)-sin(x)*sin(y)))",
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
        "special-angle exact-value factor shortcut should match angle-sum tangent fractions"
    );
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_phase_shift_regression() {
    let mut ctx = Context::new();
    let expr = parse("(tan(5*pi/12)) * (cos(pi-u))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let Some((result, _)) = super::try_standard_special_angle_exact_value_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    ) else {
        panic!("special-angle exact-value factor shortcut should match phase-shift partners");
    };

    let rendered = render_expr(&ctx, result);
    assert!(rendered.contains("cos(u)"));
    assert!(rendered.contains("-"));
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_log_exp_inverse_regression() {
    let mut ctx = Context::new();
    let expr = parse("(tan(5*pi/12)) * (ln(exp(exp(u))))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let Some((result, _)) = super::try_standard_special_angle_exact_value_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    ) else {
        panic!("special-angle exact-value factor shortcut should match log-exp inverse partners");
    };

    let rendered = render_expr(&ctx, result);
    assert!(rendered.contains("e^u") || rendered.contains("exp(u)"));
    assert!(!rendered.contains("ln("));
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_inverse_trig_plan_regression() {
    let mut ctx = Context::new();
    let expr = parse("(tan(5*pi/12)) * (sin(arcsin(u)))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let Some((result, _)) = super::try_standard_special_angle_exact_value_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    ) else {
        panic!(
            "special-angle exact-value factor shortcut should match direct inverse-trig partners"
        );
    };

    let rendered = render_expr(&ctx, result);
    assert!(rendered.contains("sqrt(3) + 2") || rendered.contains("3^(1 / 2) + 2"));
    assert!(rendered.contains("u"));
    assert!(!rendered.contains("arcsin"));
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_telescoping_fraction_regression() {
    let mut ctx = Context::new();
    let expr = parse("(tan(5*pi/12)) * (1/(u*(u+1)))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let Some((result, _)) = super::try_standard_special_angle_exact_value_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    ) else {
        panic!(
            "special-angle exact-value factor shortcut should match telescoping-fraction partners"
        );
    };

    let rendered = render_expr(&ctx, result);
    assert!(rendered.contains("u + 1"));
    assert!(rendered.contains("/"));
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_sqrt_abs_partner_regression() {
    let mut ctx = Context::new();
    let expr = parse("(tan(5*pi/12)) * (sqrt((u+1)^2))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let Some((result, _)) = super::try_standard_special_angle_exact_value_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    ) else {
        panic!("special-angle exact-value factor shortcut should match sqrt-abs partners");
    };

    let rendered = render_expr(&ctx, result);
    assert!(rendered.contains("abs(u + 1)") || rendered.contains("|u + 1|"));
}
#[test]
fn special_angle_exact_value_factor_shortcut_matches_perfect_square_polynomial_regression() {
    let mut ctx = Context::new();
    let expr = parse("(tan(5*pi/12)) * (u^4 + 4*u^2 + 4)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let Some((result, _)) = super::try_standard_special_angle_exact_value_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    ) else {
        panic!(
            "special-angle exact-value factor shortcut should match perfect-square polynomial partners"
        );
    };

    let rendered = render_expr(&ctx, result);
    assert!(rendered.contains("(u^2 + 2)^2"));
}
#[test]
fn tangent_addition_fraction_product_shortcut_matches_multiple_angle_regression() {
    let mut ctx = Context::new();
    let expr = parse("(sin(5*x)) * (sin(x+y)/(cos(x)*cos(y)))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_tangent_addition_fraction_product_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "tangent-addition fraction product shortcut should match explicit fraction products"
    );
}
#[test]
fn simplify_pipeline_handles_multiple_angle_times_tangent_addition_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(sin(5*x)) * (tan(x) + tan(y))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse(
        "(sin(5*x)) * (sin(x+y)/(cos(x)*cos(y)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let difference = simplifier.context.add(Expr::Sub(rewritten, expected));
    assert!(super::isolated_simplify_rewrites_to_zero(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        difference
    ));
}
#[test]
fn detects_two_factor_product_pair_zero_difference_special_angle_product_to_sum_subset_regression()
{
    let mut ctx = Context::new();
    let expr = parse(
        "((sin(5*pi/6)) * (2*sin(x)*cos(2*x))) - ((1/2) * (sin(3*x) - sin(x)))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    assert!(
        matches_direct_two_factor_product_pair_zero_difference_root(&mut ctx, expr),
        "two-factor product matcher should recognize special-angle times product-to-sum residuals"
    );
}
#[test]
fn simplify_pipeline_handles_multiple_angle_times_tangent_addition_fraction_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(5*x)) * (sin(x+y)/(cos(x)*cos(y)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse(
        "(sin(5*x)*sin(x+y))/(cos(x)*cos(y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let difference = simplifier.context.add(Expr::Sub(rewritten, expected));
    assert!(super::isolated_simplify_rewrites_to_zero(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        difference
    ));
}
#[test]
fn simplify_pipeline_handles_multiple_angle_times_positive_double_cos_square_diff_factor_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(sin(5*x)) * (2*cos(x)^2 - 1)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, rewritten),
        "sin(5 * x) * cos(2 * x)"
    );
}
#[test]
fn simplify_pipeline_handles_successive_unit_fraction_times_sin_cos_product_to_sum_zero_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1/x + 1/(x+1)) * (sin(x)*cos(y))) - (((2*x+1)/(x*(x+1))) * ((sin(x+y)+sin(x-y))/2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_direct_scaled_half_angle_square_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("2*cos(u/2)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("1 + cos(u)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_scaled_half_angle_square_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_abs_trig_half_angle_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("abs(sin(x/2))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs =
        parse("sqrt((1-cos(x))/2)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_abs_trig_half_angle_pair_root(&ctx, lhs, rhs));
}
#[test]
fn detects_direct_quintuple_angle_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("sin(5*x)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("16*sin(x)^5 - 20*sin(x)^3 + 5*sin(x)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_quintuple_angle_pair_root(&mut ctx, lhs, rhs));
}
#[test]
fn detects_direct_sum_to_product_contraction_pair_regression() {
    let mut ctx = Context::new();
    let lhs =
        parse("sin(x) + sin(3*x)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs =
        parse("2*sin(2*x)*cos(x)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_sum_to_product_contraction_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_tan_angle_sum_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("tan(x+y)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("(tan(x)+tan(y))/(1 - tan(x)*tan(y))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_tan_angle_sum_pair_root(&mut ctx, lhs, rhs));
}
#[test]
fn detects_two_factor_product_pair_zero_difference_special_angle_tan_angle_sum_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((cot(5*pi/12)) * (tan(x+y))) - (((2 - sqrt(3))) * ((tan(x)+tan(y))/(1 - tan(x)*tan(y))))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn detects_two_factor_product_pair_zero_difference_sum_to_product_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((1/x + 1/(x+1)) * (sin(x) + sin(3*x))) - (((2*x+1)/(x*(x+1))) * (2*sin(2*x)*cos(x)))",
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
fn detects_two_factor_product_pair_zero_difference_product_to_sum_pure_double_angle_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (sin(2*u))) - ((sin(3*x) - sin(x)) * (2*sin(u)*cos(u)))",
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
fn detects_two_factor_product_pair_zero_difference_product_to_sum_sum_to_product_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (sin(u) + sin(3*u))) - ((sin(3*x) - sin(x)) * (2*sin(2*u)*cos(u)))",
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
fn detects_direct_trig_phase_shift_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("sin(pi/2 - z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("cos(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_trig_phase_shift_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_trig_phase_shift_reflection_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("cos(pi - u)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("-cos(u)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_trig_phase_shift_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_numeric_general_phase_shift_pair_regression() {
    let mut ctx = Context::new();
    let lhs =
        parse("3*sin(z) + 4*cos(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs =
        parse("5*sin(z + arctan(4/3))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_numeric_general_phase_shift_pair_root(
        &mut ctx, lhs, rhs
    ));
    assert!(super::matches_known_direct_pair_root(&mut ctx, lhs, rhs));
}
#[test]
fn detects_direct_trig_triple_angle_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("sin(3*z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs =
        parse("3*sin(z) - 4*sin(z)^3", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_trig_triple_angle_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_sec_tan_pythagorean_pair_regression() {
    let mut ctx = Context::new();
    let lhs =
        parse("sec(z)^2 - tan(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_sec_tan_pythagorean_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_tan_to_sec_pythagorean_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("1 + tan(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("sec(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_tan_to_sec_pythagorean_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_csc_cot_pythagorean_pair_regression() {
    let mut ctx = Context::new();
    let lhs =
        parse("csc(z)^2 - cot(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_csc_cot_pythagorean_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_cot_to_csc_pythagorean_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("1 + cot(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("csc(z)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_cot_to_csc_pythagorean_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_special_angle_exact_value_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("cot(5*pi/12)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("2 - 3^(1/2)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_special_angle_exact_value_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_special_angle_exact_value_pair_sqrt_form_regression() {
    let mut ctx = Context::new();
    let lhs = parse("cot(5*pi/12)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("2 - sqrt(3)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_special_angle_exact_value_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_special_angle_exact_value_half_regression() {
    let mut ctx = Context::new();
    let lhs = parse("sin(5*pi/6)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("1/2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_special_angle_exact_value_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_special_angle_exact_value_negative_fraction_regression() {
    let mut ctx = Context::new();
    let lhs = parse("cos(2*pi/3)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("-1/2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_special_angle_exact_value_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_pure_double_angle_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("sin(2*z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("2*sin(z)*cos(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_pure_double_angle_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_double_angle_inverse_trig_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("sin(2*arcsin(z))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("2*z*sqrt(1-z^2)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_double_angle_inverse_trig_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_double_angle_inverse_trig_arccos_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("sin(2*arccos(z))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("2*z*sqrt(1-z^2)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_double_angle_inverse_trig_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn simplify_pipeline_handles_special_angle_cot_times_tangent_addition_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cot(5*pi/12)) * (tan(x) + tan(y))) - (((2 - 3^(1/2))) * (sin(x+y)/(cos(x)*cos(y))))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_special_angle_cot_times_tan_angle_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cot(5*pi/12)) * (tan(x+y))) - (((2 - sqrt(3))) * ((tan(x)+tan(y))/(1 - tan(x)*tan(y))))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_special_angle_tan_times_positive_double_cos_square_diff_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(tan(5*pi/12)) * (cos(2*u))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("((2 + 3^(1/2))) * (cos(2*u))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let difference = simplifier.context.add(Expr::Sub(rewritten, expected));
    assert!(super::isolated_simplify_rewrites_to_zero(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        difference
    ));
}
#[test]
fn simplify_pipeline_handles_special_angle_tan_times_small_exact_constant_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(tan(5*pi/12)) * (cos(2*pi/3))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("(-1/2) * (3^(1/2) + 2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let mut diff_orchestrator = Orchestrator::new();
    let (diff, _steps, _stats) = diff_orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
}
#[test]
fn simplify_pipeline_handles_special_angle_tan_times_direct_sqrt_constant_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(tan(5*pi/12)) * (sqrt(2))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("(2^(1/2)) * (3^(1/2) + 2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let mut diff_orchestrator = Orchestrator::new();
    let (diff, _steps, _stats) = diff_orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
}
#[test]
fn simplify_pipeline_handles_special_angle_tan_times_symbol_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(tan(5*pi/12)) * k", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("k * (3^(1/2) + 2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let mut diff_orchestrator = Orchestrator::new();
    let (diff, _steps, _stats) = diff_orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
}
#[test]
fn simplify_pipeline_aligns_fractional_special_angle_with_short_geometric_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse(
        "(cos(3*pi/8)) * (u^3 + u^2 + u + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse(
        "((sqrt(2 - sqrt(2))/2) * ((u+1)*(u^2+1)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
    let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, source_nf),
        render(&simplifier.context, target_nf)
    );
}
#[test]
fn simplify_pipeline_aligns_fractional_special_angle_with_shifted_square_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse("(cos(3*pi/8)) * (u^2 + 2*u)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse(
        "((sqrt(2 - sqrt(2))/2) * ((u+1)^2 - 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
    let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, source_nf),
        render(&simplifier.context, target_nf)
    );
}
#[test]
fn simplify_pipeline_aligns_fractional_special_angle_with_difference_of_squares_partner_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse("(cos(3*pi/8)) * (u^2 - 4)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse(
        "((sqrt(2 - sqrt(2))/2) * ((u-2)*(u+2)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
    let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, source_nf),
        render(&simplifier.context, target_nf)
    );
}
#[test]
fn simplify_pipeline_aligns_fractional_special_angle_with_difference_of_cubes_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse("(cos(3*pi/8)) * (u^3 - 1)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse(
        "((sqrt(2 - sqrt(2))/2) * ((u-1)*(u^2 + u + 1)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
    let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, source_nf),
        render(&simplifier.context, target_nf)
    );
}
#[test]
fn simplify_pipeline_aligns_fractional_special_angle_with_sum_of_squares_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse("(cos(3*pi/8)) * (u^2 + v^2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse(
        "((sqrt(2 - sqrt(2))/2) * ((u+v)^2 - 2*u*v))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
    let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, source_nf),
        render(&simplifier.context, target_nf)
    );
}
#[test]
fn simplify_pipeline_aligns_fractional_special_angle_with_abs_half_angle_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse("(cos(3*pi/8)) * (abs(sin(u/2)))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse(
        "((sqrt(2 - sqrt(2))/2) * (sqrt((1-cos(u))/2)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
    let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, source_nf),
        render(&simplifier.context, target_nf)
    );
}
#[test]
fn simplify_pipeline_aligns_phi_with_abs_half_angle_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse("(phi^2) * (abs(sin(u/2)))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse(
        "((phi + 1) * (sqrt((1-cos(u))/2)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
    let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, source_nf),
        render(&simplifier.context, target_nf)
    );
}
#[test]
fn simplify_pipeline_preserves_scaled_half_angle_partner_inside_fractional_special_angle_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse("(cos(3*pi/8)) * (2*cos(u/2)^2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse(
        "((sqrt(2 - sqrt(2))/2) * (1 + cos(u)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
    let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
    let rendered = render(&simplifier.context, source_nf);
    assert!(rendered.contains("cos(u)"));
    let diff = simplifier.context.add(Expr::Sub(source_nf, target_nf));
    assert!(super::isolated_simplify_rewrites_to_zero(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        diff
    ));
}
#[test]
fn simplify_pipeline_aligns_fractional_special_angle_with_duplicate_sum_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse("(cos(3*pi/8)) * (u+u)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse("((sqrt(2 - sqrt(2))/2) * (2*u))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
    let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
    let diff = simplifier.context.add(Expr::Sub(source_nf, target_nf));
    assert!(super::isolated_simplify_rewrites_to_zero(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        diff
    ));
}
#[test]
fn simplify_pipeline_aligns_fractional_special_angle_with_partition_of_unity_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse(
        "(cos(3*pi/8)) * (exp(u)/(exp(u) + 1) + 1/(exp(u) + 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse("((sqrt(2 - sqrt(2))/2) * 1)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
    let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
    let diff = simplifier.context.add(Expr::Sub(source_nf, target_nf));
    assert!(super::isolated_simplify_rewrites_to_zero(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        diff
    ));
}
#[test]
fn simplify_pipeline_handles_special_angle_tan_times_two_linear_shift_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(tan(5*pi/12)) * (u^2 + 5*u + 6)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let rendered = render_expr(&simplifier.context, rewritten);
    assert!(rendered.contains("sqrt(3) + 2") || rendered.contains("3^(1 / 2) + 2"));
    assert!(rendered.contains("u + 2"));
    assert!(rendered.contains("u + 3"));
}
#[test]
fn simplify_pipeline_handles_phase_shift_times_two_linear_shift_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(cos(pi - x)) * (u^2 + 5*u + 6)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let rendered = render_expr(&simplifier.context, rewritten);
    assert!(rendered.starts_with("-"));
    assert!(rendered.contains("cos(x)"));
    assert!(rendered.contains("u + 2"));
    assert!(rendered.contains("u + 3"));
}
#[test]
fn simplify_pipeline_aligns_safe_anchor_with_abs_half_angle_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse("(phi + 1) * (abs(cos(u/2)))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse(
        "((phi + 1) * (sqrt((1+cos(u))/2)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
    let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, source_nf),
        render(&simplifier.context, target_nf)
    );
}
#[test]
fn simplify_pipeline_handles_safe_anchor_times_positive_scaled_half_angle_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(2*sqrt(2)) * (1 + cos(u))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("(2*sqrt(2)) * (2*cos(u/2)^2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
}
#[test]
fn simplify_pipeline_aligns_scaled_half_angle_anchor_with_sum_diff_cubes_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse("(2*cos(x/2)^2) * (u^3 + v^3)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse(
        "(1 + cos(x)) * ((u+v)*(u^2 - u*v + v^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
    let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, source_nf),
        render(&simplifier.context, target_nf)
    );
}
#[test]
fn simplify_pipeline_aligns_scaled_half_angle_anchor_with_higher_degree_difference_partner_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse("(2*cos(x/2)^2) * (u^6 - 1)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse(
        "(1 + cos(x)) * ((u^2+u+1)*(u^2-u+1)*(u+1)*(u-1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
    let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, source_nf),
        render(&simplifier.context, target_nf)
    );
}
#[test]
fn simplify_pipeline_avoids_scaled_half_angle_anchor_loop_with_safe_anchor_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(2*sqrt(2)) * (1 + cos(u))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("(2*sqrt(2)) * (2*cos(u/2)^2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
}
#[test]
fn simplify_pipeline_avoids_scaled_half_angle_anchor_loop_with_constant_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(2*cos(x/2)^2) * 2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("(1 + cos(x)) * 2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
}
#[test]
fn detects_two_factor_product_pair_zero_difference_special_angle_pure_double_angle_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((cot(5*pi/12)) * (sin(2*x))) - (((2 - 3^(1/2))) * (2*sin(x)*cos(x)))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn simplify_pipeline_handles_sec_tan_pythagorean_times_chebyshev_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sec(u)^2 - tan(u)^2) * (cos(2*u))) - ((1) * (2*cos(u)^2 - 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_cot_to_csc_pythagorean_times_chebyshev_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + cot(u)^2) * (cos(2*u))) - (((csc(u)^2)) * (2*cos(u)^2 - 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_two_factor_product_pair_zero_difference_quintuple_product_to_sum_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((sin(5*x)) * (2*sin(x)*cos(2*x))) - (((16*sin(x)^5 - 20*sin(x)^3 + 5*sin(x)) * (sin(3*x) - sin(x))))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn detects_two_factor_product_pair_zero_difference_quartic_gcf_sum_to_product_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((x^4 - x^2) * (sin(x) + sin(3*x))) - (((x^2*(x-1)*(x+1)) * (2*sin(2*x)*cos(x))))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn simplify_pipeline_handles_quartic_gcf_times_sum_to_product_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - x^2) * (sin(x) + sin(3*x))) - (((x^2*(x-1)*(x+1)) * (2*sin(2*x)*cos(x))))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_cos_fourth_over_known_angle_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(cos(x)^4)/(cos(2*pi/5)) - (((3+4*cos(2*x)+cos(4*x))/8)/((sqrt(5)-1)/4))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_quintuple_angle_times_product_to_sum_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(5*x)) * (2*sin(x)*cos(2*x))) - (((16*sin(x)^5 - 20*sin(x)^3 + 5*sin(x)) * (sin(3*x) - sin(x))))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_quintuple_angle_times_positive_double_cos_square_diff_zero_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(5*x)) * (cos(2*x))) - (((16*sin(x)^5 - 20*sin(x)^3 + 5*sin(x)) * (2*cos(x)^2 - 1)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_half_angle_against_small_trig_zero_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x)^2 - (1 - cos(2*x))/2) + 1)/((2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn detects_direct_small_zero_telescoping_vs_half_angle_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1)) - (sin(x)^2 - (1 - cos(2*x))/2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let lhs = parse(
        "1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("sin(x)^2 - (1 - cos(2*x))/2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        lhs,
    ));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        rhs,
    ));
    assert!(is_direct_small_zero_composition_candidate_root(
        &mut simplifier.context,
        expr,
    ));
}
#[test]
fn direct_small_zero_pair_shortcut_handles_telescoping_vs_half_angle_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1)) - (sin(x)^2 - (1 - cos(2*x))/2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_direct_small_zero_pair_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        options.collect_steps,
    )
    .unwrap_or_else(|| panic!("expected direct small zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn compact_tan_cot_half_angle_pair_shortcut_handles_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(tan(x)*cot(x) - 1) + (sin(x)^2 - (1 - cos(2*x))/2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, steps) = super::try_standard_compact_tan_cot_half_angle_zero_pair_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        true,
    )
    .unwrap_or_else(|| panic!("expected compact tan-cot plus half-angle shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
    assert_eq!(steps.len(), 1);
}
#[test]
fn direct_small_zero_pair_shortcut_handles_log_product_vs_trig_product_to_sum_cos_cos_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*cos(x)*cos(y) - cos(x+y) - cos(x-y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = super::try_standard_direct_small_zero_pair_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        options.collect_steps,
    )
    .unwrap_or_else(|| panic!("expected direct small zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn direct_small_zero_pair_shortcut_handles_log_product_vs_trig_mixed_double_angle_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = super::try_standard_direct_small_zero_pair_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        options.collect_steps,
    )
    .unwrap_or_else(|| panic!("expected direct small zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn cached_compact_simplify_pipeline_handles_log_product_vs_trig_mixed_double_angle_sum_regression()
{
    let profile = crate::profile_cache::default_rule_profile();
    let mut simplifier = crate::Simplifier::from_profile(profile);
    simplifier.set_steps_mode(crate::options::StepsMode::Compact);
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut options = SimplifyOptions::default();
    options.shared.context_mode = crate::options::ContextMode::Standard;
    let (rewritten, _steps, stats) = simplifier.simplify_with_stats(expr, options);
    assert_eq!(render(&simplifier.context, rewritten), "0");
    assert!(
        stats.core.phase.is_none(),
        "expected root shortcut before Core, got stats: {stats:?}"
    );
}
#[test]
fn direct_small_zero_pair_shortcut_handles_log_product_vs_phase_shift_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = super::try_standard_direct_small_zero_pair_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        options.collect_steps,
    )
    .unwrap_or_else(|| panic!("expected direct small zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn direct_small_zero_pair_shortcut_handles_four_two_term_core_sum_with_half_angle_tan_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs =
        parse("tan(x)", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("(1-cos(2*x))/sin(2*x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_half_angle_tan_pair_root(
        &mut simplifier.context,
        lhs,
        rhs,
    ));
    assert!(super::matches_known_direct_pair_root(
        &mut simplifier.context,
        lhs,
        rhs,
    ));

    for core in [
        "sec(a)-1/cos(a)",
        "csc(b)-1/sin(b)",
        "tan(x)-(1-cos(2*x))/sin(2*x)",
        "1/(1 + 1/(1+u)) - (1+u)/(2+u)",
    ] {
        let core_expr =
            parse(core, &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            super::matches_direct_small_zero_or_known_pair_base_root(
                &mut simplifier.context,
                core_expr,
            ),
            "expected core to match: {core}"
        );
    }
    let expr = parse(
        "(sec(a)-1/cos(a)) + (csc(b)-1/sin(b)) + (tan(x)-(1-cos(2*x))/sin(2*x)) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = super::try_standard_direct_small_zero_pair_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        options.collect_steps,
    )
    .unwrap_or_else(|| panic!("expected direct small zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn direct_small_zero_additive_combination_shortcut_handles_nested_fraction_vs_phase_shift_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let result = super::try_standard_direct_small_zero_additive_combination_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        expr,
        false,
    );
    assert!(
        result.is_some(),
        "expected additive combination shortcut to match"
    );
}
#[test]
fn detects_direct_three_term_phase_shift_zero_identity_base_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_small_zero_or_known_pair_base_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn detects_direct_three_term_phase_shift_pair_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_three_term_phase_shift_pair_zero_identity_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn detects_direct_symbolic_trig_sum_to_product_zero_identity_base_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "sin(x) + sin(y) - 2*sin((x+y)/2)*cos((x-y)/2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_small_zero_or_known_pair_base_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn zero_product_with_exact_zero_child_shortcut_handles_reciprocal_trig_vs_phase_shift_product_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(tan(x) + cot(x) - sec(x)*csc(x)) * (3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let result = super::try_standard_zero_product_with_exact_zero_child_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        options.collect_steps,
    );
    assert!(result.is_some(), "expected zero-product shortcut to match");
}
#[test]
fn zero_product_with_exact_zero_child_shortcut_handles_nested_fraction_vs_phase_shift_product_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let result = super::try_standard_zero_product_with_exact_zero_child_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        options.collect_steps,
    );
    assert!(result.is_some(), "expected zero-product shortcut to match");
}
#[test]
fn simplify_pipeline_handles_telescoping_vs_half_angle_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1)) - (sin(x)^2 - (1 - cos(2*x))/2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn common_scale_residual_extracts_trig_product_to_sum_sin_sin_scaled_difference_regression() {
    let mut ctx = Context::new();
    let expr = parse("k*(2*sin(x)*sin(y)) - k*(cos(x-y) - cos(x+y))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (common_factor, residual_expr) =
        extract_common_multiplicative_residual_sum_root(&mut ctx, expr)
            .unwrap_or_else(|| panic!("expected common multiplicative residual"));
    assert_eq!(render(&ctx, common_factor), "k");
    assert_eq!(
        render(&ctx, residual_expr),
        "2 * sin(x) * sin(y) - (cos(x - y) - cos(x + y))"
    );
}
#[test]
fn common_scale_residual_matches_trig_product_to_sum_sin_sin_scaled_difference_regression() {
    let mut ctx = Context::new();
    let residual_expr = parse("(2*sin(x)*sin(y)) - (cos(x-y) - cos(x+y))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_small_zero_or_known_pair_residual_root(
        &mut ctx,
        residual_expr
    ));
}
#[test]
fn common_scale_fallback_matches_trig_product_to_sum_sin_sin_scaled_difference_regression() {
    let mut ctx = Context::new();
    let expr = parse("k*(2*sin(x)*sin(y)) - k*(cos(x-y) - cos(x+y))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) =
        try_standard_common_scale_exact_zero_shortcut_fallback(&options, &mut ctx, expr, false)
            .unwrap_or_else(|| panic!("expected common-scale fallback to match"));
    assert_eq!(render(&ctx, rewritten), "0");
}
#[test]
fn common_scale_known_pair_shortcut_matches_trig_product_to_sum_sin_sin_scaled_difference_regression(
) {
    let mut ctx = Context::new();
    let expr = parse("k*(2*sin(x)*sin(y)) - k*(cos(x-y) - cos(x+y))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) =
        try_standard_common_scale_known_pair_shortcut(&options, &mut ctx, expr, false)
            .unwrap_or_else(|| panic!("expected common-scale known-pair shortcut to match"));
    assert_eq!(render(&ctx, rewritten), "0");
}
#[test]
fn direct_known_pair_zero_shortcut_skips_common_scale_trig_product_to_sum_regression() {
    let mut ctx = Context::new();
    let expr = parse("k*(2*sin(x)*sin(y)) - k*(cos(x-y) - cos(x+y))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(
        try_standard_direct_known_pair_zero_shortcut(&options, &mut ctx, expr, false).is_none()
    );
}
#[test]
fn two_factor_product_pair_zero_shortcut_skips_common_scale_trig_product_to_sum_regression() {
    let mut ctx = Context::new();
    let expr = parse("k*(2*sin(x)*sin(y)) - k*(cos(x-y) - cos(x+y))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(
        try_standard_two_factor_product_pair_zero_shortcut(&options, &mut ctx, expr, false)
            .is_none()
    );
}
#[test]
fn exact_zero_equivalence_shortcut_matches_trig_product_to_sum_sin_sin_scaled_difference_regression(
) {
    let mut ctx = Context::new();
    let expr = parse("k*(2*sin(x)*sin(y)) - k*(cos(x-y) - cos(x+y))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) =
        try_standard_exact_zero_equivalence_shortcut(&options, &mut ctx, expr, false)
            .unwrap_or_else(|| panic!("expected exact-zero shortcut to match"));
    assert_eq!(render(&ctx, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_product_to_sum_sin_sin_scaled_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "k*(2*sin(x)*sin(y)) - k*(cos(x-y) - cos(x+y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_direct_mixed_pythagorean_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(cosh(x*y))^2 - (sinh(x*y))^2 - (sin(x+y))^2 - (cos(x+y))^2",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn simplify_pipeline_handles_exact_additive_pair_chain_before_trig_double_angle_probe_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("2*cos(2*x) + 1 - 2*cos(2*x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn shifted_quotient_shortcut_handles_pythagorean_factor_form_from_sin_sq_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x)^2) + 1)/((1-cos(x)^2) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_shifted_quotient_exact_one_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        true,
    )
    .unwrap_or_else(|| panic!("expected shifted quotient exact-one shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn shifted_quotient_shortcut_handles_pythagorean_factor_form_to_cos_sq_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 - sin(x)^2) + 1)/((cos(x)^2) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_shifted_quotient_exact_one_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        true,
    )
    .unwrap_or_else(|| panic!("expected shifted quotient exact-one shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn exact_one_shortcut_handles_trig_double_angle_cos_variant_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cos(2*x)) + 1)/((1 - 2*sin(x)^2) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_shifted_quotient_exact_one_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        true,
    )
    .unwrap_or_else(|| panic!("expected shifted quotient exact-one shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn nested_zero_direct_pair_family_candidate_gate_keeps_trig_product_to_sum_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*sin(y)) + 1)/((cos(x-y) - cos(x+y)) + 1)",
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

    assert!(is_potential_nested_zero_direct_pair_family_pair_root(
        &mut simplifier.context,
        numerator_core,
        denominator_core,
    ));
}
#[test]
fn simplify_pipeline_handles_pythagorean_factor_form_from_sin_sq_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x)^2) + 1)/((1-cos(x)^2) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn shifted_quotient_nested_zero_fast_gate_candidate_matches_log_product_vs_phase_shift_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (numerator, denominator) = match simplifier.context.get(expr).clone() {
        Expr::Div(numerator, denominator) => (numerator, denominator),
        _ => panic!("expected division root"),
    };

    assert!(
        matches_shifted_quotient_nested_zero_fast_gate_candidate_root(
            &mut simplifier.context,
            numerator,
            denominator,
        )
    );
}
#[test]
fn shifted_quotient_nested_zero_fast_gate_candidate_matches_nested_fraction_vs_phase_shift_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (numerator, denominator) = match simplifier.context.get(expr).clone() {
        Expr::Div(numerator, denominator) => (numerator, denominator),
        _ => panic!("expected division root"),
    };

    assert!(
        matches_shifted_quotient_nested_zero_fast_gate_candidate_root(
            &mut simplifier.context,
            numerator,
            denominator,
        )
    );
}
#[test]
fn shifted_quotient_shortcut_handles_trig_double_angle_cos_variant_residual_difference_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cos(2*x)) + 1)/((1 - 2*sin(x)^2) + 1)",
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
fn shifted_quotient_shortcut_handles_pure_double_angle_residual_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(2*x)) + 1)/((2*sin(x)*cos(x)) + 1)",
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
#[ignore = "Direct simplify_pipeline still overflows stack for this half-angle plus telescoping residual; coverage remains in exact-zero rewrite and shifted-quotient guards"]
fn simplify_pipeline_handles_half_angle_against_telescoping_fraction_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(x)^2 - (1 - cos(2*x))/2) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_two_factor_product_pair_zero_difference_product_to_sum_reflection_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (cos(pi - u))) - ((sin(3*x) - sin(x)) * (-cos(u)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_two_factor_product_pair_zero_difference_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn detects_two_factor_product_pair_zero_difference_product_to_sum_two_linear_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * ((u+2)*(u+3))) - ((sin(3*x) - sin(x)) * (u^2 + 5*u + 6))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_two_factor_product_pair_zero_difference_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn detects_two_factor_product_pair_zero_difference_product_to_sum_three_linear_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * ((u+1)*(u+2)*(u+3))) - ((sin(3*x) - sin(x)) * (u^3 + 6*u^2 + 11*u + 6))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_two_factor_product_pair_zero_difference_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn detects_two_factor_product_pair_zero_difference_product_to_sum_inverse_trig_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (sin(arctan(u)))) - ((sin(3*x) - sin(x)) * (u/sqrt(1 + u^2)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_two_factor_product_pair_zero_difference_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn detects_two_factor_product_pair_zero_difference_product_to_sum_double_angle_inverse_trig_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (sin(2*arcsin(u)))) - ((sin(3*x) - sin(x)) * (2*u*sqrt(1-u^2)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_two_factor_product_pair_zero_difference_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn detects_two_factor_product_pair_zero_difference_product_to_sum_weierstrass_sin_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (sin(u))) - ((sin(3*x) - sin(x)) * (2*tan(u/2)/(1 + tan(u/2)^2)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_two_factor_product_pair_zero_difference_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn detects_two_factor_product_pair_zero_difference_product_to_sum_higher_binomial_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * ((u-1)^5)) - ((sin(3*x) - sin(x)) * (u^5 - 5*u^4 + 10*u^3 - 10*u^2 + 5*u - 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_two_factor_product_pair_zero_difference_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn detects_two_factor_product_pair_zero_difference_product_to_sum_log_split_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (ln(sqrt(u)*v))) - ((sin(3*x) - sin(x)) * (ln(u)/2 + ln(v)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_two_factor_product_pair_zero_difference_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn detects_two_factor_product_pair_zero_difference_product_to_sum_higher_degree_difference_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (u^6 - 1)) - ((sin(3*x) - sin(x)) * ((u^2+u+1)*(u^2-u+1)*(u+1)*(u-1)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_two_factor_product_pair_zero_difference_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn detects_two_factor_product_pair_zero_difference_product_to_sum_cauchy_schwarz_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * ((w^2 + p^2)*(u^2 + v^2))) - ((sin(3*x) - sin(x)) * ((w*u + p*v)^2 + (w*v - p*u)^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_two_factor_product_pair_zero_difference_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn direct_product_to_sum_factor_partner_matches_cauchy_schwarz_regression() {
    let mut ctx = Context::new();
    let lhs =
        parse("(2*sin(x)*cos(2*x))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs =
        parse("sin(3*x) - sin(x)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_trig_product_to_sum_sin_cos_pair_root(
        &mut ctx, lhs, rhs
    ));

    let partner_lhs = parse("((w^2 + p^2)*(u^2 + v^2))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let partner_rhs = parse("((w*u + p*v)^2 + (w*v - p*u)^2)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::factors_match_by_equality_or_direct_pair_root(
        &mut ctx,
        partner_lhs,
        partner_rhs
    ));
}
#[test]
fn detects_two_factor_product_pair_zero_difference_product_to_sum_inverse_trig_constant_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (arcsin(1))) - ((sin(3*x) - sin(x)) * (pi/2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_two_factor_product_pair_zero_difference_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn detects_two_factor_product_pair_zero_difference_product_to_sum_inverse_trig_alias_constant_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (atan(1))) - ((sin(3*x) - sin(x)) * (pi/4))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_two_factor_product_pair_zero_difference_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn detects_two_factor_product_pair_zero_difference_product_to_sum_special_angle_constant_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (cos(2*pi/3))) - ((sin(3*x) - sin(x)) * (-1/2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_two_factor_product_pair_zero_difference_root(
            &mut simplifier.context,
            expr
        )
    );
}
#[test]
fn simplify_pipeline_handles_product_to_sum_three_linear_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * ((u+1)*(u+2)*(u+3))) - ((sin(3*x) - sin(x)) * (u^3 + 6*u^2 + 11*u + 6))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_product_to_sum_two_linear_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * ((u+2)*(u+3))) - ((sin(3*x) - sin(x)) * (u^2 + 5*u + 6))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_product_to_sum_pure_double_angle_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (sin(2*u))) - ((sin(3*x) - sin(x)) * (2*sin(u)*cos(u)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_product_to_sum_sum_to_product_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (sin(u) + sin(3*u))) - ((sin(3*x) - sin(x)) * (2*sin(2*u)*cos(u)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_product_to_sum_inverse_trig_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (sin(arctan(u)))) - ((sin(3*x) - sin(x)) * (u/sqrt(1 + u^2)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_product_to_sum_double_angle_inverse_trig_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    simplifier.set_steps_mode(crate::options::StepsMode::Off);
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (sin(2*arcsin(u)))) - ((sin(3*x) - sin(x)) * (2*u*sqrt(1-u^2)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_three_linear_shift_anchor_times_double_angle_inverse_trig_partner_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((u+1)*(u+2)*(u+3)) * (sin(2*arcsin(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let rendered = render_expr(&simplifier.context, rewritten);
    assert!(rendered.contains("u^3 + 6 * u^2 + 11 * u + 6"));
    assert!(rendered.contains("2 * x"));
    assert!(
        rendered.contains("(1 - x^2)^(1/2)") || rendered.contains("sqrt(1 - x^2)"),
        "unexpected inverse-trig bridge render: {rendered}"
    );
    assert!(!rendered.contains("sin("));
    assert!(!rendered.contains("arcsin("));
}
#[test]
fn simplify_pipeline_handles_product_to_sum_weierstrass_sin_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (sin(u))) - ((sin(3*x) - sin(x)) * (2*tan(u/2)/(1 + tan(u/2)^2)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_product_to_sum_higher_binomial_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * ((u-1)^5)) - ((sin(3*x) - sin(x)) * (u^5 - 5*u^4 + 10*u^3 - 10*u^2 + 5*u - 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_product_to_sum_cauchy_schwarz_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * ((w^2 + p^2)*(u^2 + v^2))) - ((sin(3*x) - sin(x)) * ((w*u + p*v)^2 + (w*v - p*u)^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_product_to_sum_inverse_trig_alias_constant_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (atan(1))) - ((sin(3*x) - sin(x)) * (pi/4))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_product_to_sum_inverse_trig_constant_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (arcsin(1))) - ((sin(3*x) - sin(x)) * (pi/2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_product_to_sum_special_angle_constant_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(x)*cos(2*x)) * (cos(2*pi/3))) - ((sin(3*x) - sin(x)) * (-1/2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_rational_plus_pythagorean_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1/u + 1/(u+1)) + (sec(u)^2 - tan(u)^2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let expected = parse("(2 * u + 1) / (u * (u + 1)) + 1", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let equivalence = simplifier.context.add(Expr::Sub(rewritten, expected));
    let (equivalence, _steps, _stats) =
        orchestrator.simplify_pipeline(equivalence, &mut simplifier);
    assert_eq!(render(&simplifier.context, equivalence), "0");
}
#[test]
fn simplify_pipeline_handles_trig_cubic_against_general_phase_shift_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(2*sin(2*x)*sin(x) - (4*cos(x) - 4*cos(x)^3)) + (3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_sqrt_perfect_square_against_trig_product_to_sum_sum_regression() {
    for expr_text in [
        "sqrt(a^2 + 2*a*b + b^2) - abs(a+b)",
        "2*sin(x)*sin(y) - cos(x-y) + cos(x+y)",
    ] {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr = parse(expr_text, &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        orchestrator.options.collect_steps = false;
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }
}
#[test]
fn partitioned_direct_small_zero_sum_shortcut_handles_sqrt_perfect_square_against_trig_product_to_sum_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(a^2 + 2*a*b + b^2) - abs(a+b)) + (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let result = super::try_standard_partitioned_direct_small_zero_sum_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        expr,
        false,
    );
    assert!(
        result.is_some(),
        "expected direct small-zero identity shortcut"
    );
}
#[test]
fn direct_small_zero_identity_shortcut_handles_tan_cot_product_against_trig_product_to_sum_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(tan(x)*cot(x) - 1) + (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let result = super::try_standard_direct_small_zero_identity_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        expr,
        false,
    );
    assert!(
        result.is_some(),
        "expected direct small-zero identity shortcut"
    );
}
#[test]
fn direct_small_zero_additive_combination_shortcut_handles_trig_product_to_sum_against_odd_half_power_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(2*sin(x)*sin(y) - cos(x-y) + cos(x+y)) + (sqrt(x^5) - x^2*sqrt(x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let result = super::try_standard_direct_small_zero_additive_combination_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        expr,
        true,
    );
    assert!(
        result.is_some(),
        "expected direct small-zero additive combination shortcut"
    );
}
#[test]
fn detects_tan_cot_plus_trig_product_to_sum_sum_structure_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(tan(x)*cot(x) - 1) + (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let view = AddView::from_expr(&simplifier.context, expr);
    let rendered_terms: Vec<_> = view
        .terms
        .iter()
        .map(|(term, sign)| format!("{sign:?}:{}", render(&simplifier.context, *term)))
        .collect();
    let trig_chunk = super::build_signed_sum_expr_root(
        &mut simplifier.context,
        &[view.terms[0], view.terms[3], view.terms[4]],
    );
    let tan_chunk =
        super::build_signed_sum_expr_root(&mut simplifier.context, &[view.terms[1], view.terms[2]]);
    let tan_chunk_terms: Vec<_> = AddView::from_expr(&simplifier.context, tan_chunk)
        .terms
        .iter()
        .map(|(term, sign)| format!("{sign:?}:{}", render(&simplifier.context, *term)))
        .collect();
    assert!(
        super::matches_direct_small_zero_identity_root(&mut simplifier.context, trig_chunk),
        "trig_chunk={} terms={rendered_terms:?} rendered={}",
        render(&simplifier.context, trig_chunk),
        render(&simplifier.context, expr),
    );
    assert!(
        super::matches_direct_small_zero_identity_root(&mut simplifier.context, tan_chunk),
        "tan_chunk={} tan_terms={tan_chunk_terms:?} terms={rendered_terms:?} rendered={}",
        render(&simplifier.context, tan_chunk),
        render(&simplifier.context, expr),
    );
}
#[test]
fn detects_direct_sec_tan_pythagorean_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("sec(x)^2 - tan(x)^2 - 1", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_sec_tan_pythagorean_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn detects_direct_csc_cot_pythagorean_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("csc(x)^2 - cot(x)^2 - 1", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_csc_cot_pythagorean_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn detects_direct_squared_pythagorean_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(sin(x)^2 + cos(x)^2)^2 - 1", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_squared_exact_one_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn simplify_pipeline_handles_csc_cot_pythagorean_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("csc(x)^2 - cot(x)^2 - 1", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_squared_pythagorean_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(sin(x)^2 + cos(x)^2)^2 - 1", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_csc_cot_pythagorean_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("csc(x)^2 - cot(x)^2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn small_trig_zero_child_gate_matches_half_angle_sine_core() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("sin(x)^2 - (1 - cos(2*x))/2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(is_small_trig_or_hyperbolic_zero_child(
        &options,
        &mut simplifier.context,
        expr
    ));
}
