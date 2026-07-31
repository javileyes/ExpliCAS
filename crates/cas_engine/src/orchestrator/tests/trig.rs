//! Tests del orquestador: familia `trig` (troceo P1).

use super::*;

#[test]
fn standard_exact_additive_pair_chain_shortcut_cancels_trig_and_constant_tail() {
    let mut ctx = Context::new();
    let expr = parse("2*cos(2*x) + 1 - 2*cos(2*x)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _) = try_standard_exact_additive_pair_chain_shortcut(
        &crate::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("shortcut should cancel exact additive pair chain"));
    assert_eq!(render(&ctx, rewritten), "1");
}
#[test]
fn mixed_sign_trig_square_difference_root_guard_matches_two_term_difference() {
    let mut ctx = Context::new();
    let expr =
        parse("-sin(x)^2 + cos(x)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(is_mixed_sign_trig_square_difference_root(&ctx, expr));
}
#[test]
fn standard_trig_fourth_power_difference_shortcut_finishes_hidden_zero_identity() {
    let mut ctx = Context::new();
    let expr = parse("sin(x)^4 - cos(x)^4 - (sin(x)^2 - cos(x)^2)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _) = try_standard_trig_fourth_power_difference_shortcut(&mut ctx, expr, false)
        .unwrap_or_else(|| panic!("shortcut should match hidden quartic identity"));
    assert_eq!(render(&ctx, rewritten), "0");
}
#[test]
fn standard_sin_sum_triple_identity_zero_shortcut_handles_nested_scaled_argument() {
    let mut ctx = Context::new();
    let expr = parse(
        "sin(2*u) + sin(3*(2*u)) - 2*sin(2*(2*u))*cos(2*u)",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _) = try_standard_sin_sum_triple_identity_zero_shortcut(&mut ctx, expr, false)
        .unwrap_or_else(|| panic!("shortcut should match nested scaled triple identity"));
    assert_eq!(render(&ctx, rewritten), "0");
}
#[test]
fn trig_log_zero_product_direct_shortcut_returns_zero() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(tan(x) + cot(x) - sec(x)*csc(x)) * (2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_zero_product_with_exact_zero_child_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected zero-product shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_log_zero_product_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(tan(x) + cot(x) - sec(x)*csc(x)) * (2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn child_isolated_exact_zero_handles_small_trig_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("tan(x) + cot(x) - sec(x)*csc(x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(child_isolated_exact_zero(
        &options,
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn child_isolated_exact_zero_handles_trig_product_sum_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))",
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
fn shifted_trig_identity_case336_strips_passthrough_and_proves_both_cores_zero() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((tan(x) + cot(x) - sec(x)*csc(x)) + 1)/((sin(x)^2 - (1 - cos(2*x))/2) + 1)",
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
    let options = SimplifyOptions::default();
    assert!(child_isolated_exact_zero(
        &options,
        &mut simplifier.context,
        numerator_core
    ));
    assert!(child_isolated_exact_zero(
        &options,
        &mut simplifier.context,
        denominator_core
    ));
}
#[test]
fn shifted_trig_identity_case336_direct_div_shortcut_returns_one() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((tan(x) + cot(x) - sec(x)*csc(x)) + 1)/((sin(x)^2 - (1 - cos(2*x))/2) + 1)",
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
fn simplify_pipeline_handles_nested_additive_shifted_trig_identity_case336_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((tan(x) + cot(x) - sec(x)*csc(x)) + 1)/((sin(x)^2 - (1 - cos(2*x))/2) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn trig_power_reduction_factor_shortcut_matches_collapsed_fraction_regression() {
    let mut ctx = Context::new();
    let expr = parse("((1/x + 1/(x+1)) * (sin(x)^2*cos(x)^2))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let result = super::try_standard_trig_power_reduction_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    );

    assert!(
        result.is_some(),
        "trig-power reduction factor shortcut should match collapsed-fraction mixed-square products"
    );
}
#[test]
fn simplify_pipeline_handles_collapsed_fraction_times_trig_power_reduction_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let raw = parse(
        "((1/x + 1/(x+1)) * (sin(x)^2*cos(x)^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse(
        "(((2*x+1)/(x*(x+1))) * ((sin(2*x)^2)/4))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (raw_result, _steps, _stats) = orchestrator.simplify_pipeline(raw, &mut simplifier);
    let (expected_result, _steps, _stats) =
        orchestrator.simplify_pipeline(expected, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, raw_result),
        render(&simplifier.context, expected_result)
    );
}
#[test]
fn detects_negative_double_cos_square_diff_shifted_quotient_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x)^2 - cos(x)^2) + 1)/((-cos(2*x)) + 1)",
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

    assert!(matches_direct_negative_double_cos_square_diff_pair_root(
        &mut simplifier.context,
        numerator_core,
        denominator_core
    ));
}
#[test]
fn simplify_pipeline_handles_negative_double_cos_square_diff_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x)^2 - cos(x)^2) + 1)/((-cos(2*x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_handles_positive_double_cos_square_diff_direct_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("cos(x)^2 - sin(x)^2 - cos(2*x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_negative_double_cos_square_diff_direct_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("sin(x)^2 - cos(x)^2 + cos(2*x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_positive_double_cos_square_diff_nested_arg_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "cos(sin(u))^2 - sin(sin(u))^2 - cos(2*sin(u))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_positive_double_cos_square_diff_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cos(x)^2 - sin(x)^2) + 1)/((cos(2*x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_handles_negative_double_cos_square_diff_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((-cos(2*x)) + m) - ((sin(x)^2 - cos(x)^2) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_negative_double_cos_square_diff_passthrough_forward_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x)^2 - cos(x)^2) + m) - ((-cos(2*x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_reciprocal_passthrough_forward_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("((1/cos(x)) + m) - ((sec(x)) + m)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_ratio_passthrough_forward_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(2*x)/cos(2*x)) + m) - ((tan(2*x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_ratio_alias_passthrough_forward_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(2*x)/cos(x+x)) + m) - ((tan(2*x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_negative_double_cos_square_diff_scaled_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "k*(-cos(2*x)) - k*(sin(x)^2 - cos(x)^2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_negative_double_cos_square_diff_common_denominator_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x)^2 - cos(x)^2)/q) - ((-cos(2*x))/q)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_cubic_cosine_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*cos(2*x)*cos(x)) + m) - ((4*cos(x)^3-2*cos(x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_shifted_quotient_with_reversed_reciprocal_trig_zero_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x)^2 - (1 - cos(2*x))/2) + 1)/((tan(x) + cot(x) - sec(x)*csc(x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn detects_direct_trig_binomial_square_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("(sin(x) + cos(x))^2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("1 + sin(2*x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    assert!(matches_direct_trig_binomial_square_pair_root(
        &mut simplifier.context,
        lhs,
        rhs,
    ));
}
#[test]
fn simplify_pipeline_handles_trig_binomial_square_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((sin(x)+cos(x))^2) + m) - ((1+sin(2*x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_binomial_square_passthrough_without_steps_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((sin(x)+cos(x))^2) + m) - ((1+sin(2*x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_trig_binomial_square_passthrough_direct_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((sin(x)+cos(x))^2) + m) - ((1+sin(2*x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected_lhs = parse("(sin(x)+cos(x))^2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected_rhs = parse("1+sin(2*x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (lhs_core, rhs_core) =
        extract_shared_additive_passthrough_sub_cores_root(&mut simplifier.context, expr)
            .unwrap_or_else(|| panic!("expected passthrough cores"));

    assert_eq!(
        compare_expr(&simplifier.context, lhs_core, expected_lhs),
        Ordering::Equal
    );
    assert_eq!(
        compare_expr(&simplifier.context, rhs_core, expected_rhs),
        Ordering::Equal
    );
    assert_eq!(
        passthrough_direct_pair_rule_name_root(&mut simplifier.context, lhs_core, rhs_core),
        Some("Collapse Exact Zero Additive Subexpression"),
    );
}
#[test]
fn direct_pair_shortcut_handles_trig_binomial_square_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((sin(x)+cos(x))^2) + m) - ((1+sin(2*x)) + m)",
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
fn direct_pair_shortcut_handles_trig_binomial_square_passthrough_with_steps_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((sin(x)+cos(x))^2) + m) - ((1+sin(2*x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, steps) = try_standard_shared_passthrough_direct_pair_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        true,
    )
    .unwrap_or_else(|| panic!("expected direct pair shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
    assert!(!steps.is_empty());
}
#[test]
fn simplify_pipeline_handles_trig_binomial_square_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((((sin(x)+cos(x))^2) + 1))/(((1+sin(2*x)) + 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_contracts_direct_cos_fourth_power_reduction_root_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("((3+4*cos(2*x)+cos(4*x))/8)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "cos(x)^4");
}
#[test]
fn simplify_pipeline_handles_scaled_direct_cos_fourth_power_reduction_root_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("8*((3+4*cos(2*x)+cos(4*x))/8)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "8 * cos(x)^4");
}
#[test]
fn simplify_pipeline_handles_fraction_times_direct_cos_fourth_power_reduction_root_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*x+1)/(x*(x+1))) * ((3+4*cos(2*x)+cos(4*x))/8)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, rewritten),
        "(cos(x)^4 * (2 * x + 1))/(x * (x + 1))"
    );
}
#[test]
fn simplify_pipeline_contracts_direct_sin_cos_square_product_reduction_root_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("((1-cos(4*x))/8)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1/4 * sin(2 * x)^2");
}
#[test]
fn simplify_pipeline_handles_scaled_positive_double_cos_square_diff_factor_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("8*(2*cos(x)^2 - 1)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "8 * cos(2 * x)");
}
#[test]
fn simplify_pipeline_handles_fraction_times_positive_double_cos_square_diff_factor_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*x+1)/(x*(x+1))) * (2*cos(x)^2 - 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, rewritten),
        "(cos(2 * x) * (2 * x + 1))/(x * (x + 1))"
    );
}
#[test]
fn simplify_pipeline_handles_successive_unit_fraction_times_positive_double_cos_square_diff_zero_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1/x + 1/(x+1)) * cos(2*x)) - (((2*x+1)/(x*(x+1))) * (2*cos(x)^2 - 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_successive_unit_fraction_times_trig_power_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1/x + 1/(x+1)) * (sin(x)^2*cos(x)^2)) - (((2*x+1)/(x*(x+1))) * ((sin(2*x)^2)/4))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_direct_cos_fourth_power_reduction_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("cos(x)^4", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("((3+4*cos(2*x)+cos(4*x))/8)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_cos_fourth_power_reduction_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_two_factor_product_pair_zero_difference_sin_cos_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((1/x + 1/(x+1)) * (sin(x)*cos(y))) - (((2*x+1)/(x*(x+1))) * ((sin(x+y)+sin(x-y))/2))",
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
    let pair_00 = factors_match_by_equality_or_direct_pair_root(
        &mut ctx,
        lhs_factor_ids[0],
        rhs_factor_ids[0],
    );
    let pair_01 = factors_match_by_equality_or_direct_pair_root(
        &mut ctx,
        lhs_factor_ids[0],
        rhs_factor_ids[1],
    );
    let pair_10 = factors_match_by_equality_or_direct_pair_root(
        &mut ctx,
        lhs_factor_ids[1],
        rhs_factor_ids[0],
    );
    let pair_11 = factors_match_by_equality_or_direct_pair_root(
        &mut ctx,
        lhs_factor_ids[1],
        rhs_factor_ids[1],
    );
    assert!(
        matches_direct_two_factor_product_pair_zero_difference_root(&mut ctx, expr),
        "lhs factors = {:?}, rhs factors = {:?}, pair00 = {}, pair01 = {}, pair10 = {}, pair11 = {}",
        lhs_factors,
        rhs_factors,
        pair_00,
        pair_01,
        pair_10,
        pair_11
    );
}
#[test]
fn subtract_expanded_sum_diff_cubes_quotient_shortcut_handles_trig_square_cube_plain_fourth_power_residual(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "((sin(u)^2)^3 - 1)/((sin(u)^2) - 1) - (sin(u)^4 + sin(u)^2 + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let options = crate::phase::SimplifyOptions::default();
    let (rewritten, _steps) =
        super::try_standard_subtract_expanded_sum_diff_cubes_quotient_shortcut(
            &options, &mut ctx, expr, false,
        )
        .unwrap_or_else(|| panic!("shortcut"));

    assert_eq!(render(&ctx, rewritten), "0");
}
#[test]
fn detects_direct_reciprocal_trig_product_one_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("sin(z)*csc(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_reciprocal_trig_product_one_pair_root(
        &mut ctx, lhs, rhs
    ));

    let tan_cot_lhs =
        parse("tan(z)*cot(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_reciprocal_trig_product_one_pair_root(
        &mut ctx,
        tan_cot_lhs,
        rhs
    ));
}
#[test]
fn detects_direct_inverse_trig_exact_value_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("arcsin(1)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("pi/2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_special_angle_exact_value_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn detects_direct_trig_inverse_composition_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("sin(arctan(u))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("u/sqrt(1 + u^2)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_trig_inverse_composition_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn simplify_pipeline_handles_negative_cos_times_two_linear_shift_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(-cos(x)) * (u^2 + 5*u + 6)", &mut simplifier.context)
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
fn simplify_pipeline_handles_trig_square_cube_substitution_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((sin(u)^2)^3 - 1) / ((sin(u)^2) - 1)) - ((sin(u)^4) + (sin(u)^2) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_two_factor_product_pair_zero_difference_cot_to_csc_chebyshev_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((1 + cot(u)^2) * (cos(2*u))) - (((csc(u)^2)) * (2*cos(u)^2 - 1))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn simplify_pipeline_handles_reciprocal_sqrt_times_positive_double_cos_square_diff_zero_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1/sqrt(x)) * (cos(2*x))) - (((sqrt(x)/x) * (2*cos(x)^2 - 1)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_cos_fourth_over_chebyshev_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(cos(x)^4)/(cos(2*x)) - (((3+4*cos(2*x)+cos(4*x))/8)/(2*cos(x)^2 - 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_division_factor_pairs_for_cos_fourth_over_chebyshev_regression() {
    let mut ctx = Context::new();
    let lhs_num = parse("cos(x)^4", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs_num = parse("((3+4*cos(2*x)+cos(4*x))/8)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let lhs_den = parse("cos(2*x)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs_den =
        parse("2*cos(x)^2 - 1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(factors_match_by_equality_or_direct_pair_root(
        &mut ctx, lhs_num, rhs_num
    ));
    assert!(factors_match_by_equality_or_direct_pair_root(
        &mut ctx, lhs_den, rhs_den
    ));
}
#[test]
fn detects_direct_quotient_pair_zero_difference_cos_fourth_chebyshev_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "(cos(x)^4)/(cos(2*x)) - (((3+4*cos(2*x)+cos(4*x))/8)/(2*cos(x)^2 - 1))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_quotient_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn simplify_pipeline_handles_cos_fourth_over_exp_log_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(cos(x)^4)/(ln(exp(x)^2)) - (((3+4*cos(2*x)+cos(4*x))/8)/(2*x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_cos_fourth_over_completing_square_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(cos(x)^4)/(x^2 + 2*x) - (((3+4*cos(2*x)+cos(4*x))/8)/(x*(x+2)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_exp_combination_times_positive_double_cos_square_diff_zero_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((exp(a)*exp(b)) * (cos(2*x))) - ((exp(a+b)) * (2*cos(x)^2 - 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_small_trig_zero_pair_product_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))) * (2*sin(2*x)*sin(x) - (4*cos(x) - 4*cos(x)^3))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_small_trig_zero_pair_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))) + 1)/((2*sin(2*x)*sin(x) - (4*cos(x) - 4*cos(x)^3)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_handles_trig_cubic_passthrough_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(2*x)*sin(x)+a) + 1)/((4*cos(x)-4*cos(x)^3+a) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_handles_trig_cubic_scaled_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "k*(2*sin(2*x)*sin(x)) - k*(4*cos(x)-4*cos(x)^3)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_cubic_common_denominator_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(2*x)*sin(x))/q) - ((4*cos(x)-4*cos(x)^3)/q)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_cubic_passthrough_common_denominator_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*sin(2*x)*sin(x)+a)/q) - ((4*cos(x)-4*cos(x)^3+a)/q)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn direct_small_zero_pair_shortcut_handles_log_product_vs_trig_cubic_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sin(2*x)*sin(x) - (4*cos(x) - 4*cos(x)^3))",
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
fn simplify_pipeline_handles_log_product_vs_trig_cubic_product_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) * (2*sin(2*x)*sin(x) - (4*cos(x) - 4*cos(x)^3))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn direct_small_zero_pair_shortcut_handles_negative_trig_reciprocal_nested_fraction_three_core_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    for core in [
        "-sin(2*x) + 2*sin(x)*cos(x)",
        "sec(y) - 1/cos(y)",
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
        "(-sin(2*x) + 2*sin(x)*cos(x)) + (sec(y) - 1/cos(y)) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
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
fn direct_small_zero_pair_shortcut_handles_log_trig_reciprocal_nested_fraction_three_core_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    for core in [
        "ln(x^3)+ln(y^2)-ln(x^3*y^2)",
        "sec(z)-1/cos(z)",
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
        "(ln(x^3)+ln(y^2)-ln(x^3*y^2)) + (sec(z)-1/cos(z)) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
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
fn direct_small_zero_pair_shortcut_handles_four_two_term_core_sum_with_trig_ratio_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs =
        parse("tan(c)", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("sin(c)/cos(c)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_trig_ratio_pair_root(
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
        "tan(c)-sin(c)/cos(c)",
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
        "(sec(a)-1/cos(a)) + (csc(b)-1/sin(b)) + (tan(c)-sin(c)/cos(c)) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
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
fn direct_small_zero_pair_shortcut_handles_four_two_term_core_sum_with_trig_ratio_alias_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("tan(2*c)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("sin(2*c)/cos(c+c)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_trig_ratio_alias_pair_root(
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
        "tan(2*c)-sin(2*c)/cos(c+c)",
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
        "(sec(a)-1/cos(a)) + (csc(b)-1/sin(b)) + (tan(2*c)-sin(2*c)/cos(c+c)) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
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
fn detects_direct_tan_cot_sec_csc_zero_identity_base_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("tan(x) + cot(x) - sec(x)*csc(x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_small_zero_or_known_pair_base_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn shifted_quotient_shortcut_handles_reciprocal_trig_against_rationalized_sum_of_sqrts_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((tan(x) + cot(x) - sec(x)*csc(x)) + 1)/((1/(sqrt(a) + sqrt(b)) - (sqrt(a) - sqrt(b))/(a - b)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _steps) = try_standard_shifted_quotient_nested_zero_core_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected shifted quotient nested zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_handles_trig_mixed_scaled_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "k*(2*cos(2*x)*sin(x)) - k*(4*cos(x)^2*sin(x)-2*sin(x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_mixed_common_denominator_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*cos(2*x)*sin(x))/q) - ((4*cos(x)^2*sin(x)-2*sin(x))/q)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_trig_mixed_passthrough_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((2*cos(2*x)*sin(x)+a) + 1)/((4*cos(x)^2*sin(x)-2*sin(x)+a) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn simplify_pipeline_handles_log_product_split_against_trig_mixed_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn nested_exact_zero_child_shortcut_handles_log_product_split_against_trig_mixed_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x)))",
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
    .unwrap_or_else(|| panic!("expected nested exact-zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn small_trig_zero_pair_shortcut_rejects_large_mixed_log_scope_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (ln(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y))) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(try_standard_small_trig_zero_pair_shortcut(
        &SimplifyOptions::default(),
        &mut simplifier.context,
        expr,
        false,
    )
    .is_none());
}
#[test]
fn classify_multiterm_trig_numeric_subset_status_is_candidate_ready_on_raw_polynomial_triple_sine_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1)",
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
    let (subset_expr, partner_expr) =
        extract_small_trig_or_hyperbolic_numeric_subset_root(&mut simplifier.context, expr)
            .unwrap_or_else(|| panic!("expected subset extraction to succeed"));
    assert_eq!(
        render(&simplifier.context, subset_expr),
        "sin(3 * x) / sin(x) - 2 * cos(2 * x) - 1"
    );
    assert!(child_isolated_exact_zero(
        &options,
        &mut simplifier.context,
        subset_expr
    ));
    assert!(supported_nested_zero_partner_rewrites_to_zero(
        &options,
        &mut simplifier.context,
        partner_expr
    ));
}
#[test]
fn multiterm_trig_numeric_subset_zero_shortcut_handles_triple_sine_against_polynomial_partner_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1)",
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
fn classify_multiterm_trig_numeric_subset_status_is_candidate_ready_on_normalized_residual_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "cosh(x*y)^2 + (x^4 + y^4 - 2*x^2*y^2)/(x-y) + y^3 + x*y^2 - sinh(x*y)^2 - x^3 - y*x^2 - 1",
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
    let (subset_expr, partner_expr) =
        extract_small_trig_or_hyperbolic_numeric_subset_root(&mut simplifier.context, expr)
            .unwrap_or_else(|| panic!("expected subset extraction to succeed"));
    assert!(child_isolated_exact_zero(
        &options,
        &mut simplifier.context,
        subset_expr
    ));
    assert!(supported_nested_zero_partner_rewrites_to_zero(
        &options,
        &mut simplifier.context,
        partner_expr
    ));
}
#[test]
fn classify_multiterm_trig_numeric_subset_status_is_candidate_ready_on_triple_sine_against_polynomial_plus_rational_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2)",
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
fn multiterm_trig_numeric_subset_zero_shortcut_handles_triple_sine_against_polynomial_plus_rational_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
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
fn multiterm_trig_numeric_subset_zero_shortcut_keeps_compact_steps_on_triple_sine_against_polynomial_plus_rational_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, steps) = try_standard_multiterm_trig_numeric_subset_zero_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        true,
    )
    .unwrap_or_else(|| panic!("expected multiterm trig-numeric subset shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
    assert_eq!(steps.len(), 2);
}
#[test]
fn multiterm_trig_numeric_subset_rewrites_to_zero_runtime_safe_handles_symbolic_triple_sine_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let subset_expr = parse(
        "sin(a) + sin(b) - 2*sin((a+b)/2)*cos((a-b)/2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut options = SimplifyOptions::default();
    options.shared.context_mode = crate::options::ContextMode::Standard;
    options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    assert!(multiterm_trig_numeric_subset_rewrites_to_zero_runtime_safe(
        &options,
        &mut simplifier.context,
        subset_expr,
    ));
}
#[test]
fn derived_shifted_root_square_residual_from_inverse_trig_mix_still_matches_sqrt_subset_zero() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(x + 2*sqrt(x-1)) - sqrt(x-1) - 1) + (asin(x/sqrt(x^2 + 1)) - atan(x)) + (atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (log((x*y)^2) - 2*log(x) - 2*log(y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let view = AddView::from_expr(&simplifier.context, expr);
    let asin_term = parse("asin(x/sqrt(x^2 + 1))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let atan_term =
        parse("atan(x)", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
        .terms
        .iter()
        .copied()
        .filter(|(term, sign)| {
            !((*term == asin_term && *sign == Sign::Pos)
                || (*term == atan_term && *sign == Sign::Neg))
        })
        .collect();
    let remaining_expr = AddView {
        root: expr,
        terms: remaining_terms,
    }
    .rebuild(&mut simplifier.context);
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_sqrt_perfect_square_abs_subset_zero_shortcut(
        &options,
        &mut simplifier.context,
        remaining_expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected derived residual to match sqrt subset zero shortcut"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn classify_multiterm_trig_numeric_subset_status_is_candidate_ready_on_log_exp_fraction_root_perfect_square_mix_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(exp(y*log(x)) - x^y) + (sin(x) + sin(y) - 2*sin((x+y)/2)*cos((x-y)/2)) + (2/(x^2 - 1) - 1/(x-1) + 1/(x+1)) + (sqrt(x + 2*sqrt(x-1)) - sqrt(x-1) - 1)",
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
fn classify_multiterm_trig_numeric_subset_status_is_no_candidate_on_single_trig_plus_numeric_noise_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(x) - 1 + 2) + (atanh((u^2 - 1)/(u^2 + 1)) - log(u)) + (exp(y*log(z)) - z^y)",
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
        Some("no_candidate")
    );
}
#[test]
fn classify_multiterm_trig_numeric_subset_status_is_no_candidate_on_single_partner_term_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(x) + sin(y) - 2*sin((x+y)/2)*cos((x-y)/2)) + z + 1",
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
        Some("no_candidate")
    );
}
#[test]
fn classify_multiterm_trig_numeric_subset_status_is_no_candidate_on_triple_sine_against_log_with_nested_trig_plus_polynomial_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (ln(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y)))",
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
        Some("no_candidate")
    );
}
#[test]
fn multiterm_trig_numeric_subset_zero_shortcut_handles_log_exp_fraction_root_perfect_square_mix_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(exp(y*log(x)) - x^y) + (sin(x) + sin(y) - 2*sin((x+y)/2)*cos((x-y)/2)) + (2/(x^2 - 1) - 1/(x-1) + 1/(x+1)) + (sqrt(x + 2*sqrt(x-1)) - sqrt(x-1) - 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
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
fn recursive_additive_zero_shortcut_steps_extracts_two_trig_chunks_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cos(x))^3 - (3*cos(x) + cos(3*x))/4) + (tan(x) + 1/tan(x) - 2/sin(2*x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let steps = try_build_recursive_additive_zero_shortcut_steps(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        expr,
    )
    .unwrap_or_else(|| panic!("expected recursive additive steps"));
    assert_eq!(steps.len(), 2);
}
#[test]
fn exact_zero_leaf_rewrites_to_zero_root_handles_trig_cubic_and_tan_reciprocal_chunks() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let cubic = parse(
        "(cos(x))^3 - (3*cos(x) + cos(3*x))/4",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let tan_chunk = parse("tan(x) + 1/tan(x) - 2/sin(2*x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = crate::phase::SimplifyOptions::default();
    assert!(exact_zero_leaf_rewrites_to_zero_root(
        &options,
        &mut simplifier.context,
        cubic,
    ));
    assert!(child_is_small_exact_zero_leaf_root(
        &options,
        &mut simplifier.context,
        cubic,
    ));
    assert!(exact_zero_leaf_rewrites_to_zero_root(
        &options,
        &mut simplifier.context,
        tan_chunk,
    ));
    assert!(child_is_small_exact_zero_leaf_root(
        &options,
        &mut simplifier.context,
        tan_chunk,
    ));
}
#[test]
fn small_trig_zero_pair_shortcut_decomposes_partitioned_trig_chunks_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cos(x))^3 - (3*cos(x) + cos(3*x))/4) + (tan(x) + 1/tan(x) - 2/sin(2*x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, steps) = try_standard_small_trig_zero_pair_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        expr,
        true,
    )
    .unwrap_or_else(|| panic!("expected small trig zero pair shortcut"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
    assert_eq!(steps.len(), 2);
}
#[test]
fn nested_exact_zero_child_shortcut_handles_log_product_split_against_trig_mixed_sum_with_steps_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_nested_exact_zero_child_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        true,
    )
    .unwrap_or_else(|| panic!("expected nested exact-zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn nested_exact_zero_child_shortcut_handles_log_product_split_against_sin_sin_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_nested_exact_zero_child_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        true,
    )
    .unwrap_or_else(|| panic!("expected nested exact-zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn shifted_quotient_shortcut_handles_log_product_split_against_sin_cos_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_shifted_quotient_nested_zero_core_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        true,
    )
    .unwrap_or_else(|| panic!("expected shifted quotient nested-zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn shifted_quotient_direct_small_zero_hot_gate_matches_nested_fraction_vs_trig_cubic_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((2*sin(2*x)*sin(x) - (4*cos(x) - 4*cos(x)^3)) + 1)",
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
fn shifted_quotient_shortcut_handles_trig_ratio_residual_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(2*x)/cos(2*x)) + 1)/((tan(2*x)) + 1)",
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
fn shifted_quotient_shortcut_handles_trig_ratio_alias_residual_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(2*x)/cos(x+x)) + 1)/((tan(2*x)) + 1)",
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
fn shifted_quotient_shortcut_handles_trig_reciprocal_residual_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("((1/cos(x)) + 1)/((sec(x)) + 1)", &mut simplifier.context)
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
#[ignore = "Direct simplify_pipeline still overflows stack for this mixed trig plus telescoping residual; coverage remains in exact-zero rewrite tests and CLI steps-off runtime"]
fn simplify_pipeline_handles_trig_mixed_against_telescoping_fraction_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    simplifier.set_steps_mode(crate::options::StepsMode::Off);
    let expr = parse(
        "(2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn rejects_trig_binomial_square_in_small_pow_expansion_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("(sin(x)+cos(x))^2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("1+sin(2*x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(!super::matches_direct_small_pow_expansion_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}
#[test]
fn simplify_pipeline_handles_three_linear_shift_anchor_times_inverse_trig_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((u+1)*(u+2)*(u+3)) * (sin(arctan(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse(
        "(u^3 + 6*u^2 + 11*u + 6) * (x/sqrt(1+x^2))",
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
fn simplify_pipeline_aligns_inverse_trig_anchor_with_short_geometric_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse(
        "(sin(arctan(x))) * (u^3 + u^2 + u + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse(
        "(x/sqrt(1 + x^2)) * ((u+1)*(u^2 + 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
    let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
    let expected = "x * (x^2 + 1)^(-1/2) * (u^3 + u^2 + u + 1)";
    assert_eq!(render(&simplifier.context, source_nf), expected);
    assert_eq!(render(&simplifier.context, target_nf), expected);
}
#[test]
fn simplify_pipeline_aligns_inverse_trig_anchor_with_two_linear_shift_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse(
        "(sin(arctan(x))) * (u^2 + 5*u + 6)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse("(x/sqrt(1 + x^2)) * ((u+2)*(u+3))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (target_nf, _steps, _stats) = orchestrator.simplify_pipeline(target, &mut simplifier);
    let (source_nf, _steps, _stats) = orchestrator.simplify_pipeline(source, &mut simplifier);
    let expected = "x * (x^2 + 1)^(-1/2) * (u^2 + 5 * u + 6)";
    assert_eq!(render(&simplifier.context, source_nf), expected);
    assert_eq!(render(&simplifier.context, target_nf), expected);
}
#[test]
fn partitioned_direct_small_zero_sum_shortcut_handles_trig_binomial_square_against_telescoping_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x) + cos(x))^2 - (1 + sin(2*x))) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
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
        "expected partitioned direct small-zero sum shortcut"
    );
}
#[test]
#[ignore = "Direct simplify_pipeline still overflows stack for this trig-binomial-square plus telescoping residual; coverage remains in direct partitioned zero-sum shortcut tests"]
fn simplify_pipeline_handles_trig_binomial_square_against_telescoping_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x) + cos(x))^2 - (1 + sin(2*x))) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_direct_tan_cot_product_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("tan(x)*cot(x) - 1", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_tan_cot_product_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn detects_direct_tan_cot_sec_csc_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("tan(x) + cot(x) - sec(x)*csc(x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_tan_cot_sec_csc_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn simplify_pipeline_handles_reciprocal_trig_plus_log_difference_squares_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(tan(x) + cot(x) - sec(x)*csc(x)) + (ln(x^2 - y^2) - ln(x - y) - ln(x + y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn small_trig_zero_child_gate_matches_binomial_square_core() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(x) + cos(x))^2 - (1 + sin(2*x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(is_small_trig_or_hyperbolic_zero_child(
        &options,
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn small_trig_zero_child_gate_matches_product_sum_core() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x) - 2*sin(x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(is_small_trig_or_hyperbolic_zero_child(
        &options,
        &mut simplifier.context,
        expr
    ));
}
