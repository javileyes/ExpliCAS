//! Tests del orquestador: familia `logs_exp` (troceo P1).

use super::*;

#[test]
fn simplify_pipeline_handles_log_fractional_power_gap_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(log(x*sqrt(x)) + log(sqrt(x)/x^2)) + (sqrt(y)/(sqrt(y)-1) - sqrt(y)/(sqrt(y)+1) - (2*sqrt(y))/(y-1)) + (((1/x) - (1/y))/((y-x)/(x*y)) - 1)",
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
fn child_isolated_exact_zero_handles_small_log_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("ln(x^3) + ln(y^2) - ln(x^3 * y^2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(child_isolated_exact_zero(
        &options,
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn simplify_pipeline_handles_log_product_split_against_nested_fraction_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_log_zero_leaf_pair_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (ln((x*y)^2) - ln(x^2) - ln(y^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_log_product_against_geometric_factor_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}
#[test]
fn perfect_square_trinomial_factor_shortcut_matches_fundamental_exp_partner_regression() {
    let mut ctx = Context::new();
    let expr = parse("(x^2 + 2*x + 1) * (cosh(u) - sinh(u))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let Some((result, _)) = super::try_standard_perfect_square_trinomial_factor_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    ) else {
        panic!("perfect-square factor shortcut should match exp-decomposition partners");
    };

    let rendered = render_expr(&ctx, result);
    assert!(rendered.contains("(x + 1)^2"));
    assert!(rendered.contains("e^u") || rendered.contains("exp(-u)"));
}
#[test]
fn detects_direct_exponential_combination_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("exp(a)*exp(b)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("exp(a+b)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_exponential_combination_pair_root(
        &mut ctx, lhs, rhs
    ));
}
#[test]
fn simplify_pipeline_handles_safe_anchor_times_log_split_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((exp(x) - exp(-x))/2) * (ln(sqrt(u)*v))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("(sinh(x)) * (ln(u)/2 + ln(v))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
}
#[test]
fn simplify_pipeline_handles_safe_anchor_times_exp_linear_shift_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((exp(x) - exp(-x))/2) * (e*exp(u))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("(sinh(x)) * (exp(u+1))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
}
#[test]
fn detects_two_factor_product_pair_zero_difference_exp_chebyshev_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((exp(a)*exp(b)) * (cos(2*x))) - ((exp(a+b)) * (2*cos(x)^2 - 1))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn detects_two_factor_product_pair_zero_difference_perfect_square_exp_sum_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((x^2 + 2*x + 1) * (cosh(u) - sinh(u))) - (((x+1)^2) * exp(-u))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn direct_small_zero_pair_shortcut_handles_log_product_vs_rationalized_sum_of_sqrts_sum_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (1/(sqrt(a) + sqrt(b)) - (sqrt(a) - sqrt(b))/(a - b))",
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
fn direct_small_zero_pair_shortcut_handles_log_product_vs_reciprocal_nested_fraction_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (((1/x + 1/y)/(1/x - 1/y)) - (x+y)/(y-x))",
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
fn simplify_pipeline_handles_log_product_vs_rationalized_sum_of_sqrts_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (1/(sqrt(a) + sqrt(b)) - (sqrt(a) - sqrt(b))/(a - b))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_log_product_vs_reciprocal_nested_fraction_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (((1/x + 1/y)/(1/x - 1/y)) - (x+y)/(y-x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn direct_small_zero_pair_shortcut_handles_log_product_vs_small_rational_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1))",
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
fn direct_small_zero_pair_shortcut_handles_log_product_vs_telescoping_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
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
fn direct_small_zero_pair_shortcut_handles_log_square_vs_telescoping_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
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
fn direct_small_zero_pair_shortcut_handles_log_product_vs_sum_diff_cubes_quotient_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + ((a^3-b^3)/(a-b) - (a^2 + a*b + b^2))",
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
fn direct_small_zero_pair_shortcut_handles_log_product_vs_small_quotient_cancel_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + ((x^2 - 1)/(x - 1) - (x+1))",
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
fn direct_small_zero_pair_shortcut_handles_log_product_vs_sqrt_abs_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (sqrt(a^2 + 2*a*b + b^2) - abs(a+b))",
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
fn direct_small_zero_pair_shortcut_handles_log_product_vs_geometric_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) - (x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1))",
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
fn direct_small_zero_additive_combination_shortcut_handles_log_product_vs_sophie_germain_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (x^4 + 4*y^4 - (x^2-2*x*y+2*y^2)*(x^2+2*x*y+2*y^2))",
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
fn direct_small_zero_additive_combination_shortcut_handles_log_square_vs_sophie_germain_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (x^4 + 4*y^4 - (x^2-2*x*y+2*y^2)*(x^2+2*x*y+2*y^2))",
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
fn direct_small_zero_additive_combination_shortcut_handles_ln_abs_vs_sophie_germain_sum_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))) + (x^4 + 4*y^4 - (x^2-2*x*y+2*y^2)*(x^2+2*x*y+2*y^2))",
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
fn detects_direct_ln_abs_product_zero_identity_base_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_small_zero_or_known_pair_base_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn simplify_pipeline_handles_log_product_vs_sophie_germain_product_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) * (x^4 + 4*y^4 - (x^2-2*x*y+2*y^2)*(x^2+2*x*y+2*y^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn zero_product_with_exact_zero_child_shortcut_handles_log_product_vs_sophie_germain_product_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) * (x^4 + 4*y^4 - (x^2-2*x*y+2*y^2)*(x^2+2*x*y+2*y^2))",
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
fn same_denominator_distribution_pair_zero_shortcut_handles_log_product_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + ((a+b+c)/x - a/x - b/x - c/x)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _steps) = super::try_standard_same_denominator_distribution_pair_zero_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected same-denominator pair shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn direct_small_zero_pair_shortcut_handles_log_product_vs_affine_common_denominator_sum_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (a + b/x - (a*x+b)/x)",
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
fn simplify_pipeline_handles_log_product_vs_same_denominator_distribution_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + ((a+b+c)/x - a/x - b/x - c/x)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn nested_exact_zero_child_shortcut_handles_log_product_split_against_nested_fraction_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1))",
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
fn nested_exact_zero_child_shortcut_rejects_large_mixed_log_scope_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (ln(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y))) + (x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(try_standard_nested_exact_zero_child_shortcut(
        &SimplifyOptions::default(),
        &mut simplifier.context,
        expr,
        false,
    )
    .is_none());
}
#[test]
fn supported_nested_zero_partner_rewrites_to_zero_handles_root_perfect_square_exp_fraction_partner_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "sqrt(2 * sqrt(x - 1) + x) - sqrt(x - 1) + exp(y*log(x)) - x^y + 1/(x + 1) - 1/(x - 1) + 2/(x^2 - 1) - 1",
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
fn simplify_pipeline_handles_log_exp_fraction_root_perfect_square_mix_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(exp(y*log(x)) - x^y) + (sin(x) + sin(y) - 2*sin((x+y)/2)*cos((x-y)/2)) + (2/(x^2 - 1) - 1/(x-1) + 1/(x+1)) + (sqrt(x + 2*sqrt(x-1)) - sqrt(x-1) - 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn exact_zero_subset_passthrough_shortcut_handles_triple_sine_against_log_with_polynomial_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "-atanh(sin(y)) + ln(sqrt((sin(y) + 1) / (1 - sin(y)))) - 2*cos(2*x) + sin(3*x) / sin(x) + y^3 + x*y^2 - 1",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_exact_zero_subset_passthrough_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected exact-zero subset passthrough shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "y^3 + x * y^2");
}
#[test]
fn binary_exact_zero_subset_passthrough_pair_shortcut_handles_triple_sine_against_log_with_polynomial_partner_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(-atanh(sin(y)) + ln(sqrt((sin(y) + 1) / (1 - sin(y)))) - 2*cos(2*x) + sin(3*x) / sin(x) + y^3 + x*y^2 - 1) + ((x^4 + y^4 - 2*x^2*y^2)/(x-y) - x^3 - y*x^2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_binary_exact_zero_subset_passthrough_pair_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected binary exact-zero subset passthrough shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn exact_one_shortcut_handles_nonadditive_log_power_pair_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((ln((x+1)^2)) + 1)/((2*ln(abs(x+1))) + 1)",
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
fn shifted_quotient_exact_one_gate_candidate_matches_nested_fraction_vs_log_product_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)",
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

    assert!(matches_shifted_quotient_exact_one_root_gate_candidate(
        &mut simplifier.context,
        numerator_core,
        denominator_core,
    ));
}
#[test]
fn shifted_quotient_exact_one_shortcut_handles_nested_fraction_vs_log_product_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)",
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
fn shifted_quotient_exact_one_shortcut_handles_log_product_vs_sophie_germain_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^4 + 4*y^4 - (x^2-2*x*y+2*y^2)*(x^2+2*x*y+2*y^2)) + 1)",
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
fn shifted_quotient_direct_small_zero_hot_gate_matches_log_product_vs_sophie_germain_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^4 + 4*y^4 - (x^2-2*x*y+2*y^2)*(x^2+2*x*y+2*y^2)) + 1)",
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
fn shifted_quotient_direct_small_zero_hot_gate_matches_log_product_vs_small_quotient_cancel_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((((x^2 - 1)/(x - 1) - (x+1))) + 1)",
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
fn shifted_quotient_direct_small_zero_hot_gate_matches_log_product_vs_telescoping_fraction_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((1/(u*(u+1)) - 1/u + 1/(u+1)) + 1)",
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
fn shifted_quotient_nested_zero_fast_gate_candidate_matches_log_product_vs_same_denominator_distribution_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((((a+b+c)/x - a/x - b/x - c/x)) + 1)",
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
fn shifted_quotient_nested_zero_fast_gate_candidate_matches_log_product_vs_difference_quotient_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((((x^2 - 1)/(x - 1) - (x+1))) + 1)",
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
fn shifted_quotient_nested_zero_fast_gate_candidate_matches_log_product_vs_geometric_difference_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1)",
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
fn stripped_positive_one_passthrough_preserves_log_product_and_geometric_difference_zero_families_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (numerator, denominator) = match simplifier.context.get(expr).clone() {
        Expr::Div(numerator, denominator) => (numerator, denominator),
        _ => panic!("expected division root"),
    };
    let numerator_core = strip_positive_one_passthrough_root(&mut simplifier.context, numerator)
        .unwrap_or_else(|| panic!("expected numerator passthrough core"));
    let denominator_core =
        strip_positive_one_passthrough_root(&mut simplifier.context, denominator)
            .unwrap_or_else(|| panic!("expected denominator passthrough core"));

    assert!(matches_direct_log_product_contract_zero_identity_root(
        &mut simplifier.context,
        numerator_core,
    ));
    assert!(matches_direct_small_zero_pair_root(
        &mut simplifier.context,
        numerator_core,
        denominator_core,
    ));
}
#[test]
fn simplify_pipeline_handles_tangent_addition_anchor_times_log_split_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(tan(x) + tan(y)) * (ln(sqrt(u)*v))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let rendered = render_expr(&simplifier.context, rewritten);
    assert!(rendered.contains("sin(x + y)"));
    assert!(rendered.contains("cos(x)"));
    assert!(rendered.contains("cos(y)"));
    assert!(rendered.contains("ln("));
    assert!(rendered.contains("u"));
    assert!(rendered.contains("v"));
    assert!(!rendered.contains("tan("));
}
#[test]
fn direct_small_zero_additive_combination_shortcut_handles_log_zero_leaf_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (ln((x*y)^2) - ln(x^2) - ln(y^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, steps) = super::try_standard_direct_small_zero_additive_combination_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        expr,
        true,
    )
    .unwrap_or_else(|| panic!("expected direct small-zero additive combination shortcut"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
    assert_eq!(steps.len(), 2);
}
#[test]
fn direct_small_zero_additive_combination_shortcut_handles_ln_abs_vs_sqrt_power_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))) + (sqrt(x^7) - x^3*sqrt(x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _steps) = super::try_standard_direct_small_zero_additive_combination_shortcut(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected direct small-zero additive combination shortcut"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_direct_log_square_product_split_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "log((x*y)^2) - log(x^2) - log(y^2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_log_square_product_split_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn detects_direct_log_square_product_split_zero_identity_with_scaled_general_base_terms_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "log(b,(x*y)^2) - 2*log(b,x) - 2*log(b,y)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_log_square_product_split_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));

    let squared_expr = parse(
        "log(b,(x*y)^2)^2 - (2*log(b,x)+2*log(b,y))^2",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_squared_exact_one_zero_identity_root(
        &mut simplifier.context,
        squared_expr
    ));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        squared_expr
    ));
}
#[test]
fn simplify_pipeline_handles_general_base_log_grouped_power_squared_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((log(b,(x*y)^2))^2) + m) - (((2*log(b,x)+2*log(b,y))^2) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_direct_log_product_contract_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("ln(x^3) + ln(y^2) - ln(x^3 * y^2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_log_product_contract_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn detects_direct_log_difference_squares_split_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "ln(x^2 - y^2) - ln(x - y) - ln(x + y)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        matches_direct_log_difference_squares_split_zero_identity_root(
            &mut simplifier.context,
            expr
        )
    );
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
    assert!(matches_direct_small_zero_or_known_pair_base_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn detects_direct_ln_abs_product_split_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_ln_abs_product_split_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn simplify_pipeline_handles_log_square_vs_ln_abs_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(log((x*y)^2) - log(x^2) - log(y^2)) - (2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
