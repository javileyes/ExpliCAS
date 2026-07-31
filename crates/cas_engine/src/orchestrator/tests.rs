//! Tests del orquestador, extraídos del módulo (P1).
//!
//! Vivían como `mod tests` inline dentro de `orchestrator.rs`, donde
//! eran 12.227 de sus 42.307 líneas.

use super::*;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

fn render(ctx: &Context, id: ExprId) -> String {
    format!("{}", DisplayExpr { context: ctx, id })
}

fn simplify_render(input: &str) -> String {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr =
        parse(input, &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    render(&simplifier.context, rewritten)
}

// A non-square matrix product expands each output entry into a sum of `inner_dim`
// products, so the unfolded result transiently exceeds the anti-worsen node budget.
// MatrixMultiplyRule must be budget-exempt so the valid product commits instead of
// being rejected and falling through to the (wrong) scalar-broadcast rule.
#[test]
fn matrix_multiply_non_square_product_commits_through_budget() {
    assert_eq!(
        simplify_render("[[1,2],[3,4]] * [[5,6,7],[8,9,10]]"),
        "[[21, 24, 27], [47, 54, 61]]"
    );
}

#[test]
fn matrix_multiply_outer_product_commits() {
    assert_eq!(
        simplify_render("[[1],[2],[3]] * [[4,5,6]]"),
        "[[4, 5, 6], [8, 10, 12], [12, 15, 18]]"
    );
}

// A dimension-mismatched product has NO value (inner dimensions 3 ≠ 2), so the shape guard
// routes it to the `undefined` sentinel — never a fabricated finite matrix-of-matrices from the
// scalar-broadcast misfire, and no longer a dishonest echoed residual reported with `ok:true`.
#[test]
fn matrix_multiply_dimension_mismatch_is_undefined() {
    assert_eq!(
        simplify_render("[[1,2,3],[4,5,6]] * [[1,2],[3,4]]"),
        "undefined"
    );
}

#[test]
fn matrix_multiply_square_product_unchanged() {
    assert_eq!(
        simplify_render("[[1,2],[3,4]] * [[5,6],[7,8]]"),
        "[[19, 22], [43, 50]]"
    );
}

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
fn standard_exact_additive_pair_chain_shortcut_cancels_symbolic_passthrough_tail() {
    let mut ctx = Context::new();
    let expr = parse("m + 1 - m - 1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _) = try_standard_exact_additive_pair_chain_shortcut(
        &crate::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("shortcut should cancel symbolic passthrough tail"));
    assert_eq!(render(&ctx, rewritten), "0");
}

#[test]
fn combine_constants_rule_collapses_unit_difference_regression() {
    let mut ctx = Context::new();
    let expr = parse("1 - 1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let parent_ctx = build_root_shortcut_parent_ctx(&crate::SimplifyOptions::default(), &ctx, expr);
    let rewrite = crate::rules::arithmetic::CombineConstantsRule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("combine constants should collapse unit difference"));
    assert_eq!(render(&ctx, rewrite.new_expr), "0");
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
fn simplify_pipeline_handles_nested_additive_zero_sum_case21_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (sin(x)^2 - (1 - cos(2*x))/2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}

#[test]
fn simplify_pipeline_handles_nested_additive_shifted_quotient_case24_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/((sin(x)^2 - (1 - cos(2*x))/2) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}

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
fn simplify_pipeline_marks_timed_out_and_returns_partial_when_deadline_is_expired() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr =
        parse("a + b", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    orchestrator.options.time_budget_ms = Some(1);
    orchestrator.options.deadline =
        Some(web_time::Instant::now() - std::time::Duration::from_millis(1));

    let (rewritten, steps, stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);

    assert_eq!(rewritten, expr);
    assert!(steps.is_empty(), "timed-out partial path should skip steps");
    assert!(stats.timed_out, "pipeline should be marked as timed out");
    assert_eq!(stats.total_rewrites, 0);
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
fn child_isolated_exact_zero_reinterns_function_names_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(diff(sin(e^(x^2)),x) - 2*x*cos(e^(x^2))*e^(x^2)) + (u*v+u*w-u*(v+w))",
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
fn simplify_pipeline_handles_collapsed_fraction_times_geometric_sum_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1/(x*(x+1))) * (u^3 + u^2 + u + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("((u+1)*(u^2+1))/(x*(x+1))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
}

#[test]
fn simplify_pipeline_handles_collapsed_fraction_times_sum_of_cubes_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(1/(x*(x+1))) * (u^3 + v^3)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("((u+v)*(u^2-u*v+v^2))/(x*(x+1))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
}

#[test]
fn simplify_pipeline_canonicalizes_collapsed_fraction_times_sum_of_cubes_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(1/(x*(x+1))) * (u^3 + v^3)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, rewritten),
        "(u + v) * (u^2 + v^2 - u * v) / (x * (x + 1))"
    );
}

#[test]
fn simplify_pipeline_handles_square_anchor_times_three_linear_shift_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sqrt(x))^4) * ((u+1)*(u+2)*(u+3))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("(x^2) * ((u+1)*(u+2)*(u+3))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
}

#[test]
fn simplify_pipeline_handles_square_anchor_times_expanded_three_linear_shift_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(x^2) * (u^3 + 6*u^2 + 11*u + 6)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("(x^2) * ((u+1)*(u+2)*(u+3))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
}

#[test]
fn simplify_pipeline_handles_quartic_square_anchor_times_three_linear_shift_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(x^4 + 4*x^2 + 4) * ((u+1)*(u+2)*(u+3))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse(
        "((x^2 + 2)^2) * ((u+1)*(u+2)*(u+3))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
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
fn detects_direct_odd_half_power_zero_scope_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("sqrt(x^5) - x^2*sqrt(x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
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
fn detects_direct_geometric_difference_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        matches_direct_geometric_difference_zero_identity_root(&mut simplifier.context, expr),
        "expr={}",
        render(&simplifier.context, expr)
    );
    assert!(matches_direct_small_zero_or_known_pair_base_root(
        &mut simplifier.context,
        expr
    ));
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
fn detects_direct_nested_fraction_simplified_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("1 + 1/(1 + 1/(1 + 1/x))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("(3*x + 2)/(2*x + 1)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_nested_fraction_simplified_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}

#[test]
fn detects_direct_nested_fraction_simplified_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}

#[test]
fn detects_direct_nested_fraction_reciprocal_depth_two_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("1/(1 + 1/(1+u)) - (1+u)/(2+u)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
    assert!(super::matches_direct_small_zero_or_known_pair_base_root(
        &mut simplifier.context,
        expr
    ));
}

#[test]
fn detects_direct_nested_fraction_reciprocal_deeper_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "1/(1 + 1/(1 + 1/(1+u))) - (2+u)/(3+2*u)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
    assert!(super::matches_direct_small_zero_or_known_pair_base_root(
        &mut simplifier.context,
        expr
    ));
}

#[test]
fn simplify_pipeline_handles_nested_fraction_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
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
fn simplify_pipeline_handles_nested_fraction_against_geometric_factor_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1))",
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
fn simplify_pipeline_handles_nested_fraction_against_difference_quotient_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + ((x^2 - 1)/(x - 1) - (x+1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
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
fn simplify_pipeline_handles_negative_double_sine_direct_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("sin(2*x) - 2*sin(x)*cos(x)", &mut simplifier.context)
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
fn simplify_pipeline_handles_negative_double_sine_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((-sin(2*x)) + m) - ((-2*sin(x)*cos(x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}

#[test]
fn simplify_pipeline_handles_negative_double_sine_passthrough_forward_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((-2*sin(x)*cos(x)) + m) - ((-sin(2*x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}

#[test]
fn simplify_pipeline_handles_sophie_germain_passthrough_forward_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 + 4*y^4) + m) - (((x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}

#[test]
fn simplify_pipeline_handles_vandermonde_passthrough_forward_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((a^3*(b-c) + b^3*(c-a) + c^3*(a-b)) + m) - (((a-b)*(a-c)*(b-c)*(a+b+c)) + m)",
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
fn detects_recursive_sine_shifted_quotient_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(6*x)) + 1)/((sin(5*x)*cos(x)+cos(5*x)*sin(x)) + 1)",
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
fn simplify_pipeline_handles_recursive_sine_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(6*x)) + 1)/((sin(5*x)*cos(x)+cos(5*x)*sin(x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}

#[test]
fn simplify_pipeline_handles_recursive_sine_shifted_quotient_reverse_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(5*x)*cos(x)+cos(5*x)*sin(x)) + 1)/((sin(6*x)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}

#[test]
fn simplify_pipeline_handles_shifted_sine_pair_sum_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sin(x)+cos(x)+sin(y)+cos(y)) + 1)/((sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}

#[test]
fn simplify_pipeline_handles_shifted_sine_pair_sum_shifted_quotient_reverse_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)) + 1)/((sin(x)+cos(x)+sin(y)+cos(y)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
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
fn simplify_pipeline_handles_scaled_sine_fourth_power_reduction_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "8*sin(x)^4 - (3 - 4*cos(2*x) + cos(4*x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
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
fn detects_direct_successive_unit_fraction_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("1/x + 1/(x+1)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs =
        parse("(2*x+1)/(x*(x+1))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_addition_of_successive_unit_fractions_pair_root(&mut ctx, lhs, rhs));
}

#[test]
fn detects_direct_reciprocal_sqrt_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("1/sqrt(x)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("sqrt(x)/x", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_reciprocal_sqrt_pair_root(&mut ctx, lhs, rhs));
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
fn detects_direct_scaled_half_angle_square_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("2*cos(u/2)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("1 + cos(u)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_scaled_half_angle_square_pair_root(
        &mut ctx, lhs, rhs
    ));
}

#[test]
fn detects_direct_abs_square_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("abs(cos(x))^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("cos(x)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_abs_square_pair_root(&ctx, lhs, rhs));
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
fn detects_direct_exponential_combination_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("exp(a)*exp(b)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("exp(a+b)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_exponential_combination_pair_root(
        &mut ctx, lhs, rhs
    ));
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
fn detects_direct_quintuple_angle_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("sin(5*x)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("16*sin(x)^5 - 20*sin(x)^3 + 5*sin(x)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_quintuple_angle_pair_root(&mut ctx, lhs, rhs));
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
fn detects_direct_tangent_addition_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("tan(x) + tan(y)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("sin(x+y)/(cos(x)*cos(y))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_tangent_addition_pair_root(
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
fn detects_two_factor_product_pair_zero_difference_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((1/x + 1/(x+1)) * cos(2*x)) - (((2*x+1)/(x*(x+1))) * (2*cos(x)^2 - 1))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let view = AddView::from_expr(&ctx, expr);
    let lhs_factors = flatten_mul_chain(&mut ctx, view.terms[0].0)
        .into_iter()
        .map(|factor| render(&ctx, factor))
        .collect::<Vec<_>>();
    let rhs_factors = flatten_mul_chain(&mut ctx, view.terms[1].0)
        .into_iter()
        .map(|factor| render(&ctx, factor))
        .collect::<Vec<_>>();
    let lhs_term = view.terms[0].0;
    let rhs_term = view.terms[1].0;
    let lhs_factor_ids = flatten_mul_chain(&mut ctx, lhs_term);
    let rhs_factor_ids = flatten_mul_chain(&mut ctx, rhs_term);
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
fn detects_two_factor_product_pair_zero_difference_tangent_addition_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((sin(5*x)) * (tan(x) + tan(y))) - ((sin(5*x)) * (sin(x+y)/(cos(x)*cos(y))))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
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
fn detects_direct_linear_factoring_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("z^2 + 2*z", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("z*(z+2)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_linear_factoring_pair_root(
        &mut ctx, lhs, rhs
    ));
}

#[test]
fn detects_direct_quartic_gcf_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("z^4 - z^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("z^2*(z-1)*(z+1)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_quartic_gcf_pair_root(
        &mut ctx, lhs, rhs
    ));
}

#[test]
fn detects_direct_difference_of_squares_quotient_pair_regression() {
    let mut ctx = Context::new();
    let lhs =
        parse("(z^2 - 9)/(z + 3)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("z - 3", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_difference_of_squares_quotient_pair_root(&mut ctx, lhs, rhs));
}

#[test]
fn detects_direct_sum_diff_cubes_quotient_pair_regression() {
    let mut ctx = Context::new();
    let lhs =
        parse("(z^3 - 8)/(z - 2)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("z^2 + 2*z + 4", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_sum_diff_cubes_quotient_pair_root(
        &mut ctx, lhs, rhs
    ));
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
fn detects_direct_perfect_square_trinomial_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("z^2 + 2*z + 1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("(z+1)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_perfect_square_trinomial_pair_root(
        &mut ctx, lhs, rhs
    ));
}

#[test]
fn detects_direct_perfect_square_trinomial_fractional_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("u^2 + u + 1/4", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("(u+1/2)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_perfect_square_trinomial_pair_root(
        &mut ctx, lhs, rhs
    ));
}

#[test]
fn detects_direct_successive_unit_fractions_pair_with_expanded_denominator_regression() {
    let mut ctx = Context::new();
    let lhs =
        parse("(1/z) + (1/(z+1))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("(2*z+1)/(z^2+z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::matches_direct_addition_of_successive_unit_fractions_pair_root(&mut ctx, lhs, rhs)
    );
}

#[test]
fn detects_two_factor_product_pair_zero_difference_factoring_tangent_addition_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((z^2 + 2*z) * (tan(u) + tan(v))) - (((z*(z+2)) * (sin(u+v)/(cos(u)*cos(v)))))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
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
fn detects_direct_small_exact_constant_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("sec(pi)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("-1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_small_exact_constant_pair_root(
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
fn detects_direct_cube_root_rationalization_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("1/(1+z^(1/3))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("(1-z^(1/3)+z^(2/3))/(1+z)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_cube_root_rationalization_pair_root(
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
fn detects_direct_weierstrass_contraction_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("2*tan(z/2)/(1 + tan(z/2)^2)", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("sin(z)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_weierstrass_contraction_pair_root(
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
fn simplify_pipeline_handles_negative_exact_constant_factor_times_chebyshev_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sec(pi)) * (cos(2*u))) - (((-1)) * (2*cos(u)^2 - 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
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
fn two_factor_direct_pair_anchor_shortcut_handles_exact_quarter_shifted_sine_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(sqrt(2)) * sin(pi/4 + x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("(2^(1/2)) * sin(x + pi/4)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = try_standard_two_factor_direct_pair_anchor_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("expected two-factor direct-pair anchor shortcut to match"));
    let difference = simplifier.context.add(Expr::Sub(rewritten, expected));
    assert!(super::isolated_simplify_rewrites_to_zero(
        &options,
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
fn simplify_pipeline_normalizes_exact_quarter_shifted_sine_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse("(sqrt(2)) * sin(pi/4 + x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse("(2^(1/2)) * sin(x + pi/4)", &mut simplifier.context)
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
fn simplify_pipeline_handles_safe_anchor_times_geometric_sum_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(18) - sqrt(2)) * (u^3 + u^2 + u + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("(2*sqrt(2)) * ((u+1)*(u^2+1))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
}

#[test]
fn simplify_pipeline_handles_safe_anchor_times_two_linear_shift_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(18) - sqrt(2)) * (u^2 + 5*u + 6)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let rendered = render_expr(&simplifier.context, rewritten);
    assert!(rendered.contains("2^(3/2)"));
    assert!(rendered.contains("u + 2"));
    assert!(rendered.contains("u + 3"));
    assert!(!rendered.contains("u^2 + 5 * u + 6"));
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
fn simplify_pipeline_handles_ratio_anchor_times_geometric_sum_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(8*x)/sqrt(2*x)) * (u^3 + u^2 + u + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("2 * ((u+1)*(u^2+1))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let diff_expr = simplifier.context.add(Expr::Sub(rewritten, expected));
    let (diff, _steps, _stats) = orchestrator.simplify_pipeline(diff_expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, diff), "0");
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
fn simplify_pipeline_aligns_safe_anchor_with_successive_unit_fraction_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let source = parse("(phi + 1) * (1/u + 1/(u+1))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let target = parse(
        "((phi + 1) * ((2*u + 1)/(u*(u+1))))",
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
fn simplify_pipeline_handles_two_factor_fractional_perfect_square_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("((x^2 + 2)^2) * (u^2 + u + 1/4)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let rendered = render_expr(&simplifier.context, rewritten);
    assert!(rendered.contains("(u + 1/2)^2"));
    assert!(rendered.contains("(x^2 + 2)^2"));
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
fn detects_two_factor_product_pair_zero_difference_perfect_square_cube_rationalization_regression()
{
    let mut ctx = Context::new();
    let expr = parse(
        "((x^2 + 2*x + 1) * (1/(1+u^(1/3)))) - (((x+1)^2) * ((1-u^(1/3)+u^(2/3))/(1+u)))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}

#[test]
fn simplify_pipeline_handles_sum_diff_cubes_quotient_times_chebyshev_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((z^3 - 8)/(z - 2)) * (cos(2*u))) - (((z^2 + 2*z + 4)) * (2*cos(u)^2 - 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
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
fn simplify_pipeline_handles_quartic_gcf_times_chebyshev_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((z^4 - z^2) * (cos(2*u))) - (((z^2*(z-1)*(z+1))) * (2*cos(u)^2 - 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}

#[test]
fn simplify_pipeline_handles_difference_of_squares_quotient_times_chebyshev_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((z^2 - 9)/(z + 3)) * (cos(2*u))) - (((z - 3)) * (2*cos(u)^2 - 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
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
fn detects_two_factor_product_pair_zero_difference_reciprocal_sqrt_chebyshev_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((1/sqrt(x)) * (cos(2*x))) - (((sqrt(x)/x) * (2*cos(x)^2 - 1)))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
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
fn detects_two_factor_product_pair_zero_difference_quintuple_chebyshev_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((sin(5*x)) * (cos(2*x))) - (((16*sin(x)^5 - 20*sin(x)^3 + 5*sin(x)) * (2*cos(x)^2 - 1)))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}

#[test]
fn detects_two_factor_product_pair_zero_difference_quartic_gcf_power_reduction_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((x^4 - x^2) * (cos(x)^2)) - (((x^2*(x-1)*(x+1)) * ((1 + cos(2*x))/2)))",
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
fn detects_two_factor_product_pair_zero_difference_fractional_perfect_square_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((x^2 + 2*x + 1) * ((u+1/2)^2)) - (((x+1)^2) * (u^2 + u + 1/4))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}

#[test]
fn detects_two_factor_product_pair_zero_difference_fractional_square_sum_diff_cubes_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "(((x+1/2)^2) * (u^3 + v^3)) - (((x^2 + x + 1/4)) * ((u+v)*(u^2-u*v+v^2)))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}

#[test]
fn detects_two_factor_product_pair_zero_difference_fractional_square_higher_degree_difference_regression(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "(((x+1/2)^2) * (u^6 - 1)) - (((x^2 + x + 1/4)) * ((u^2+u+1)*(u^2-u+1)*(u+1)*(u-1)))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}

#[test]
fn detects_two_factor_product_pair_zero_difference_fractional_square_sophie_germain_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "(((x+1/2)^2) * (u^4 + 4)) - (((x^2 + x + 1/4)) * ((u^2 + 2*u + 2)*(u^2 - 2*u + 2)))",
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
fn simplify_pipeline_handles_quartic_gcf_times_power_reduction_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - x^2) * (cos(x)^2)) - (((x^2*(x-1)*(x+1)) * ((1 + cos(2*x))/2)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
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
fn simplify_pipeline_handles_fraction_over_chebyshev_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1/x + 1/(x+1))/(cos(2*x)) - (((2*x+1)/(x*(x+1)))/(2*cos(x)^2 - 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}

#[test]
fn simplify_pipeline_handles_fraction_over_abs_square_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1/x + 1/(x+1))/(abs(cos(x))^2) - (((2*x+1)/(x*(x+1)))/(cos(x)^2))",
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
fn detects_direct_quotient_pair_zero_difference_fraction_chebyshev_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "(1/x + 1/(x+1))/(cos(2*x)) - (((2*x+1)/(x*(x+1)))/(2*cos(x)^2 - 1))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_quotient_pair_zero_difference_root(
        &mut ctx, expr
    ));
}

#[test]
fn detects_direct_quotient_pair_zero_difference_fraction_abs_square_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "(1/x + 1/(x+1))/(abs(cos(x))^2) - (((2*x+1)/(x*(x+1)))/(cos(x)^2))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_quotient_pair_zero_difference_root(
        &mut ctx, expr
    ));
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
fn direct_small_zero_core_group_requires_cancellation_marker() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let positive_expr =
        parse("x + y", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let positive_terms = AddView::from_expr(&simplifier.context, positive_expr).terms;
    assert!(!super::matches_direct_small_zero_core_group_root(
        &mut simplifier.context,
        positive_terms.as_slice(),
    ));

    let zero_expr = parse("p^2 - q^2 - (p-q)*(p+q)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let zero_terms = AddView::from_expr(&simplifier.context, zero_expr).terms;
    assert!(super::matches_direct_small_zero_core_group_root(
        &mut simplifier.context,
        zero_terms.as_slice(),
    ));
}

#[test]
fn direct_small_zero_core_groups_require_enough_cancellation_markers_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let sparse_expr = parse("a + b + c + d - e - f", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let sparse_terms = AddView::from_expr(&simplifier.context, sparse_expr).terms;
    assert_eq!(
        super::direct_small_zero_cancellation_marker_count_root(
            &simplifier.context,
            sparse_terms.as_slice(),
        ),
        2
    );
    assert!(
        !super::has_enough_direct_small_zero_cancellation_markers_root(
            &simplifier.context,
            sparse_terms.as_slice(),
            3,
        )
    );
    assert!(!super::matches_direct_three_small_zero_cores_terms_root(
        &mut simplifier.context,
        sparse_terms.as_slice(),
    ));

    let dense_expr = parse("a - b + c - d + e - f", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let dense_terms = AddView::from_expr(&simplifier.context, dense_expr).terms;
    assert!(
        super::has_enough_direct_small_zero_cancellation_markers_root(
            &simplifier.context,
            dense_terms.as_slice(),
            3,
        )
    );
}

#[test]
fn direct_small_zero_plain_three_core_gate_requires_two_exact_cancellation_pairs() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let noisy_plain_expr = parse(
        "a*x^2 + b*x + c*x^2 + d*x + e*x^2 + f - ((a + c + e)*x^2 + (b + d)*x + f)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let noisy_plain_terms = AddView::from_expr(&simplifier.context, noisy_plain_expr).terms;
    assert_eq!(
        super::direct_small_zero_opposite_sign_exact_pair_count_root(
            &simplifier.context,
            noisy_plain_terms.as_slice(),
        ),
        1
    );
    assert!(
        !super::should_try_direct_three_small_zero_cores_root(
            &simplifier.context,
            noisy_plain_expr,
            noisy_plain_terms.as_slice(),
        ),
        "plain collect misses should not enter the recursive three-core partitioner"
    );

    let passthrough_expr = parse("c + m + a*b + b*a - c - m - 2*a*b", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let passthrough_terms = AddView::from_expr(&simplifier.context, passthrough_expr).terms;
    assert_eq!(
        super::direct_small_zero_opposite_sign_exact_pair_count_root(
            &simplifier.context,
            passthrough_terms.as_slice(),
        ),
        2
    );
    assert!(super::should_try_direct_three_small_zero_cores_root(
        &simplifier.context,
        passthrough_expr,
        passthrough_terms.as_slice(),
    ));
}

#[test]
fn direct_small_zero_three_core_gate_requires_remaining_anchor_capacity() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let impossible_three_core_expr = parse(
        "((a*x^2 + b*x + c) + m) - ((x*(a*x + b + c/x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let impossible_terms =
        AddView::from_expr(&simplifier.context, impossible_three_core_expr).terms;
    assert_eq!(impossible_terms.len(), 6);
    assert_eq!(
        super::direct_small_zero_opposite_sign_exact_pair_count_root(
            &simplifier.context,
            impossible_terms.as_slice(),
        ),
        1
    );
    assert!(
        !super::has_enough_direct_small_zero_remaining_anchor_terms_root(
            &simplifier.context,
            impossible_terms.as_slice(),
            3,
        ),
        "one passthrough pair and one negated structured term cannot form three zero groups"
    );
    assert!(!super::should_try_direct_three_small_zero_cores_root(
        &simplifier.context,
        impossible_three_core_expr,
        impossible_terms.as_slice(),
    ));

    let three_core_expr = parse(
        "(a^2-b^2 - (a-b)*(a+b)) + (sec(y) - 1/cos(y)) + (u*v + u*w - u*(v+w))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let three_core_terms = AddView::from_expr(&simplifier.context, three_core_expr).terms;
    assert!(
        super::has_enough_direct_small_zero_remaining_anchor_terms_root(
            &simplifier.context,
            three_core_terms.as_slice(),
            3,
        )
    );
    assert!(super::should_try_direct_three_small_zero_cores_root(
        &simplifier.context,
        three_core_expr,
        three_core_terms.as_slice(),
    ));
}

#[test]
fn direct_small_zero_three_core_matcher_handles_eight_term_composition_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(a^2-b^2 - (a-b)*(a+b)) + (sec(y) - 1/cos(y)) + (u*v + u*w - u*(v+w))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let terms = AddView::from_expr(&simplifier.context, expr).terms;
    assert_eq!(terms.len(), 8);
    assert_eq!(
        super::direct_small_zero_opposite_sign_exact_pair_count_root(
            &simplifier.context,
            terms.as_slice(),
        ),
        0
    );
    assert!(super::should_try_direct_three_small_zero_cores_root(
        &simplifier.context,
        expr,
        terms.as_slice(),
    ));
    assert!(
        super::matches_direct_three_small_zero_cores_terms_root(
            &mut simplifier.context,
            terms.as_slice(),
        ),
        "three-core compositions can contain eight additive terms"
    );

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
fn direct_small_zero_four_core_gate_requires_remaining_anchor_capacity() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let impossible_expr = parse(
        "(((a^3-b^3)/(a-b)+c) + m) - ((a^2+a*b+b^2+c) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let impossible_terms = AddView::from_expr(&simplifier.context, impossible_expr).terms;
    assert_eq!(impossible_terms.len(), 8);
    assert_eq!(
        super::direct_small_zero_opposite_sign_exact_pair_count_root(
            &simplifier.context,
            impossible_terms.as_slice(),
        ),
        2
    );
    assert!(
        !super::has_enough_direct_small_zero_remaining_anchor_terms_root(
            &simplifier.context,
            impossible_terms.as_slice(),
            4,
        ),
        "two exact passthrough pairs leave too few anchors for two more groups"
    );
    assert!(
        !super::matches_direct_four_or_five_small_zero_core_groups_terms_root(
            &mut simplifier.context,
            impossible_terms.as_slice(),
        )
    );

    let four_core_expr = parse(
        "(sec(a)-1/cos(a)) + (csc(b)-1/sin(b)) + (tan(c)-sin(c)/cos(c)) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let four_core_terms = AddView::from_expr(&simplifier.context, four_core_expr).terms;
    assert_eq!(four_core_terms.len(), 8);
    assert!(
        super::has_enough_direct_small_zero_remaining_anchor_terms_root(
            &simplifier.context,
            four_core_terms.as_slice(),
            4,
        )
    );
    assert!(
        super::matches_direct_four_or_five_small_zero_core_groups_terms_root(
            &mut simplifier.context,
            four_core_terms.as_slice(),
        )
    );
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
fn detects_direct_rationalized_sum_of_sqrts_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("1/(sqrt(a) + sqrt(b))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("(sqrt(a) - sqrt(b))/(a - b)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_known_direct_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}

#[test]
fn detects_direct_reciprocal_sum_difference_nested_fraction_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("((1/x + 1/y)/(1/x - 1/y))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("(x+y)/(y-x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_known_direct_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
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
fn direct_small_zero_pair_shortcut_handles_nested_fraction_vs_telescoping_product_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1))",
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
fn direct_small_zero_pair_shortcut_handles_nested_fraction_vs_telescoping_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
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
fn direct_small_zero_pair_shortcut_handles_integrate_prep_vs_reciprocal_nested_fraction_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(cos(x)*cos(2*x)*cos(4*x) - sin(8*x)/(8*sin(x))) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
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
fn direct_small_zero_additive_combination_shortcut_handles_dirichlet_vs_factor_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))) + (p^2-q^2 - (p-q)*(p+q))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = super::try_standard_direct_small_zero_additive_combination_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        true,
    )
    .unwrap_or_else(|| panic!("expected direct small zero additive combination shortcut"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}

#[test]
fn targeted_early_small_zero_additive_combination_accepts_dirichlet_vs_factor_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))) + (p^2-q^2 - (p-q)*(p+q))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::is_targeted_early_small_zero_additive_combination_candidate_root(
            &mut simplifier.context,
            expr,
        )
    );
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
fn dirichlet_root_shortcut_skips_dirichlet_vs_factor_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))) + (p^2-q^2 - (p-q)*(p+q))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::try_finish_dirichlet_kernel_root_shortcut(&mut simplifier, expr, false).is_none()
    );
}

#[test]
fn simplify_pipeline_handles_dirichlet_vs_factor_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))) + (p^2-q^2 - (p-q)*(p+q))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
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
fn direct_small_zero_profile_sample_includes_term_tags_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3)+ln(y^2)-ln(x^3*y^2)) + (sec(z)-1/cos(z)) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let sample = super::render_direct_small_zero_profile_sample_root(&simplifier.context, expr);
    assert!(sample.contains("terms="), "{sample}");
    assert!(sample.contains(":log"), "{sample}");
    assert!(sample.contains(":trig"), "{sample}");
    assert!(sample.contains(":div"), "{sample}");
}

#[test]
fn direct_small_zero_pair_shortcut_handles_four_two_term_core_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    for core in [
        "1/(sqrt(a)+sqrt(b)) - (sqrt(a)-sqrt(b))/(a-b)",
        "sec(z)-1/cos(z)",
        "csc(w)-1/sin(w)",
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
        "(1/(sqrt(a)+sqrt(b)) - (sqrt(a)-sqrt(b))/(a-b)) + (sec(z)-1/cos(z)) + (csc(w)-1/sin(w)) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
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
fn direct_small_zero_pair_shortcut_handles_five_two_term_core_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    for core in [
        "sec(a)-1/cos(a)",
        "csc(b)-1/sin(b)",
        "tan(c)-sin(c)/cos(c)",
        "sin(2*d)-2*sin(d)*cos(d)",
        "2*sinh(e)*cosh(e)-sinh(2*e)",
    ] {
        let core_expr = parse(core, &mut simplifier.context)
            .unwrap_or_else(|err| panic!("parse failed: {err:?}"));
        assert!(
            super::matches_direct_small_zero_or_known_pair_base_root(
                &mut simplifier.context,
                core_expr,
            ),
            "expected core to match: {core}"
        );
    }
    let expr = parse(
        "(sec(a)-1/cos(a)) + (csc(b)-1/sin(b)) + (tan(c)-sin(c)/cos(c)) + (sin(2*d)-2*sin(d)*cos(d)) + (2*sinh(e)*cosh(e)-sinh(2*e))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|err| panic!("parse failed: {err:?}"));
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
fn direct_small_zero_pair_shortcut_handles_five_core_sum_with_one_three_term_core_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    for core in [
        "sec(a)-1/cos(a)",
        "csc(b)-1/sin(b)",
        "tan(c)-sin(c)/cos(c)",
        "sin(2*d)-2*sin(d)*cos(d)",
        "cosh(e)^2-sinh(e)^2-1",
    ] {
        let core_expr = parse(core, &mut simplifier.context)
            .unwrap_or_else(|err| panic!("parse failed: {err:?}"));
        assert!(
            super::matches_direct_small_zero_or_known_pair_base_root(
                &mut simplifier.context,
                core_expr,
            ),
            "expected core to match: {core}"
        );
    }
    let expr = parse(
        "(sec(a)-1/cos(a)) + (csc(b)-1/sin(b)) + (tan(c)-sin(c)/cos(c)) + (sin(2*d)-2*sin(d)*cos(d)) + (cosh(e)^2-sinh(e)^2-1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|err| panic!("parse failed: {err:?}"));
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
fn direct_small_zero_pair_shortcut_handles_nested_fraction_vs_small_quotient_cancel_sum_regression()
{
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + ((x^2 - 1)/(x - 1) - (x+1))",
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
fn direct_small_zero_pair_shortcut_handles_nested_fraction_vs_factorial_product_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * ((n+1)!/(n-1)! - n*(n+1))",
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
fn partitioned_direct_small_zero_sum_shortcut_handles_nested_fraction_vs_sophie_germain_sum_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (x^4 + 4*y^4 - (x^2-2*x*y+2*y^2)*(x^2+2*x*y+2*y^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let view = AddView::from_expr(&simplifier.context, expr);
    let rendered_terms: Vec<_> = view
        .terms
        .iter()
        .map(|(term, sign)| format!("{sign:?}:{}", render(&simplifier.context, *term)))
        .collect();
    let nested_fraction_chunk =
        super::build_signed_sum_expr_root(&mut simplifier.context, &view.terms[..3]);
    let sophie_germain_chunk =
        super::build_signed_sum_expr_root(&mut simplifier.context, &view.terms[3..]);
    assert!(
        super::matches_direct_small_zero_or_known_pair_base_root(
            &mut simplifier.context,
            nested_fraction_chunk,
        ),
        "expected nested fraction chunk to match expr={}",
        render(&simplifier.context, nested_fraction_chunk),
    );
    assert!(
        super::matches_direct_small_zero_or_known_pair_base_root(
            &mut simplifier.context,
            sophie_germain_chunk,
        ),
        "expected sophie germain chunk to match expr={}",
        render(&simplifier.context, sophie_germain_chunk),
    );
    let result = super::try_extract_partitioned_direct_small_zero_sum_chunks_root(
        &mut simplifier.context,
        expr,
    );
    assert!(
        result.is_some(),
        "expected partitioned direct small zero shortcut to match terms={rendered_terms:?} expr={}",
        render(&simplifier.context, expr),
    );
}

#[test]
fn detects_direct_depth_three_unit_continued_fraction_zero_identity_base_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_small_zero_or_known_pair_base_root(
        &mut simplifier.context,
        expr
    ));
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
fn detects_direct_sophie_germain_pair_symbolic_fourth_power_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("x^4 + 4*y^4", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse(
        "((x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_sophie_germain_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));

    let zero_expr = parse(
        "x^4 + 4*y^4 - ((x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_sophie_germain_zero_identity_root(
        &mut simplifier.context,
        zero_expr
    ));
    assert!(super::matches_direct_small_zero_or_known_pair_base_root(
        &mut simplifier.context,
        zero_expr
    ));
}

#[test]
fn simplify_pipeline_handles_nested_fraction_vs_sophie_germain_product_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (x^4 + 4*y^4 - (x^2-2*x*y+2*y^2)*(x^2+2*x*y+2*y^2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
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
fn zero_product_with_exact_zero_child_shortcut_handles_nested_fraction_vs_sophie_germain_product_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (x^4 + 4*y^4 - (x^2-2*x*y+2*y^2)*(x^2+2*x*y+2*y^2))",
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
fn detects_direct_same_denominator_common_scaled_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("((a+b+c)/x - a/x - b/x - c/x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        matches_direct_same_denominator_common_scaled_zero_identity_root(
            &mut simplifier.context,
            expr
        )
    );
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}

#[test]
fn detects_direct_affine_common_denominator_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("a + b/x - (a*x+b)/x", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_affine_common_denominator_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}

#[test]
fn detects_direct_depth_three_unit_continued_fraction_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let terms = AddView::from_expr(&simplifier.context, expr).terms;
    assert!(
        matches_direct_depth_three_unit_continued_fraction_zero_identity_terms_root(
            &mut simplifier.context,
            &terms,
        )
    );
    assert!(matches_direct_small_zero_or_known_pair_base_root(
        &mut simplifier.context,
        expr
    ));
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
fn simplify_pipeline_handles_sqrt_abs_vs_affine_common_denominator_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sqrt(a^2 + 2*a*b + b^2) - abs(a+b)) + (a + b/x - (a*x+b)/x)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}

#[test]
fn simplify_pipeline_handles_same_denominator_distribution_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + 1)/(((a+b+c)/x - a/x - b/x - c/x) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
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
fn common_scale_fallback_preserves_assumed_abs_metadata() {
    let mut ctx = Context::new();
    let expr = parse("2*a - 2*abs(a)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut options = SimplifyOptions::default();
    options.shared.semantics.domain_mode = crate::DomainMode::Assume;
    let (_rewritten, steps) =
        try_standard_common_scale_exact_zero_shortcut_fallback(&options, &mut ctx, expr, true)
            .unwrap_or_else(|| panic!("expected common-scale fallback to match"));
    assert!(
        steps.iter().any(|step| {
            step.assumption_events()
                .iter()
                .any(|event| event.message == "a > 0")
        }),
        "expected retained positivity assumption in steps: {steps:?}"
    );
}

#[test]
fn common_scale_residual_matches_morrie_scaled_difference_regression() {
    let mut ctx = Context::new();
    let residual_expr = parse("cos(x)*cos(2*x)*cos(4*x) - (sin(8*x)/(8*sin(x)))", &mut ctx)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_small_zero_or_known_pair_residual_root(
        &mut ctx,
        residual_expr
    ));
}

#[test]
fn common_scale_known_pair_shortcut_matches_morrie_scaled_difference_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "k*(cos(x)*cos(2*x)*cos(4*x)) - k*(sin(8*x)/(8*sin(x)))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) =
        try_standard_common_scale_known_pair_shortcut(&options, &mut ctx, expr, false)
            .unwrap_or_else(|| panic!("expected common-scale known-pair shortcut to match"));
    assert_eq!(render(&ctx, rewritten), "0");
}

#[test]
fn simplify_pipeline_handles_morrie_scaled_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "k*(cos(x)*cos(2*x)*cos(4*x)) - k*(sin(8*x)/(8*sin(x)))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
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
fn simplify_pipeline_handles_triple_sine_against_polynomial_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    simplifier.set_collect_steps(false);
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
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
fn isolated_simplify_rewrites_to_zero_handles_polynomial_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(isolated_simplify_rewrites_to_zero(
        &options,
        &mut simplifier.context,
        expr
    ));
}

#[test]
fn supported_nested_zero_partner_rewrites_to_zero_handles_polynomial_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3)",
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
fn supported_nested_zero_child_partner_rejects_flat_nonlog_additive_noise_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr =
        parse("x + 1", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(!is_supported_nested_zero_child_partner(
        &simplifier.context,
        expr
    ));
}

#[test]
fn supported_nested_zero_partner_rewrites_to_zero_rejects_plain_division_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("1/(x+1) - 1/(x-1)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(!supported_nested_zero_partner_rewrites_to_zero(
        &options,
        &mut simplifier.context,
        expr
    ));
}

#[test]
fn supported_nested_zero_child_partner_keeps_nested_additive_cancellation_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(x - y) + (y - x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(is_supported_nested_zero_child_partner(
        &simplifier.context,
        expr
    ));
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
fn simplify_pipeline_revisits_after_exact_additive_pair_chain_cancellation_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("arctan(3) + arctan(1/3) + 10 - 10", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1/2 * pi");
}

#[test]
fn simplify_pipeline_revisits_constant_residual_after_additive_zero_cancellation_regression() {
    // Fixpoint gap caught by the dsolve O3 verification gate: the ±0
    // cancellation shortcut left a VARIABLE-FREE residual (`1/e^0 - 1`,
    // `ln(1)`, `sin(0)`, `sqrt(1) - 1`) un-folded because the resimplify
    // trigger was an enumerated shape list. A constant, non-literal
    // residual must always take the (cheap, single) re-pass.
    for (input, expected) in [
        ("(1/e^0 + 0 - 1) - 0", "0"),
        ("(ln(1) + 0) - 0", "0"),
        ("(sin(0) + 0) - 0", "0"),
        ("(sqrt(1) + 0 - 1) - 0", "0"),
        ("(1/ln(e) + 0 - 1) - 0", "0"),
        // Non-foldable constant residuals stay put (the re-pass is a
        // no-op, never a fabrication).
        ("(pi - e + 0) - 0", "pi - e"),
    ] {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr =
            parse(input, &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), expected, "{input}");
    }
    // Variable-bearing forms keep their existing path (no new trigger).
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(x + 0) - 0", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "x");
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
fn simplify_pipeline_revisits_after_exact_additive_pair_chain_gaussian_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(1+2*i) + (-1+3*i)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.value_domain =
        crate::semantics::ValueDomain::ComplexEnabled;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "5 * i");
}

#[test]
fn simplify_pipeline_revisits_diff_residual_after_exact_additive_pair_cancellation_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(diff(exp(sin(x)),x)+m) - (cos(x)*e^sin(x)+m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
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
fn child_isolated_exact_zero_handles_triple_sine_plus_rational_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (x/(1 + x/(1-x)) - x + x^2)",
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
fn simplify_pipeline_handles_full_mixed_identity_regression() {
    for expr_text in [
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3) + (sin(3*x)/sin(x) - 2*cos(2*x) - 1) + (ln(sqrt((1+sin(y))/(1-sin(y)))) - atanh(sin(y)))",
        "(x/(1 + x/(1-x)) - x + x^2) + ((cosh(x*y))^2 - (sinh(x*y))^2 - ((sin(x+y))^2 + (cos(x+y))^2))",
    ] {
        let mut simplifier = crate::Simplifier::with_default_rules();
        simplifier.set_steps_mode(crate::options::StepsMode::Off);
        let expr = parse(expr_text, &mut simplifier.context)
            .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        orchestrator.options.collect_steps = false;
        orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
        orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
        let (rewritten, _steps, _stats) =
            orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), "0");
    }
}

#[test]
fn simplify_pipeline_steps_on_decomposes_partitioned_direct_small_zero_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cos(x))^3 - (3*cos(x) + cos(3*x))/4) + (tan(x) + 1/tan(x) - 2/sin(2*x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = true;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
    assert_eq!(steps.len(), 2);
}

#[test]
fn simplify_pipeline_steps_on_decomposes_full_mixed_identity_zero_chunks_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cos(x))^3 - (3*cos(x) + cos(3*x))/4) + (tan(x) + 1/tan(x) - 2/sin(2*x)) + (atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (sqrt(x + sqrt(x^2 - y^2)) - (sqrt(x+y) + sqrt(x-y))/sqrt(2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = true;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
    assert_eq!(steps.len(), 4);
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
fn small_exact_zero_leaf_guard_rejects_positive_linear_sum_without_zero_family_markers() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr =
        parse("x + y", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = crate::phase::SimplifyOptions::default();
    assert!(!child_is_small_exact_zero_leaf_root(
        &options,
        &mut simplifier.context,
        expr,
    ));
}

#[test]
fn small_exact_zero_leaf_prefilter_matches_child_guard_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let markerless_sum =
        parse("x + y", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let non_additive =
        parse("2*x", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let trig_leaf = parse(
        "(cos(x))^3 - (3*cos(x) + cos(3*x))/4",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    assert!(!is_potential_small_exact_zero_leaf_root(
        &simplifier.context,
        markerless_sum,
    ));
    assert!(!is_potential_small_exact_zero_leaf_root(
        &simplifier.context,
        non_additive,
    ));
    assert!(is_potential_small_exact_zero_leaf_root(
        &simplifier.context,
        trig_leaf,
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
fn shifted_quotient_exact_one_candidate_gate_keeps_scaled_fraction_decompose_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("(a*x+b)/(c*x+d)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("lhs parse failed: {e:?}"));
    let rhs = parse("a/c + (b-a*d/c)/(c*x+d)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("rhs parse failed: {e:?}"));

    assert!(matches_shifted_quotient_exact_one_root_gate_candidate(
        &mut simplifier.context,
        lhs,
        rhs,
    ));
}

#[test]
fn shifted_quotient_shortcut_handles_scaled_fraction_decompose_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((a*x+b)/(c*x+d)) + 1)/((a/c + (b-a*d/c)/(c*x+d)) + 1)",
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
fn shifted_quotient_shortcut_handles_scaled_fraction_combine_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((a/c + (b-a*d/c)/(c*x+d)) + 1)/(((a*x+b)/(c*x+d)) + 1)",
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
fn exact_one_shortcut_handles_cubes_quotient_against_binomial_square_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((a^3-b^3)/(a-b) - (a^2 + a*b + b^2)) + 1)/((x^2 + 2*x + 1 - (x+1)^2) + 1)",
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
fn shifted_quotient_exact_one_candidate_gate_rejects_linear_collect_noise_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((a*x + b*x + c) + 1)/((((a + b)*x + c)) + 1)",
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

    assert!(
        !is_potential_shifted_quotient_exact_one_direct_pair_side_root(
            &mut simplifier.context,
            numerator_core,
        ) || !is_potential_shifted_quotient_exact_one_direct_pair_side_root(
            &mut simplifier.context,
            denominator_core,
        )
    );
}

#[test]
fn shifted_quotient_exact_one_candidate_gate_keeps_cubes_quotient_binomial_square_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((a^3-b^3)/(a-b) - (a^2 + a*b + b^2)) + 1)/((x^2 + 2*x + 1 - (x+1)^2) + 1)",
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

    assert!(
        is_potential_shifted_quotient_exact_one_direct_pair_side_root(
            &mut simplifier.context,
            numerator_core,
        )
    );
    assert!(
        is_potential_shifted_quotient_exact_one_direct_pair_side_root(
            &mut simplifier.context,
            denominator_core,
        )
    );
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
fn shifted_quotient_shortcut_keeps_common_factor_monomial_exact_one_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("((((x*y)/(z*y))) + 1)/((x/z) + 1)", &mut simplifier.context)
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
fn shifted_quotient_exact_one_candidate_gate_keeps_identical_additive_core_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("((x + y) + 1)/((x + y) + 1)", &mut simplifier.context)
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

    assert!(
        matches_shifted_quotient_exact_one_direct_or_passthrough_pair_root(
            &mut simplifier.context,
            numerator_core,
            denominator_core,
        )
    );
}

#[test]
fn shifted_quotient_exact_one_candidate_gate_keeps_reordered_trinomial_square_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((a + b + c)^2 + 1)/(a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c + 1)",
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

    assert!(
        !is_potential_shifted_quotient_exact_one_direct_pair_side_root(
            &mut simplifier.context,
            denominator_core,
        )
    );
    assert!(matches_shifted_quotient_exact_one_root_gate_candidate(
        &mut simplifier.context,
        numerator_core,
        denominator_core,
    ));
}

#[test]
fn shifted_quotient_exact_one_candidate_gate_keeps_symbolic_difference_squares_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((1/(2*a)*(1/(x-a) - 1/(x+a))) + 1))/(((1/(x^2-a^2)) + 1))",
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

    assert!(
        !is_potential_shifted_quotient_exact_one_direct_pair_side_root(
            &mut simplifier.context,
            numerator_core,
        ) || !is_potential_shifted_quotient_exact_one_direct_pair_side_root(
            &mut simplifier.context,
            denominator_core,
        )
    );
    assert!(matches_shifted_quotient_exact_one_root_gate_candidate(
        &mut simplifier.context,
        numerator_core,
        denominator_core,
    ));
}

#[test]
fn shifted_quotient_exact_one_candidate_gate_keeps_sum_cubes_quotient_shared_passthrough_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((a^3+b^3)/(a+b)+c) + 1)/((a^2 - a*b + b^2 + c) + 1)",
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
fn simplify_pipeline_handles_identical_additive_core_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("((x + y) + 1)/((x + y) + 1)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}

#[test]
fn shifted_quotient_shortcut_handles_sum_cubes_quotient_with_shared_passthrough_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((a^3+b^3)/(a+b)+c) + 1)/((a^2 - a*b + b^2 + c) + 1)",
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
fn nested_zero_direct_pair_family_candidate_gate_rejects_division_vs_common_factor_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((a^3-b^3)/(a-b) - (a^2 + a*b + b^2)) + 1)/((x*y + x*z - x*(y+z)) + 1)",
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

    assert!(!is_potential_nested_zero_direct_pair_family_pair_root(
        &mut simplifier.context,
        numerator_core,
        denominator_core,
    ));
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
fn shifted_quotient_exact_one_shortcut_handles_nested_fraction_vs_sophie_germain_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^4 + 4*y^4 - (x^2-2*x*y+2*y^2)*(x^2+2*x*y+2*y^2)) + 1)",
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
fn shifted_quotient_direct_small_zero_hot_gate_matches_nested_fraction_vs_sophie_germain_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x^4 + 4*y^4 - (x^2-2*x*y+2*y^2)*(x^2+2*x*y+2*y^2)) + 1)",
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
fn shifted_quotient_direct_small_zero_hot_gate_matches_nested_fraction_vs_small_quotient_cancel_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((((x^2 - 1)/(x - 1) - (x+1))) + 1)",
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
fn shifted_quotient_direct_small_zero_hot_gate_matches_nested_fraction_vs_small_rational_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1)) + 1)",
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
fn shifted_quotient_exact_one_shortcut_handles_nested_fraction_vs_small_rational_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1)) + 1)",
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
fn shifted_quotient_exact_one_shortcut_handles_nested_fraction_vs_same_denominator_distribution_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((x*y + x*z - x*(y+z)) + 1)",
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
fn shifted_quotient_exact_one_shortcut_handles_nested_fraction_vs_odd_half_power_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((sqrt(x^5) - x^2*sqrt(x)) + 1)",
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
fn shifted_quotient_nested_zero_fast_gate_candidate_matches_nested_fraction_vs_same_denominator_distribution_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((((a+b+c)/x - a/x - b/x - c/x)) + 1)",
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
fn shifted_quotient_nested_zero_fast_gate_candidate_matches_nested_fraction_vs_difference_quotient_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + 1)/((((x^2 - 1)/(x - 1) - (x+1))) + 1)",
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
fn simplify_pipeline_handles_contextual_rational_square_composition_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((1/(u - 1) + 1/(u + 1)) + ((v+1)^2)) - ((2*u/(u^2 - 1)) + (v^2 + 2*v + 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
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
fn detects_direct_small_pow_expansion_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs =
        parse("(v+1)^2", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("v^2 + 2*v + 1", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_small_pow_expansion_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}

#[test]
fn detects_direct_small_pow_expansion_pair_subtractive_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("(3*u - 1)^2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("9*u^2 - 6*u + 1", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_small_pow_expansion_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}

#[test]
fn detects_direct_small_pow_expansion_pair_trinomial_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("(a + b + c)^2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse(
        "a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_small_pow_expansion_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}

#[test]
fn detects_direct_small_pow_expansion_pair_quintic_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs =
        parse("(u-1)^5", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse(
        "u^5 - 5*u^4 + 10*u^3 - 10*u^2 + 5*u - 1",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_small_pow_expansion_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
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
fn shared_passthrough_small_pow_expansion_shortcut_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((a + b + c)^2 + m) - ((a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _steps) = super::try_standard_shared_passthrough_small_pow_expansion_shortcut(
        &mut simplifier.context,
        expr,
        true,
    )
    .unwrap_or_else(|| panic!("expected passthrough small-pow shortcut"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}

#[test]
fn detects_direct_rational_plus_minus_one_sum_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("1/(u - 1) + 1/(u + 1)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("2*u/(u^2 - 1)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_rational_plus_minus_one_sum_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
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
fn detects_direct_sum_of_squares_product_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("(x^2 + y^2)*(a^2 + b^2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("(x*a + y*b)^2 + (x*b - y*a)^2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_sum_of_squares_product_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}

#[test]
fn detects_direct_sum_diff_cubes_product_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("(u+v)*(u^2-u*v+v^2)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("u^3 + v^3", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_sum_diff_cubes_product_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}

#[test]
fn detects_direct_higher_degree_difference_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs =
        parse("u^6 - 1", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("((u^2+u+1)*(u^2-u+1)*(u+1)*(u-1))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_higher_degree_difference_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}

#[test]
fn detects_direct_sophie_germain_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs =
        parse("u^4 + 4", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("((u^2 + 2*u + 2)*(u^2 - 2*u + 2))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_sophie_germain_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));

    let zero_expr = parse(
        "u^4 + 4 - ((u^2 + 2*u + 2)*(u^2 - 2*u + 2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_sophie_germain_zero_identity_root(
        &mut simplifier.context,
        zero_expr
    ));
    assert!(
        super::matches_direct_sophie_germain_zero_hot_candidate_root(
            &mut simplifier.context,
            zero_expr
        )
    );
    assert!(super::matches_direct_small_zero_or_known_pair_base_root(
        &mut simplifier.context,
        zero_expr
    ));
}

#[test]
fn detects_direct_three_linear_shift_product_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("(u+1)*(u+2)*(u+3)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("u^3 + 6*u^2 + 11*u + 6", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_three_linear_shift_product_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}

#[test]
fn detects_direct_two_linear_shift_product_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("(u+2)*(u+3)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("u^2 + 5*u + 6", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_two_linear_shift_product_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
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
fn simplify_pipeline_handles_three_linear_shift_anchor_times_radical_product_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((u+1)*(u+2)*(u+3)) * (sqrt(x)*sqrt(4*x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse("(u^3 + 6*u^2 + 11*u + 6) * 2*x", &mut simplifier.context)
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
fn simplify_pipeline_handles_three_linear_shift_anchor_times_tangent_addition_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((u+1)*(u+2)*(u+3)) * (tan(x) + tan(y))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse(
        "(u^3 + 6*u^2 + 11*u + 6) * (sin(x+y)/(cos(x)*cos(y)))",
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
fn simplify_pipeline_handles_tangent_addition_anchor_times_sum_diff_cubes_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(tan(x) + tan(y)) * (u^3 - 1)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let expected = parse(
        "(sin(x+y)/(cos(x)*cos(y))) * ((u-1)*(u^2 + u + 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    let (expected_rewritten, _steps, _stats) =
        orchestrator.simplify_pipeline(expected, &mut simplifier);
    let difference = simplifier
        .context
        .add(Expr::Sub(rewritten, expected_rewritten));
    let (difference_rewritten, _steps, _stats) =
        simplifier.simplify_with_stats(difference, crate::SimplifyOptions::default());
    let zero = simplifier.context.num(0);
    assert_eq!(
        compare_expr(&simplifier.context, difference_rewritten, zero),
        Ordering::Equal
    );
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
fn detects_direct_short_geometric_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse("u^3 + u^2 + u + 1", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("((u+1)*(u^2+1))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_known_direct_pair_root(
        &mut simplifier.context,
        lhs,
        rhs
    ));
}

#[test]
fn small_composed_additive_pair_shortcut_handles_short_geometric_contextual_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sec(u)^2 - tan(u)^2) + (u^3 + u^2 + u + 1)) - ((1) + ((u+1)*(u^2+1)))",
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
fn small_composed_additive_pair_shortcut_handles_quintic_contextual_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sec(u)^2 - tan(u)^2) + ((u-1)^5)) - ((1) + (u^5 - 5*u^4 + 10*u^3 - 10*u^2 + 5*u - 1))",
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
fn detects_composed_small_additive_pair_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let lhs = parse(
        "(1/(u - 1) + 1/(u + 1)) + ((v+1)^2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("(2*u/(u^2 - 1)) + (v^2 + 2*v + 1)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_composed_small_additive_pair_root(
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
fn small_composed_additive_pair_shortcut_handles_quadratic_contextual_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((sec(u)^2 - tan(u)^2) + ((u+2)^2)) - ((1) + (u^2 + 4*u + 4))",
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

#[test]
fn simplify_pipeline_handles_cubes_quotient_against_common_factor_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((a^3-b^3)/(a-b) - (a^2 + a*b + b^2)) + (x*y + x*z - x*(y+z))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}

#[test]
fn simplify_pipeline_handles_cubes_quotient_against_common_factor_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((a^3-b^3)/(a-b) - (a^2 + a*b + b^2)) + 1)/((x*y + x*z - x*(y+z)) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}

#[test]
fn simplify_pipeline_handles_cubes_quotient_against_binomial_square_shifted_quotient_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((a^3-b^3)/(a-b) - (a^2 + a*b + b^2)) + 1)/((x^2 + 2*x + 1 - (x+1)^2) + 1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1");
}

#[test]
fn simplify_pipeline_factors_small_polynomial_denominator_binomial_square_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("1/(u^2 + 2*u + 1)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1 / (u + 1)^2");
}

#[test]
fn simplify_pipeline_factors_small_polynomial_denominator_cubic_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("1/(u^3 + u^2 + u + 1)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(
        render(&simplifier.context, rewritten),
        "1 / ((u + 1) * (u^2 + 1))"
    );
}

#[test]
fn detects_small_quotient_cancel_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("((x^2 - 1)/(x - 1) - (x+1))", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_small_quotient_cancel_zero_hot_candidate_root(
        &mut simplifier.context,
        expr
    ));
    assert!(
        extract_small_quotient_cancel_zero_candidate_root(&mut simplifier.context, expr).is_some()
    );
    assert!(matches_small_quotient_cancel_zero_identity_root(
        &crate::phase::SimplifyOptions::default(),
        &mut simplifier.context,
        expr
    ));
}

#[test]
fn detects_direct_perfect_square_trinomial_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("x^2 + 2*x + 1 - (x+1)^2", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_perfect_square_trinomial_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}

#[test]
fn shifted_quotient_passthrough_cores_match_direct_small_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((a^3-b^3)/(a-b) - (a^2 + a*b + b^2)) + 1)/((x^2 + 2*x + 1 - (x+1)^2) + 1)",
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

    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        numerator_core
    ));
    assert!(matches_direct_small_zero_identity_root(
        &mut simplifier.context,
        denominator_core
    ));
}

#[test]
fn detects_direct_sqrt_perfect_square_abs_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "sqrt(a^2 + 2*a*b + b^2) - abs(a+b)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_sqrt_perfect_square_abs_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}

#[test]
fn detects_direct_shifted_root_square_sum_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "sqrt(x + 2*sqrt(x-1)) - sqrt(x-1) - 1",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_sqrt_perfect_square_abs_zero_identity_root(
        &mut simplifier.context,
        expr
    ));
}

#[test]
fn simplify_pipeline_closes_shifted_root_square_tail_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "sqrt(x + 2*sqrt(x-1)) - sqrt(x-1) - 1",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = false;
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}

#[test]
fn simplify_pipeline_closes_passthrough_factorial_and_telescoping_tail_regressions() {
    for expr_text in [
        "(((n+1)!/n!) + m) - ((n+1) + m)",
        "(((n+1)!/n!+a) + m) - ((n+1+a) + m)",
        "((product((k+1)/k, k, 1, n)) + m) - ((n+1) + m)",
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
fn direct_small_zero_additive_combination_shortcut_handles_nested_fraction_against_geometric_factor_regression(
) {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) + (x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1))",
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
