//! Tests del orquestador: familia `general` (troceo P1).

use super::*;

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
fn simplify_pipeline_handles_negative_double_sine_direct_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("sin(2*x) - 2*sin(x)*cos(x)", &mut simplifier.context)
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
