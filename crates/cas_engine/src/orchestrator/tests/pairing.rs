//! Tests del orquestador: familia `pairing` (troceo P1).

use super::*;

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
fn detects_direct_small_exact_constant_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("sec(pi)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("-1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(super::matches_direct_small_exact_constant_pair_root(
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
