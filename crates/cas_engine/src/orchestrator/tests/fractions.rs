//! Tests del orquestador: familia `fractions` (troceo P1).

use super::*;

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
fn detects_direct_successive_unit_fraction_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("1/x + 1/(x+1)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs =
        parse("(2*x+1)/(x*(x+1))", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_addition_of_successive_unit_fractions_pair_root(&mut ctx, lhs, rhs));
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
