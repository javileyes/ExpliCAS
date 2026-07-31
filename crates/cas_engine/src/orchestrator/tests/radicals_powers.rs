//! Tests del orquestador: familia `radicals_powers` (troceo P1).

use super::*;

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
fn matrix_multiply_square_product_unchanged() {
    assert_eq!(
        simplify_render("[[1,2],[3,4]] * [[5,6],[7,8]]"),
        "[[19, 22], [43, 50]]"
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
fn detects_direct_reciprocal_sqrt_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("1/sqrt(x)", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("sqrt(x)/x", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_reciprocal_sqrt_pair_root(&mut ctx, lhs, rhs));
}
#[test]
fn detects_direct_abs_square_pair_regression() {
    let mut ctx = Context::new();
    let lhs = parse("abs(cos(x))^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let rhs = parse("cos(x)^2", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_abs_square_pair_root(&ctx, lhs, rhs));
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
