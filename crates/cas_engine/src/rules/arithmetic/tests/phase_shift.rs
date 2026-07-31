//! Tests de las reglas aritméticas: familia `phase_shift` (troceo P1).

use super::*;

#[test]
fn maybe_shifted_quotient_exact_zero_direct_residual_candidate_rejects_plain_multiplicative_pair() {
    let mut ctx = Context::new();
    let expr = parse("2*x - 3*x", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::maybe_shifted_quotient_exact_zero_direct_residual_candidate(&ctx, expr));
}
#[test]
fn maybe_shifted_quotient_exact_zero_direct_residual_candidate_accepts_hyperbolic_quotient_pair() {
    let mut ctx = Context::new();
    let expr = parse("cosh(x) - ((e^x + e^(-x))/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::super::maybe_shifted_quotient_exact_zero_direct_residual_candidate(&ctx, expr));
}
#[test]
fn maybe_shifted_quotient_exact_zero_direct_residual_route_candidate_accepts_hyperbolic_pair() {
    let mut ctx = Context::new();
    let lhs_core = parse("cosh(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core =
        parse("((e^x + e^(-x))/2)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let residual_expr = parse("cosh(x) - ((e^x + e^(-x))/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        super::super::maybe_shifted_quotient_exact_zero_direct_residual_route_candidate(
            &mut ctx,
            lhs_core,
            rhs_core,
            residual_expr,
        )
    );
}
#[test]
fn maybe_shifted_quotient_exact_zero_direct_residual_route_candidate_accepts_trig_power_reduction_pair(
) {
    let mut ctx = Context::new();
    let lhs_core = parse("sin(x)^4", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core = parse("(3 - 4*cos(2*x) + cos(4*x))/8", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));
    let residual_expr = parse("sin(x)^4 - (3 - 4*cos(2*x) + cos(4*x))/8", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        super::super::maybe_shifted_quotient_exact_zero_direct_residual_route_candidate(
            &mut ctx,
            lhs_core,
            rhs_core,
            residual_expr,
        )
    );
}
#[test]
fn maybe_shifted_quotient_exact_zero_direct_residual_route_candidate_rejects_abs_sqrt_pair() {
    let mut ctx = Context::new();
    let lhs_core =
        parse("sqrt(x + 2*sqrt(x-1))", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core = parse("sqrt(x-1) + 1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let residual_expr = parse("sqrt(x + 2*sqrt(x-1)) - sqrt(x-1) - 1", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        !super::super::maybe_shifted_quotient_exact_zero_direct_residual_route_candidate(
            &mut ctx,
            lhs_core,
            rhs_core,
            residual_expr,
        )
    );
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_sec_squared_identity() {
    let mut ctx = Context::new();
    let expr = parse("((sec(x)^2) + 1)/((1 + tan(x)^2) + 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_half_angle_square_identity() {
    let mut ctx = Context::new();
    let expr = parse("((sin(x)^2) + 1)/(((1-cos(2*x))/2) + 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_half_angle_against_binomial_square_identity() {
    let mut ctx = Context::new();
    let expr = parse(
        "((sin(x)^2 - (1-cos(2*x))/2) + 1)/(((sin(x)+cos(x))^2 - (1+sin(2*x))) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Trig Square Identity");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_binomial_square_against_sec_tan_identity() {
    let mut ctx = Context::new();
    let expr = parse(
        "(((sin(x)+cos(x))^2 - (1+sin(2*x))) + 1)/((sec(x)^2 - tan(x)^2 - 1) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
}
#[test]
fn collapse_exact_zero_common_scaled_difference_rule_matches_phase_shift_pair_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "k*(sin(x)+cos(x)+sin(y)+cos(y)) - k*(sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroCommonScaledDifferenceRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
    assert!(!rewrite.substeps.is_empty());
}
#[test]
fn direct_core_equivalence_rewrite_matches_exact_quarter_phase_shift_pair_sum() {
    let mut ctx = Context::new();
    let lhs = parse("sin(x)+cos(x)+sin(y)+cos(y)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse lhs: {err}"));
    let rhs = parse("sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_trig_exact_quarter_phase_shift_pair_equivalence_rewrite(
            &mut ctx, lhs, rhs,
        )
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.final_expr()
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
    assert_eq!(rewrite.substeps.len(), 2);
}
#[test]
fn collapse_exact_zero_common_scaled_difference_rule_matches_general_phase_shift_sum_with_passthrough(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "k*(3*sin(x)+4*cos(x)+a) - k*(5*sin(x+arctan(4/3))+a)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroCommonScaledDifferenceRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
}
#[test]
fn collapse_exact_zero_common_scaled_difference_rule_matches_shifted_to_shifted_phase_shift_pair() {
    let mut ctx = Context::new();
    let expr = parse(
        "k*(5*sin(arctan(4/3) + x)) - k*(5*cos(x - arctan(3/4)))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroCommonScaledDifferenceRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
}
#[test]
fn collapse_exact_zero_additive_subexpression_rule_matches_phase_shift_pair_sum_with_passthrough() {
    let mut ctx = Context::new();
    let expr = parse(
        "((sin(x)+cos(x)+sin(y)+cos(y)) + m) - ((sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)) + m)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.final_expr()
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
    assert!(rewrite.chained.is_empty());
    assert!(!rewrite.substeps.is_empty());
}
#[test]
fn collapse_exact_zero_product_factor_rule_matches_factorized_shifted_phase_pair() {
    let mut ctx = Context::new();
    let expr = parse("sqrt(2)*(sin(x+pi/4)-cos(x-pi/4))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroProductFactorRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
    assert!(rewrite.substeps.is_empty());
}
#[test]
fn collapse_exact_zero_common_scaled_difference_rule_matches_exact_shifted_sine_cosine_scaled_difference(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "k*(sqrt(2)*sin(x+pi/4)) - k*(sqrt(2)*cos(x-pi/4))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroCommonScaledDifferenceRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_shifted_hyperbolic_pythagorean() {
    let mut ctx = Context::new();
    let expr =
        parse("cosh(x)^2 - 1 - sinh(x)^2", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Hyperbolic Pythagorean Identity");
}
#[test]
fn collapse_exact_zero_common_scaled_difference_rule_matches_shifted_hyperbolic_pythagorean() {
    let mut ctx = Context::new();
    let expr = parse("k*(cosh(x)^2 - 1) - k*(sinh(x)^2)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroCommonScaledDifferenceRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Hyperbolic Pythagorean Identity");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_shifted_hyperbolic_double_angle() {
    let mut ctx = Context::new();
    let expr = parse("((cosh(2*x)) + 1)/((2*cosh(x)^2 - 1) + 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Hyperbolic Double-Angle Identity");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_exp_hyperbolic_sum_pair() {
    let mut ctx = Context::new();
    let expr = parse("(e^x + 1)/(sinh(x) + cosh(x) + 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Hyperbolic Sum to Exponential");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_cosh_exp_definition_pair() {
    let mut ctx = Context::new();
    let expr = parse("(cosh(x) + 1)/(((e^x + e^(-x))/2) + 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(
        &rewrite.description,
        "Recognize Hyperbolic from Exponential",
    );
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_sinh_exp_definition_pair() {
    let mut ctx = Context::new();
    let expr = parse("(sinh(x) + 1)/(((e^x - e^(-x))/2) + 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(
        &rewrite.description,
        "Recognize Hyperbolic from Exponential",
    );
}
#[test]
fn direct_core_equivalence_rewrite_matches_shifted_hyperbolic_pythagorean_pair() {
    let mut ctx = Context::new();
    let lhs = parse("cosh(x)^2 - 1", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sinh(x)^2", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Hyperbolic Pythagorean Identity");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_core_equivalence_rewrite_keeps_symbolic_scale_sum_pair_before_phase_shift_reject_regression(
) {
    let mut ctx = Context::new();
    let lhs = parse("a*x^2 + b*x + c", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("x*(c/x + a*x + b)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_some());
}
#[test]
fn direct_core_equivalence_rewrite_rejects_shifted_surface_trig_symbolic_base_mismatch_before_default_simplify(
) {
    let mut ctx = Context::new();
    let lhs = parse("sin(x + pi/4)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("-sin(y + pi/4)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}
#[test]
fn shifted_quotient_power_merge_residual_rewrite_matches_symbolic_power_pair() {
    let mut ctx = Context::new();
    let lhs = parse("x^a * x^b", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("x^(a+b)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_shifted_quotient_power_merge_residual_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn shifted_quotient_power_merge_residual_rewrite_matches_root_power_pair() {
    let mut ctx = Context::new();
    let lhs = parse("sqrt(x) * x^a", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("x^(a+1/2)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_shifted_quotient_power_merge_residual_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn shifted_quotient_power_merge_residual_rewrite_matches_division_power_pair() {
    let mut ctx = Context::new();
    let lhs = parse("x^a / x^b", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("x^(a-b)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_shifted_quotient_power_merge_residual_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn shifted_quotient_power_merge_residual_rewrite_matches_reciprocal_power_pair() {
    let mut ctx = Context::new();
    let lhs = parse("1 / x^a", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("x^(-a)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_shifted_quotient_power_merge_residual_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn shifted_quotient_power_merge_residual_rewrite_matches_constant_division_power_pair() {
    let mut ctx = Context::new();
    let lhs = parse("2^a / 2^b", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("2^(a-b)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_shifted_quotient_power_merge_residual_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn shifted_quotient_power_merge_residual_rewrite_matches_constant_reciprocal_power_pair() {
    let mut ctx = Context::new();
    let lhs = parse("1 / 2^a", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("2^(-a)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_shifted_quotient_power_merge_residual_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn shifted_quotient_cancel_common_factors_residual_rewrite_matches_simple_monomial_pair() {
    let mut ctx = Context::new();
    let lhs = parse("(x*y)/y", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("x", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_shifted_quotient_cancel_common_factors_residual_rewrite(
        &mut ctx, lhs, rhs,
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
    assert_eq!(rewrite.required_conditions.len(), 1);
}
#[test]
fn shifted_quotient_cancel_common_factors_residual_rewrite_matches_fraction_pair() {
    let mut ctx = Context::new();
    let lhs = parse("(x*y)/(z*y)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("x/z", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_shifted_quotient_cancel_common_factors_residual_rewrite(
        &mut ctx, lhs, rhs,
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
    assert_eq!(rewrite.required_conditions.len(), 1);
}
#[test]
fn shifted_quotient_cancel_common_factors_residual_rewrite_rejects_additive_source() {
    let mut ctx = Context::new();
    let lhs = parse("(x+y)/y", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("x/y + 1", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_shifted_quotient_cancel_common_factors_residual_rewrite(
        &mut ctx, lhs, rhs,
    );

    assert!(rewrite.is_none());
}
#[test]
fn shifted_quotient_fraction_combine_residual_rewrite_matches_fraction_pair() {
    let mut ctx = Context::new();
    let lhs = parse("a/x + b/y", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("(a*y + b*x)/(x*y)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_shifted_quotient_fraction_combine_residual_rewrite(
        &mut ctx, lhs, rhs,
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Combine Fractions");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn shifted_quotient_fraction_combine_residual_rewrite_matches_scaled_fraction_pair() {
    let mut ctx = Context::new();
    let lhs = parse("a/c + b/(c*x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("(a*x + b)/(c*x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_shifted_quotient_fraction_combine_residual_rewrite(
        &mut ctx, lhs, rhs,
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Combine Fractions");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn shifted_quotient_nested_fraction_residual_rewrite_matches_simple_pair() {
    let mut ctx = Context::new();
    let lhs = parse("1/(1/a + 1/b)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("a*b/(a+b)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_shifted_quotient_nested_fraction_residual_rewrite(
        &mut ctx, lhs, rhs,
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Simplify Nested Fraction");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_negative_even_root_power_reciprocal_rewrite_matches_shifted_trig_ratio_cofactor() {
    let mut ctx = Context::new();
    let lhs = parse("tan(x)*cos(x)^(-1/2)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sin(x)/sqrt(cos(x)^3)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_negative_even_root_power_reciprocal_rewrite(
        &mut ctx, lhs, rhs,
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Negative Even-Root Power Reciprocal Cancellation",
    );
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_hyperbolic_sinh_sum_to_product() {
    let mut ctx = Context::new();
    let expr = parse(
        "((sinh(x)+sinh(y)) + 1)/((2*sinh((x+y)/2)*cosh((x-y)/2)) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Hyperbolic Product-to-Sum Identity");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_trig_and_tan_cot_sec_csc_core() {
    let mut ctx = Context::new();
    let expr = parse(
        "((2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + 1)/((tan(x) + cot(x) - sec(x)*csc(x)) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_eq!(rewrite.required_conditions.len(), 3);
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_trig_and_tan_cot_product_core() {
    let mut ctx = Context::new();
    let expr = parse(
        "((2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + 1)/((tan(x)*cot(x) - 1) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_eq!(rewrite.required_conditions.len(), 3);
}
#[test]
fn direct_small_zero_core_shifted_quotient_rewrite_matches_trig_and_tan_cot_product_core() {
    let mut ctx = Context::new();
    let expr = parse(
        "((2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + 1)/((tan(x)*cot(x) - 1) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
        panic!("expected quotient");
    };
    let numerator_core = super::super::strip_positive_one_passthrough(&mut ctx, numerator)
        .unwrap_or_else(|| panic!("numerator core"));
    let denominator_core = super::super::strip_positive_one_passthrough(&mut ctx, denominator)
        .unwrap_or_else(|| panic!("denominator core"));

    let rewrite = super::super::try_build_direct_small_zero_core_shifted_quotient_rewrite(
        &mut ctx,
        numerator,
        denominator,
        numerator_core,
        denominator_core,
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(rewrite.required_conditions.len(), 3);
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_trig_and_sec_tan_pythagorean_core() {
    let mut ctx = Context::new();
    let expr = parse(
        "((2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + 1)/((sec(x)^2 - tan(x)^2 - 1) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_eq!(rewrite.required_conditions.len(), 1);
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_trig_and_symmetric_partial_fraction_core() {
    let mut ctx = Context::new();
    let expr = parse(
        "((2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + 1)/((1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1)) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_eq!(rewrite.required_conditions.len(), 4);
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_trig_and_rationalized_sum_of_sqrts_core() {
    let mut ctx = Context::new();
    let expr = parse(
        "((2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + 1)/((1/(sqrt(a) + sqrt(b)) - (sqrt(a) - sqrt(b))/(a - b)) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_eq!(rewrite.required_conditions.len(), 5);
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_hyperbolic_cosh_difference_to_product() {
    let mut ctx = Context::new();
    let expr = parse(
        "((cosh(x)-cosh(y)) + 1)/((2*sinh((x+y)/2)*sinh((x-y)/2)) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Hyperbolic Product-to-Sum Identity");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_phase_shift_pair_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "((sin(x)+cos(x)+sin(y)+cos(y)) + 1)/((sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let expected = parse(
        "(sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)+1)/(sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)+1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse expected: {err}"));

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        expected
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_symbolic_power_merge_pair() {
    let mut ctx = Context::new();
    let expr = parse("((x^a*x^b) + 1)/(x^(a+b) + 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_division_power_merge_pair() {
    let mut ctx = Context::new();
    let expr = parse("((x^a/x^b) + 1)/(x^(a-b) + 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_reciprocal_power_merge_pair() {
    let mut ctx = Context::new();
    let expr =
        parse("((1/x^a) + 1)/(x^(-a) + 1)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_constant_division_power_merge_pair() {
    let mut ctx = Context::new();
    let expr = parse("((2^a/2^b) + 1)/(2^(a-b) + 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_constant_reciprocal_power_merge_pair() {
    let mut ctx = Context::new();
    let expr =
        parse("((1/2^a) + 1)/(2^(-a) + 1)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_simple_common_factor_pair() {
    let mut ctx = Context::new();
    let expr =
        parse("(((x*y)/y) + 1)/(x + 1)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
    assert_eq!(rewrite.required_conditions.len(), 2);
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_fraction_common_factor_pair() {
    let mut ctx = Context::new();
    let expr = parse("(((x*y)/(z*y)) + 1)/((x/z) + 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
    assert_eq!(rewrite.required_conditions.len(), 2);
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_fraction_combine_pair() {
    let mut ctx = Context::new();
    let expr = parse("((a/x + b/y) + 1)/(((a*y + b*x)/(x*y)) + 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Combine Fractions");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_nested_fraction_pair() {
    let mut ctx = Context::new();
    let expr = parse("((1/(1/a + 1/b)) + 1)/((a*b/(a+b)) + 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Simplify Nested Fraction");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_fraction_decompose_plus_tail_pair() {
    let mut ctx = Context::new();
    let expr = parse(
        "(((a*x + b*y + c)/(x*y)) + 1)/((a/y + b/x + c/(x*y)) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_fraction_decompose_split_pair() {
    let mut ctx = Context::new();
    let expr = parse(
        "(((a*x + b*y + c*z)/(x*y*z)) + 1)/((a/(y*z) + b/(x*z) + c/(x*y)) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_fraction_decompose_with_whole_term_pair() {
    let mut ctx = Context::new();
    let expr = parse(
        "(((d*x*y*z + a*x*y + b*x*z + c*y*z)/(x*y*z)) + 1)/((a/z + b/y + c/x + d) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
}
#[test]
fn direct_finite_product_equivalence_rewrite_matches_shifted_factorized_telescoping_product() {
    let mut ctx = Context::new();
    let lhs = parse("product(1 - 1/(k+a)^2, k, m, n)", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("((m+a-1)*(n+a+1))/((m+a)*(n+a))", &mut ctx)
        .unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_finite_product_equivalence_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Finite Telescoping Product");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_dirichlet_shifted_quotient() {
    let mut ctx = Context::new();
    let expr = parse(
        "((1 + 2*cos(x) + 2*cos(2*x)) + 1)/((sin(5*x/2)/sin(x/2)) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Dirichlet Kernel Identity");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_complete_square() {
    let mut ctx = Context::new();
    let expr = parse(
        "((a*x^2 - b*x + c) + 1)/((a*(x - b/(2*a))^2 + c - b^2/(4*a)) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert!(
        rewrite.description.is_empty() || rewrite.description == "Complete the Square",
        "expected direct exact-one collapse or the legacy complete-square label, got {:?}",
        rewrite.description
    );
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_trinomial_square() {
    let mut ctx = Context::new();
    let expr = parse(
        "((a + b + c)^2 + 1)/(a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Expand binomial/trinomial power");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_symbolic_scale_sum_distribution_core() {
    let mut ctx = Context::new();
    let expr = parse(
        "((a*x^2 + b*x + c) + 1)/((x*(c/x + a*x + b)) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_grouped_symbolic_scale_sum_distribution_core() {
    let mut ctx = Context::new();
    let expr = parse(
        "((a*x^2 + c*x^2 + e*x^2 + b*x + d*x) + 1)/((x*(b + d) + x^2*(a + c + e)) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
}
#[test]
fn collapse_exact_one_shifted_quotient_rule_matches_power_reciprocal_symbolic_scale_sum_distribution_core(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "((a*x^4 + b*x^3 + c*x^2 + d) + 1)/((x^2*(d/x^2 + a*x^2 + b*x + c)) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactOneShiftedQuotientRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let one = ctx.num(1);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
}
#[test]
fn complete_square_binomial_expr_flips_negative_shift_for_positive_orientation() {
    let mut ctx = Context::new();
    let x = parse("x", &mut ctx).unwrap_or_else(|err| panic!("parse x: {err}"));
    let neg_shift = parse("-b/(2*a)", &mut ctx).unwrap_or_else(|err| panic!("parse shift: {err}"));

    let expr = super::super::build_complete_square_binomial_expr(
        &mut ctx,
        x,
        neg_shift,
        cas_math::expr_nary::Sign::Pos,
    );

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: expr
            }
        ),
        "x - b / (2 * a)"
    );
}
#[test]
fn complete_square_binomial_expr_flips_negative_shift_for_negative_orientation() {
    let mut ctx = Context::new();
    let x = parse("x", &mut ctx).unwrap_or_else(|err| panic!("parse x: {err}"));
    let neg_shift = parse("-b/(2*a)", &mut ctx).unwrap_or_else(|err| panic!("parse shift: {err}"));

    let expr = super::super::build_complete_square_binomial_expr(
        &mut ctx,
        x,
        neg_shift,
        cas_math::expr_nary::Sign::Neg,
    );

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: expr
            }
        ),
        "b / (2 * a) + x"
    );
}
#[test]
fn expand_trig_phase_shift_to_enable_cancellation_rule_matches_general_shifted_sine() {
    let mut ctx = Context::new();
    let expr = parse("3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigPhaseShiftToEnableCancellationRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
    assert_eq!(rewrite.substeps.len(), 2);
}
#[test]
fn expand_trig_phase_shift_to_enable_cancellation_rule_matches_complementary_general_shifted_cosine(
) {
    let mut ctx = Context::new();
    let expr = parse("5*sin(arctan(4/3) + x) - 5*cos(x - arctan(3/4))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigPhaseShiftToEnableCancellationRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
    assert_eq!(rewrite.substeps.len(), 2);
}
#[test]
fn trig_phase_shift_cancellation_match_prefers_forward_shifted_to_shifted_for_complementary_shifted_pair(
) {
    let mut ctx = Context::new();
    let focus_expr = parse("5*sin(arctan(4/3) + x)", &mut ctx)
        .unwrap_or_else(|err| panic!("focus parse: {err}"));
    let target_expr = parse("5*cos(x - arctan(3/4))", &mut ctx)
        .unwrap_or_else(|err| panic!("target parse: {err}"));

    let rewrite_match = super::super::try_find_trig_phase_shift_cancellation_match(
        &mut ctx,
        focus_expr,
        target_expr,
        false,
    )
    .unwrap_or_else(|| panic!("rewrite match"));

    assert_eq!(
        rewrite_match.mode,
        super::super::TrigPhaseShiftCancellationMode::ShiftedToShifted
    );
}
#[test]
fn resolve_surface_shifted_candidate_vs_plain_trig_target_rejects_cross_fn_quarter_base_mismatch_symmetrically(
) {
    let mut ctx = Context::new();
    let candidate = parse("sqrt(2)*sin(x + 1/4*pi)", &mut ctx)
        .unwrap_or_else(|err| panic!("candidate parse: {err}"));
    let target = parse("cos(y)", &mut ctx).unwrap_or_else(|err| panic!("target parse: {err}"));

    assert_eq!(
        super::super::resolve_surface_shifted_candidate_vs_plain_trig_target_for_phase_shift(
            &mut ctx, candidate, target,
        ),
        Some(false)
    );
}
#[test]
fn shifted_general_target_match_detects_complementary_shifted_cosine_pair() {
    let mut ctx = Context::new();
    let focus_expr = parse("5*sin(arctan(4/3) + x)", &mut ctx)
        .unwrap_or_else(|err| panic!("focus parse: {err}"));
    let target_expr = parse("5*cos(x - arctan(3/4))", &mut ctx)
        .unwrap_or_else(|err| panic!("target parse: {err}"));

    let focus_data =
        super::super::extract_general_phase_shift_term_data_for_cancellation(&mut ctx, focus_expr)
            .unwrap_or_else(|| panic!("focus data"));
    let (arg, sin_coeff, cos_coeff, sin_sign, cos_sign) =
        super::super::extract_general_phase_shift_linear_signature_for_cancellation(
            &mut ctx, focus_data,
        )
        .unwrap_or_else(|| panic!("focus linear signature"));
    let target_data =
        super::super::extract_general_phase_shift_term_data_for_cancellation(&mut ctx, target_expr)
            .unwrap_or_else(|| panic!("target data"));

    assert!(
        super::super::matches_general_phase_shift_shifted_term_candidate_for_cancellation(
            &mut ctx,
            target_data,
            (arg, sin_coeff, cos_coeff, sin_sign, cos_sign),
            false,
        )
    );
}
#[test]
fn shifted_general_target_match_detects_general_shifted_sine_pair_regression() {
    let mut ctx = Context::new();
    let focus_expr =
        parse("3*sin(x) + 4*cos(x)", &mut ctx).unwrap_or_else(|err| panic!("focus parse: {err}"));
    let target_expr = parse("5*sin(x + arctan(4/3))", &mut ctx)
        .unwrap_or_else(|err| panic!("target parse: {err}"));

    let (arg, sin_coeff, cos_coeff, sin_sign, cos_sign) =
        super::super::extract_weighted_phase_shift_linear_combination_for_cancellation(
            &mut ctx, focus_expr,
        )
        .unwrap_or_else(|| panic!("focus linear signature"));
    let target_data =
        super::super::extract_general_phase_shift_term_data_for_cancellation(&mut ctx, target_expr)
            .unwrap_or_else(|| panic!("target data"));

    assert!(
        super::super::matches_general_phase_shift_shifted_term_candidate_for_cancellation(
            &mut ctx,
            target_data,
            (arg, sin_coeff, cos_coeff, sin_sign, cos_sign),
            false,
        )
    );
}
#[test]
fn expand_trig_phase_shift_to_enable_cancellation_rule_matches_exact_sixth_shifted_sine() {
    let mut ctx = Context::new();
    let expr = parse("cos(x) + sqrt(3)*sin(x) - 2*sin(x + pi/6)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigPhaseShiftToEnableCancellationRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
    assert_eq!(rewrite.substeps.len(), 2);
}
#[test]
fn expand_trig_phase_shift_to_enable_cancellation_rule_matches_exact_quarter_shifted_sine_mul_pi_fraction(
) {
    let mut ctx = Context::new();
    let expr = parse("sin(x) + cos(x) - sqrt(2)*sin(x + 1/4*pi)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigPhaseShiftToEnableCancellationRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
    assert_eq!(rewrite.substeps.len(), 2);
}
#[test]
fn expand_trig_phase_shift_to_enable_cancellation_rule_matches_exact_quarter_shifted_sine_pow_half_coeff(
) {
    let mut ctx = Context::new();
    let expr = parse("sin(x) + cos(x) - 2^(1/2)*sin(x + 1/4*pi)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigPhaseShiftToEnableCancellationRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
    assert_eq!(rewrite.substeps.len(), 2);
}
#[test]
fn expand_trig_phase_shift_to_enable_cancellation_rule_matches_exact_quarter_shifted_sine_unit_coeff(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "1/2*sqrt(2)*sin(x) + 1/2*sqrt(2)*cos(x) - sin(x + 1/4*pi)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigPhaseShiftToEnableCancellationRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
    assert_eq!(rewrite.substeps.len(), 2);
}
#[test]
fn expand_trig_phase_shift_to_enable_cancellation_rule_matches_exact_quarter_shifted_sine_symbolic_coeff(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "1/2*sqrt(2)*k*sin(x) + 1/2*sqrt(2)*k*cos(x) - k*sin(x + 1/4*pi)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigPhaseShiftToEnableCancellationRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
    assert_eq!(rewrite.substeps.len(), 2);
}
#[test]
fn exact_trig_phase_shift_zero_scope_fast_structural_triple_matches_quarter_regression() {
    let mut ctx = Context::new();
    let expr = parse("sin(x) + cos(x) - sqrt(2)*sin(x + 1/4*pi)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::super::try_build_exact_trig_phase_shift_zero_scope_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
}
#[test]
fn exact_trig_phase_shift_zero_scope_fast_structural_triple_matches_sixth_regression() {
    let mut ctx = Context::new();
    let expr = parse("sqrt(3)*sin(x) + cos(x) - 2*sin(x + 1/6*pi)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::super::try_build_exact_trig_phase_shift_zero_scope_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
}
#[test]
fn exact_trig_phase_shift_zero_scope_fast_general_triple_matches_arctan_regression() {
    let mut ctx = Context::new();
    let expr = parse("3*sin(x) + 4*cos(x) - 5*sin(x + arctan(4/3))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::super::try_build_exact_trig_phase_shift_zero_scope_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_eq!(rewrite.substeps.len(), 2);
}
#[test]
fn trig_phase_shift_cancellation_match_rejects_variable_shifted_term_against_plain_algebraic_target(
) {
    let mut ctx = Context::new();
    let focus_expr = parse("sqrt(2)*sin(x + 1/4*pi)", &mut ctx)
        .unwrap_or_else(|err| panic!("focus parse: {err}"));
    let target_expr = parse("a", &mut ctx).unwrap_or_else(|err| panic!("target parse: {err}"));

    let rewrite_match = super::super::try_find_trig_phase_shift_cancellation_match(
        &mut ctx,
        focus_expr,
        target_expr,
        false,
    );

    assert!(rewrite_match.is_none());
}
#[test]
fn trig_phase_shift_cancellation_match_rejects_linear_focus_shifted_candidate_against_plain_trig_target(
) {
    let mut ctx = Context::new();
    let focus_expr =
        parse("sin(x) + cos(x)", &mut ctx).unwrap_or_else(|err| panic!("focus parse: {err}"));
    let target_expr = parse("sin(y)", &mut ctx).unwrap_or_else(|err| panic!("target parse: {err}"));

    let rewrite_match = super::super::try_find_trig_phase_shift_cancellation_match(
        &mut ctx,
        focus_expr,
        target_expr,
        true,
    );

    assert!(rewrite_match.is_none());
}
#[test]
fn trig_phase_shift_cancellation_match_rejects_nontrig_symbolic_scale_sum_pair_regression() {
    let mut ctx = Context::new();
    let focus_expr =
        parse("a*x^2 + b*x + c", &mut ctx).unwrap_or_else(|err| panic!("focus parse: {err}"));
    let target_expr =
        parse("x*(c/x + a*x + b)", &mut ctx).unwrap_or_else(|err| panic!("target parse: {err}"));

    let rewrite_match = super::super::try_find_trig_phase_shift_cancellation_match(
        &mut ctx,
        focus_expr,
        target_expr,
        false,
    );

    assert!(rewrite_match.is_none());
}
#[test]
fn trig_phase_shift_cancellation_match_rejects_plain_single_trig_pair_without_shift_signal_regression(
) {
    let mut ctx = Context::new();
    let focus_expr = parse("sin(x)", &mut ctx).unwrap_or_else(|err| panic!("focus parse: {err}"));
    let target_expr =
        parse("-cos(x)", &mut ctx).unwrap_or_else(|err| panic!("target parse: {err}"));

    let rewrite_match = super::super::try_find_trig_phase_shift_cancellation_match(
        &mut ctx,
        focus_expr,
        target_expr,
        false,
    );

    assert!(rewrite_match.is_none());
}
#[test]
fn phase_shift_binary_fragment_fast_rejects_single_plain_against_shift_signal() {
    let mut ctx = Context::new();
    let lhs = parse("3*sin(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs parse: {err}"));
    let rhs =
        parse("5*sin(x + arctan(4/3))", &mut ctx).unwrap_or_else(|err| panic!("rhs parse: {err}"));

    assert!(
        super::super::binary_add_pair_is_surface_plain_trig_against_shift_signal_for_phase_shift(
            &mut ctx, lhs, rhs,
        )
    );
}
#[test]
fn trig_phase_shift_cancellation_match_matches_general_shifted_sine_direct_linear_pair_regression()
{
    let mut ctx = Context::new();
    let focus_expr =
        parse("3*sin(x) + 4*cos(x)", &mut ctx).unwrap_or_else(|err| panic!("focus parse: {err}"));
    let target_expr = parse("5*sin(x + arctan(4/3))", &mut ctx)
        .unwrap_or_else(|err| panic!("target parse: {err}"));

    let rewrite_match = super::super::try_find_trig_phase_shift_cancellation_match(
        &mut ctx,
        focus_expr,
        target_expr,
        false,
    )
    .unwrap_or_else(|| panic!("rewrite match"));

    assert_eq!(
        rewrite_match.mode,
        super::super::TrigPhaseShiftCancellationMode::LinearToShifted
    );
}
#[test]
fn extract_structural_exact_phase_shift_term_data_accepts_mul_one_pi_over_four_regression() {
    let mut ctx = Context::new();
    let expr = parse("sqrt(2)*sin((1*pi)/4 + x)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr parse: {err}"));
    let x = parse("x", &mut ctx).unwrap_or_else(|err| panic!("x parse: {err}"));

    let (arg, coeff, kind, sin_sign, cos_sign) =
        super::super::extract_structural_exact_phase_shift_term_data_for_cancellation(
            &mut ctx, expr,
        )
        .unwrap_or_else(|| panic!("extract"));
    let one = ctx.num(1);

    assert_eq!(compare_expr(&ctx, arg, x), Ordering::Equal);
    assert_eq!(compare_expr(&ctx, coeff, one), Ordering::Equal);
    assert!(matches!(
        kind,
        super::super::PhaseShiftKindForCancellation::Quarter
    ));
    assert_eq!((sin_sign, cos_sign), (1, 1));
}
#[test]
fn expand_trig_phase_shift_rule_rejects_binary_add_with_nontrig_partner() {
    let mut ctx = Context::new();
    let expr =
        parse("a + sqrt(2)*sin(x + 1/4*pi)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigPhaseShiftToEnableCancellationRule;
    let rewrite = rule.apply(&mut ctx, expr, &parent_ctx);

    assert!(rewrite.is_none());
}
#[test]
fn expand_trig_phase_shift_rule_keeps_productive_binary_add_exact_exact_pair() {
    let mut ctx = Context::new();
    let expr = parse("sin(x + 1/4*pi) + -(cos(x - 1/4*pi))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigPhaseShiftToEnableCancellationRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
}
#[test]
fn expand_trig_phase_shift_rule_rejects_binary_add_exact_exact_arg_mismatch() {
    let mut ctx = Context::new();
    let expr = parse("sin(x + 1/4*pi) + -(sin(y + 1/4*pi))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigPhaseShiftToEnableCancellationRule;
    let rewrite = rule.apply(&mut ctx, expr, &parent_ctx);

    assert!(rewrite.is_none());
}
#[test]
fn expand_trig_phase_shift_rule_rejects_binary_add_other_trig_other_trig_pair() {
    let mut ctx = Context::new();
    let expr = parse("-(sin(x)*cos(1/4*pi)) + -(cos(y)*sin(1/4*pi))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigPhaseShiftToEnableCancellationRule;
    let rewrite = rule.apply(&mut ctx, expr, &parent_ctx);

    assert!(rewrite.is_none());
}
#[test]
fn expand_trig_phase_shift_rule_rejects_single_plain_against_shifted_fragment() {
    let mut ctx = Context::new();
    let expr = parse("3*sin(x) + -(5*sin(x + arctan(4/3)))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigPhaseShiftToEnableCancellationRule;
    let rewrite = rule.apply(&mut ctx, expr, &parent_ctx);

    assert!(rewrite.is_none());
}
#[test]
fn collapse_exact_zero_three_term_subset_rule_matches_phase_shift_with_passthrough_one() {
    let mut ctx = Context::new();
    let expr = parse("3*sin(x) + 4*cos(x) + 1 - 5*sin(x + arctan(4/3))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "1"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
    assert!(!rewrite.substeps.is_empty());
}
#[test]
fn collapse_exact_zero_three_term_subset_rule_matches_phase_shift_with_passthrough_one_sixth() {
    let mut ctx = Context::new();
    let expr = parse("cos(x) + sqrt(3)*sin(x) + 1 - 2*sin(x + pi/6)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite.new_expr
            }
        ),
        "1"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
    assert!(!rewrite.substeps.is_empty());
}
