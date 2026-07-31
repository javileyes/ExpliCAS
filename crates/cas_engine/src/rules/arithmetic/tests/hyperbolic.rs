//! Tests de las reglas aritméticas: familia `hyperbolic` (troceo P1).

use super::*;

#[test]
fn maybe_two_term_hyperbolic_direct_core_equivalence_candidate_rejects_polynomial_partner() {
    let mut ctx = Context::new();
    let lhs_core = parse("sinh(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core = parse("x^2 - 1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        !super::super::maybe_two_term_hyperbolic_direct_core_equivalence_candidate(
            &mut ctx, lhs_core, rhs_core
        )
    );
}
#[test]
fn maybe_two_term_hyperbolic_direct_core_equivalence_candidate_accepts_tanh_ratio_partner() {
    let mut ctx = Context::new();
    let lhs_core = parse("tanh(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core = parse("(e^x - e^(-x))/(e^x + e^(-x))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        super::super::maybe_two_term_hyperbolic_direct_core_equivalence_candidate(
            &mut ctx, lhs_core, rhs_core
        )
    );
}
#[test]
fn maybe_two_term_hyperbolic_direct_identity_candidate_rejects_cross_swap_pair() {
    let mut ctx = Context::new();
    let lhs_core = parse("sinh(x)*cosh(y)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core = parse("cosh(x)*sinh(y)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        !super::super::maybe_two_term_hyperbolic_direct_identity_candidate(
            &mut ctx, lhs_core, rhs_core
        )
    );
}
#[test]
fn maybe_two_term_hyperbolic_direct_identity_candidate_keeps_mul_vs_function_pair() {
    let mut ctx = Context::new();
    let lhs_core =
        parse("2*sinh(x)*cosh(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core = parse("sinh(2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        super::super::maybe_two_term_hyperbolic_direct_identity_candidate(
            &mut ctx, lhs_core, rhs_core
        )
    );
}
#[test]
fn scaled_single_hyperbolic_zero_scope_reject_preserves_double_angle_product_match() {
    let mut ctx = Context::new();
    let expr =
        parse("2*sinh(x)*cosh(x)-sinh(2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::reject_linear_hyperbolic_combination_before_zero_scope(&mut ctx, expr));
}
#[test]
fn linear_hyperbolic_zero_scope_reject_matches_symbolic_scale_mismatch() {
    let mut ctx = Context::new();
    let expr =
        parse("x*cosh(2*x+1) - sinh(2*x+1)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::super::reject_linear_hyperbolic_combination_before_zero_scope(&mut ctx, expr));
}
#[test]
fn linear_hyperbolic_zero_scope_reject_matches_expanded_scaled_mismatch() {
    let mut ctx = Context::new();
    let expr = parse("2*x*cosh(2*x+1) + 3*cosh(2*x+1) - sinh(2*x+1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::super::reject_linear_hyperbolic_combination_before_zero_scope(&mut ctx, expr));
}
#[test]
fn linear_hyperbolic_zero_scope_reject_preserves_same_family_numeric_cancellation() {
    let mut ctx = Context::new();
    let expr = parse("2*cosh(x) - cosh(x) - cosh(x)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::reject_linear_hyperbolic_combination_before_zero_scope(&mut ctx, expr));
}
#[test]
fn maybe_two_term_tanh_exp_equivalence_candidate_accepts_tanh_ratio_partner() {
    let mut ctx = Context::new();
    let lhs_core = parse("tanh(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core = parse("(e^x - e^(-x))/(e^x + e^(-x))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::super::maybe_two_term_tanh_exp_equivalence_candidate(
        &mut ctx, lhs_core, rhs_core
    ));
}
#[test]
fn maybe_two_term_tanh_exp_equivalence_candidate_rejects_missing_exp_partner() {
    let mut ctx = Context::new();
    let lhs_core = parse("tanh(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core =
        parse("sinh(x) + cosh(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        !super::super::maybe_two_term_tanh_exp_equivalence_candidate(&mut ctx, lhs_core, rhs_core)
    );
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_direct_hyperbolic_pythagorean_zero_scope() {
    let mut ctx = Context::new();
    let expr =
        parse("cosh(x)^2 - sinh(x)^2 - 1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

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
fn direct_exact_hyperbolic_zero_scope_matches_additive_pythagorean_orientation() {
    let mut ctx = Context::new();
    let expr =
        parse("cosh(x)^2 - sinh(x)^2 - 1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite =
        super::super::try_build_exact_hyperbolic_equivalence_zero_scope_rewrite(&mut ctx, expr)
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
fn direct_exact_hyperbolic_zero_scope_rejects_scaled_single_term_mismatch() {
    let mut ctx = Context::new();
    let expr =
        parse("x*cosh(2*x+1) - sinh(2*x+1)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        super::super::try_build_exact_hyperbolic_equivalence_zero_scope_rewrite(&mut ctx, expr)
            .is_none()
    );
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_exp_hyperbolic_sum() {
    let mut ctx = Context::new();
    let expr =
        parse("e^x - sinh(x) - cosh(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Hyperbolic Sum to Exponential");
}
#[test]
fn collapse_exact_zero_common_scaled_difference_rule_matches_exp_hyperbolic_sum_pair() {
    let mut ctx = Context::new();
    let expr = parse("k*(e^x) - k*(sinh(x) + cosh(x))", &mut ctx)
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
    assert_empty_or_legacy_description(&rewrite.description, "Hyperbolic Sum to Exponential");
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_hyperbolic_sinh_sum_to_product() {
    let mut ctx = Context::new();
    let expr = parse(
        "sinh(x) + sinh(y) - 2*sinh((x+y)/2)*cosh((x-y)/2)",
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
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Hyperbolic Product-to-Sum Identity");
}
#[test]
fn maybe_direct_small_zero_additive_combination_candidate_accepts_log_square_and_hyperbolic_cubic_sum(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}
#[test]
fn direct_small_zero_additive_combination_rewrite_matches_log_square_and_hyperbolic_cubic_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite =
        super::super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_hyperbolic_cosh_difference_to_product() {
    let mut ctx = Context::new();
    let expr = parse(
        "cosh(x) - cosh(y) - 2*sinh((x+y)/2)*sinh((x-y)/2)",
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
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Hyperbolic Product-to-Sum Identity");
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_hyperbolic_cosh_sum_to_product_with_passthrough(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "((cosh(x)+cosh(y)) + m) - ((2*cosh((x+y)/2)*cosh((x-y)/2)) + m)",
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
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(&rewrite.description, "Hyperbolic Product-to-Sum Identity");
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_recursive_hyperbolic_sinh_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "sinh(6*x) - (sinh(5*x)*cosh(x) + cosh(5*x)*sinh(x))",
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
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(
        &rewrite.description,
        "Hyperbolic Angle Sum/Difference Identity",
    );
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_recursive_hyperbolic_cosh_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "cosh(6*x) - (cosh(5*x)*cosh(x) + sinh(5*x)*sinh(x))",
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
                id: rewrite.new_expr
            }
        ),
        "0"
    );
    assert_empty_or_legacy_description(
        &rewrite.description,
        "Hyperbolic Angle Sum/Difference Identity",
    );
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_hyperbolic_product_sum_triple_angle() {
    let mut ctx = Context::new();
    let expr = parse("2*sinh(2*x)*cosh(x) - (4*sinh(x) + 4*sinh(x)^3)", &mut ctx)
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
        "0"
    );
    assert_empty_or_legacy_description(
        &rewrite.description,
        "Hyperbolic Product-to-Sum and Triple-Angle Identity",
    );
}
#[test]
fn direct_safe_hyperbolic_core_equivalence_rewrite_matches_cosh_cubic_polynomial_pair() {
    let mut ctx = Context::new();
    let lhs = parse("2*sinh(2*x)*sinh(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("4*cosh(x)^3 - 4*cosh(x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_safe_hyperbolic_core_equivalence_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Hyperbolic Product-to-Sum and Triple-Angle Identity",
    );
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn hyperbolic_sinh_cubic_polynomial_helper_rewrites_direct_product() {
    let mut ctx = Context::new();
    let expr = parse("2*sinh(2*x)*cosh(x)", &mut ctx).unwrap_or_else(|err| panic!("expr: {err}"));
    let rhs = parse("4*sinh(x) + 4*sinh(x)^3", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewritten =
        super::super::try_rewrite_hyperbolic_product_sum_sinh_cubic_polynomial_for_cancellation(
            &mut ctx, expr,
        )
        .unwrap_or_else(|| panic!("rewrite"));

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx, rewritten, rhs
    ));
}
#[test]
fn direct_safe_hyperbolic_core_equivalence_rewrite_matches_sinh_cubic_polynomial_pair() {
    let mut ctx = Context::new();
    let lhs = parse("2*sinh(2*x)*cosh(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("4*sinh(x) + 4*sinh(x)^3", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_safe_hyperbolic_core_equivalence_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Hyperbolic Product-to-Sum and Triple-Angle Identity",
    );
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_safe_hyperbolic_core_equivalence_rewrite_matches_exp_hyperbolic_sum_pair() {
    let mut ctx = Context::new();
    let lhs = parse("e^x", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sinh(x) + cosh(x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_safe_hyperbolic_core_equivalence_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Hyperbolic Sum to Exponential");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_safe_hyperbolic_core_equivalence_rewrite_matches_atanh_ln_definition_pair() {
    let mut ctx = Context::new();
    let lhs = parse("2*atanh(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("ln((1 + x)/(1 - x))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_safe_hyperbolic_core_equivalence_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Inverse Hyperbolic Log Definition");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_safe_hyperbolic_core_equivalence_rewrite_matches_atanh_square_ratio_ln_pair() {
    let mut ctx = Context::new();
    let lhs =
        parse("atanh((x^2 - 1)/(x^2 + 1))", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("ln(x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_safe_hyperbolic_core_equivalence_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Inverse Hyperbolic Log Definition");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_safe_hyperbolic_core_equivalence_rewrite_rejects_atanh_common_log_pair() {
    let mut ctx = Context::new();
    let lhs = parse("2*atanh(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("log10((1 + x)/(1 - x))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_safe_hyperbolic_core_equivalence_rewrite(&mut ctx, lhs, rhs);

    assert!(rewrite.is_none());
}
#[test]
fn direct_safe_hyperbolic_core_equivalence_rewrite_rejects_atanh_without_log_target() {
    let mut ctx = Context::new();
    let lhs =
        parse("atanh((x^2 - 1)/(x^2 + 1))", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sin(x) + cos(x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_safe_hyperbolic_core_equivalence_rewrite(&mut ctx, lhs, rhs);

    assert!(rewrite.is_none());
}
#[test]
fn atanh_common_log_definition_mismatch_pair_detector_matches_scaled_forms() {
    let mut ctx = Context::new();
    let lhs =
        parse("1/2*log10((1 + x)/(1 - x))", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("atanh(x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    assert!(super::super::is_atanh_common_log_definition_mismatch_pair(
        &ctx, lhs, rhs
    ));
}
#[test]
fn exact_zero_identity_rewrite_rejects_atanh_common_log_pair() {
    let mut ctx = Context::new();
    let expr = parse("1/2*log10((1 + x)/(1 - x)) - atanh(x)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr);

    assert!(rewrite.is_none());
}
#[test]
fn exact_zero_identity_rewrite_matches_atanh_square_ratio_ln_pair() {
    let mut ctx = Context::new();
    let expr = parse("atanh((x^2 - 1)/(x^2 + 1)) - ln(x)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(rewrite.final_expr(), ctx.num(0));
    assert_empty_or_legacy_description(&rewrite.description, "Inverse Hyperbolic Log Definition");
}
#[test]
fn exact_zero_product_factor_rule_rejects_atanh_common_log_pair_factor() {
    let mut ctx = Context::new();
    let expr = parse("1/2*(log10((1 + x)/(1 - x)) - 2*atanh(x))", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));
    let parent_ctx = ParentContext::root().with_domain_mode(crate::DomainMode::Generic);

    let rewrite =
        super::super::try_build_exact_zero_product_factor_rewrite(&mut ctx, expr, &parent_ctx);

    assert!(rewrite.is_none());
}
#[test]
fn collapse_exact_zero_three_term_subset_rule_rejects_atanh_common_log_pair_with_plain_passthrough()
{
    let mut ctx = Context::new();
    let expr = parse("log10((1+x)/(1-x)) + 2*y^2 - 2*atanh(x)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));
    let parent_ctx = ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;

    let rewrite = rule.apply(&mut ctx, expr, &parent_ctx);

    assert!(rewrite.is_none());
}
#[test]
fn direct_core_equivalence_rewrite_matches_tanh_exp_definition_pair() {
    let mut ctx = Context::new();
    let lhs = parse("tanh(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs =
        parse("(e^x - e^(-x))/(e^x + e^(-x))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Recognize Hyperbolic from Exponential",
    );
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_core_equivalence_rewrite_rejects_mismatched_symbolic_hyperbolic_pair_before_default_simplify(
) {
    let mut ctx = Context::new();
    let lhs = parse("cosh(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("cosh(y)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}
#[test]
fn direct_core_equivalence_rewrite_rejects_same_arg_hyperbolic_square_gap_before_default_simplify()
{
    let mut ctx = Context::new();
    let lhs = parse("cosh(x)^2", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sinh(x)^2", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}
#[test]
fn direct_core_equivalence_rewrite_keeps_hyperbolic_pythagorean_pair_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("cosh(x)^2 - 1", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sinh(x)^2", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_some());
}
#[test]
fn direct_core_equivalence_rewrite_rejects_hyperbolic_additive_atomic_tail_before_default_simplify()
{
    let mut ctx = Context::new();
    let lhs = parse("cosh(x)^2 + a", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sinh(x)^2", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}
#[test]
fn direct_core_equivalence_rewrite_keeps_hyperbolic_pythagorean_plus_one_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("sinh(x)^2 + 1", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("cosh(x)^2", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_some());
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_hyperbolic_cosh_cubic_polynomial() {
    let mut ctx = Context::new();
    let expr = parse("2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x))", &mut ctx)
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
        "0"
    );
    assert_eq!(
        rewrite.description,
        "Hyperbolic Product-to-Sum and Triple-Angle Identity"
    );
}
#[test]
fn collapse_exact_zero_same_denominator_rule_matches_hyperbolic_sinh_sum_to_product() {
    let mut ctx = Context::new();
    let expr = parse(
        "((sinh(x)+sinh(y))/q) - ((2*sinh((x+y)/2)*cosh((x-y)/2))/q)",
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
    assert_empty_or_legacy_description(&rewrite.description, "Hyperbolic Product-to-Sum Identity");
}
#[test]
fn expand_hyperbolic_angle_sum_diff_to_enable_cancellation_rule_matches_sinh_sum() {
    let mut ctx = Context::new();
    let expr = parse("sinh(x+y) - (sinh(x)*cosh(y) + cosh(x)*sinh(y))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandHyperbolicAngleSumDiffToEnableCancellationRule;
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
    assert_eq!(
        rewrite.description,
        "Expand hyperbolic angle sum/difference"
    );
}
#[test]
fn hyperbolic_angle_sum_diff_expansion_skips_powered_affine_primitive_terms() {
    let mut ctx = Context::new();
    let expr = parse("1/3*cosh(1-2*x)^3 - cosh(1-2*x)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(&mut ctx, expr).is_none(),
        "cubic primitive terms should not trigger broad hyperbolic expansion"
    );

    let direct = parse("2*sinh(x+y)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rewritten = try_rewrite_hyperbolic_angle_sum_diff_for_cancellation(&mut ctx, direct)
        .unwrap_or_else(|| panic!("direct hyperbolic sum should still expand"));
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewritten
            }
        ),
        "2 * (sinh(x) * cosh(y) + cosh(x) * sinh(y))"
    );
}
#[test]
fn collapse_exact_zero_three_term_subset_rule_matches_hyperbolic_sum_with_passthrough_one() {
    let mut ctx = Context::new();
    let expr = parse(
        "sinh(x+y) + 1 - (sinh(x)*cosh(y) + cosh(x)*sinh(y))",
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
                id: rewrite.new_expr
            }
        ),
        "1"
    );
    assert_empty_or_legacy_description(
        &rewrite.description,
        "Hyperbolic Angle Sum/Difference Identity",
    );
    assert!(rewrite.substeps.is_empty());
}
#[test]
fn expand_hyperbolic_pythagorean_factor_to_enable_cancellation_rule_matches_cubic_residual() {
    let mut ctx = Context::new();
    let expr = parse("4*cosh(x)*sinh(x)^2 + 4*cosh(x) - 4*cosh(x)^3", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandHyperbolicPythagoreanFactorToEnableCancellationRule;
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
    assert_eq!(rewrite.substeps.len(), 3);
}
#[test]
fn expand_hyperbolic_pythagorean_factor_to_enable_cancellation_rule_matches_symbolic_coefficient() {
    let mut ctx = Context::new();
    let expr = parse(
        "4*a*cosh(x)*sinh(x)^2 + 4*a*cosh(x) - 4*a*cosh(x)^3",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandHyperbolicPythagoreanFactorToEnableCancellationRule;
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
    assert_eq!(rewrite.substeps.len(), 3);
}
#[test]
fn expand_hyperbolic_pythagorean_factor_to_enable_cancellation_rule_matches_sinh_factored_residual()
{
    let mut ctx = Context::new();
    let expr = parse("sinh(2*x+1)*(cosh(2*x+1)^2 - 1) - sinh(2*x+1)^3", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandHyperbolicPythagoreanFactorToEnableCancellationRule;
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
    assert_eq!(rewrite.substeps.len(), 2);
}
