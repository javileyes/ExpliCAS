//! Tests de las reglas aritméticas: familia `trig_angles` (troceo P1).

use super::*;

#[test]
fn maybe_two_term_trig_sum_to_product_equivalence_candidate_rejects_nontrig_partner_term() {
    let mut ctx = Context::new();
    let lhs_core =
        parse("sin(x) - 2*cos(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core = parse("1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        !super::super::maybe_two_term_trig_sum_to_product_equivalence_candidate(
            &ctx, lhs_core, rhs_core
        )
    );
}
#[test]
fn maybe_two_term_trig_product_to_sum_equivalence_candidate_accepts_sin_cos_pair() {
    let mut ctx = Context::new();
    let lhs_core = parse("2*sin(x)*cos(y)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core =
        parse("sin(x+y) + sin(x-y)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        super::super::maybe_two_term_trig_product_to_sum_equivalence_candidate(
            &mut ctx, lhs_core, rhs_core
        )
    );
}
#[test]
fn maybe_two_term_trig_product_to_sum_equivalence_candidate_rejects_reciprocal_mixed_pair() {
    let mut ctx = Context::new();
    let lhs_core =
        parse("tan(x)*cot(x) - 1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core = parse("2*sin(x)*sin(y) - cos(x-y) + cos(x+y)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        !super::super::maybe_two_term_trig_product_to_sum_equivalence_candidate(
            &mut ctx, lhs_core, rhs_core
        )
    );
}
#[test]
fn maybe_two_term_embedded_double_angle_expansion_candidate_accepts_scaled_sine_pair() {
    let mut ctx = Context::new();
    let lhs_core = parse("y*sin(2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core =
        parse("2*y*sin(x)*cos(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        super::super::maybe_two_term_embedded_double_angle_expansion_candidate(
            &mut ctx, lhs_core, rhs_core
        )
    );
}
#[test]
fn maybe_two_term_embedded_double_angle_expansion_candidate_rejects_triple_angle_quotient_pair() {
    let mut ctx = Context::new();
    let lhs_core = parse("2*cos(2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core = parse("sin(3*x)/sin(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        !super::super::maybe_two_term_embedded_double_angle_expansion_candidate(
            &mut ctx, lhs_core, rhs_core
        )
    );
}
#[test]
fn expand_trig_sum_to_product_to_enable_cancellation_rule_matches_symbolic_sine_sum() {
    let mut ctx = Context::new();
    let expr = parse("sin(x) + sin(y) - 2*sin((x+y)/2)*cos((x-y)/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigSumToProductToEnableCancellationRule;
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
    assert_empty_or_legacy_description(&rewrite.description, "Expand sine sum to product");
}
#[test]
fn expand_trig_sum_to_product_to_enable_cancellation_rule_matches_symbolic_cosine_difference() {
    let mut ctx = Context::new();
    let expr = parse("cos(x) - cos(y) + 2*sin((x+y)/2)*sin((x-y)/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigSumToProductToEnableCancellationRule;
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
    assert_empty_or_legacy_description(&rewrite.description, "Expand cosine difference to product");
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_general_sine_sum_to_product() {
    let mut ctx = Context::new();
    let expr = parse("sin(5*x) + sin(x) - 2*sin(3*x)*cos(2*x)", &mut ctx)
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
    assert_empty_or_legacy_description(&rewrite.description, "Expand sine sum to product");
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_general_sine_sum_to_product_with_passthrough()
{
    let mut ctx = Context::new();
    let expr = parse(
        "((sin(5*x)+sin(x)) + m) - ((2*sin(3*x)*cos(2*x)) + m)",
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
    assert_empty_or_legacy_description(&rewrite.description, "Expand sine sum to product");
}
#[test]
fn collapse_exact_zero_additive_subexpression_ignores_all_positive_pythagorean_passthrough() {
    let mut ctx = Context::new();
    let expr =
        parse("sin(x)^2 + cos(x)^2 + 1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;

    assert!(rule.apply(&mut ctx, expr, &parent_ctx).is_none());
}
#[test]
fn fast_recursive_trig_angle_sum_diff_helper_matches_recursive_six_sine() {
    let mut ctx = Context::new();
    let expr = parse("sin(6*x) - (sin(5*x)*cos(x)+cos(5*x)*sin(x))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::super::try_build_fast_recursive_trig_angle_sum_diff_zero_scope_rewrite(
        &mut ctx, expr,
    )
    .unwrap_or_else(|| panic!("fast rewrite"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Angle Sum/Diff Identity");
}
#[test]
fn collapse_exact_zero_additive_subexpression_rule_matches_half_angle_square_identity() {
    let mut ctx = Context::new();
    let expr =
        parse("sin(x)^2 - ((1-cos(2*x))/2)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let zero = ctx.num(0);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.final_expr(),
        zero
    ));
}
#[test]
fn collapse_exact_zero_additive_subexpression_rule_matches_half_angle_square_with_passthrough() {
    let mut ctx = Context::new();
    let expr = parse("((sin(x)^2) + m) - (((1-cos(2*x))/2) + m)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let zero = ctx.num(0);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.final_expr(),
        zero
    ));
}
#[test]
fn collapse_exact_zero_additive_subexpression_rule_matches_double_angle_cos_one_minus_two_sin_sq_with_passthrough(
) {
    let mut ctx = Context::new();
    let expr = parse("((cos(2*x)) + m) - ((1 - 2*sin(x)^2) + m)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let zero = ctx.num(0);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.final_expr(),
        zero
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Double Angle Expansion");
}
#[test]
fn collapse_exact_zero_additive_subexpression_rule_matches_double_angle_cos_two_cos_sq_minus_one_with_passthrough(
) {
    let mut ctx = Context::new();
    let expr = parse("((2*cos(x)^2 - 1) + m) - ((cos(2*x)) + m)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let zero = ctx.num(0);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.final_expr(),
        zero
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Double Angle Expansion");
}
#[test]
fn collapse_exact_zero_additive_subexpression_rule_matches_scaled_double_angle_cos_one_minus_two_sin_sq_with_passthrough(
) {
    let mut ctx = Context::new();
    let expr = parse("((2*cos(2*x)) + m) - ((2 - 4*sin(x)^2) + m)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let zero = ctx.num(0);

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.final_expr(),
        zero
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Double Angle Expansion");
}
#[test]
fn exact_zero_trig_double_angle_cos_variant_zero_scope_rewrite_matches_split_constants() {
    let mut ctx = Context::new();
    let expr = parse("3 - 4*sin(x)^2 - 2*cos(2*x) - 1", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite =
        super::super::try_build_exact_zero_trig_double_angle_cos_variant_zero_scope_rewrite(
            &mut ctx, expr,
        )
        .unwrap_or_else(|| panic!("rewrite"));
    let zero = ctx.num(0);

    assert_empty_or_legacy_description(&rewrite.description, "Double Angle Expansion");
    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.final_expr(),
        zero
    ));
}
#[test]
fn direct_trig_double_angle_cos_variant_equivalence_rewrite_matches_scaled_one_minus_two_sin_sq() {
    let mut ctx = Context::new();
    let lhs = parse("2*cos(2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse lhs: {err}"));
    let rhs = parse("2 - 4*sin(x)^2", &mut ctx).unwrap_or_else(|err| panic!("parse rhs: {err}"));

    let rewrite = super::super::try_build_direct_trig_double_angle_cos_variant_equivalence_rewrite(
        &mut ctx, lhs, rhs,
    )
    .unwrap_or_else(|| panic!("rewrite"));
    let zero = ctx.num(0);

    assert_empty_or_legacy_description(&rewrite.description, "Double Angle Expansion");
    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.final_expr(),
        zero
    ));
}
#[test]
fn collapse_exact_zero_product_factor_rule_matches_rational_zero_times_trig_product_to_sum_sin_sin_identity(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "(1 + 1/(1 + 1/(1 + 1/x)) - (3*x + 2)/(2*x + 1)) * (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
        &mut ctx,
    )
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
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_product_to_sum_sin_sin_three_term_identity() {
    let mut ctx = Context::new();
    let expr = parse("2*sin(x)*sin(y) - cos(x-y) + cos(x+y)", &mut ctx)
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
    assert!(
        rewrite.description.is_empty()
            || rewrite.description == "Product-to-Sum Identity"
            || rewrite.description == "Double Angle Expansion",
        "expected direct collapse or legacy label {:?}, got {:?}",
        "Product-to-Sum Identity",
        rewrite.description
    );
}
#[test]
fn exact_trig_product_to_sum_sin_sin_three_term_zero_helper_matches_raw_residual() {
    let mut ctx = Context::new();
    let expr = parse("2*sin(x)*sin(y) - cos(x-y) + cos(x+y)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite =
        super::super::try_build_exact_trig_product_to_sum_sin_sin_three_term_zero_rewrite(
            &mut ctx, expr,
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
}
#[test]
fn exact_trig_equivalence_zero_scope_rewrite_matches_two_term_double_angle() {
    let mut ctx = Context::new();
    let expr =
        parse("cos(2*x) - (1 - 2*sin(x)^2)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::super::try_build_exact_trig_equivalence_zero_scope_rewrite(&mut ctx, expr)
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
fn exact_trig_equivalence_zero_scope_rewrite_matches_reversed_two_term_double_angle() {
    let mut ctx = Context::new();
    let expr =
        parse("(1 - 2*sin(x)^2) - cos(2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::super::try_build_exact_trig_equivalence_zero_scope_rewrite(&mut ctx, expr)
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
fn exact_trig_sum_to_product_zero_scope_rewrite_matches_raw_sin_sin_residual() {
    let mut ctx = Context::new();
    let expr = parse("2*sin(x)*sin(y) - cos(x-y) + cos(x+y)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite =
        super::super::try_build_exact_trig_sum_to_product_zero_scope_rewrite(&mut ctx, expr)
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
fn maybe_exact_trig_equivalence_zero_scope_candidate_rejects_structural_trig_sum_to_product_scope()
{
    let mut ctx = Context::new();
    let expr = parse("sin(x) + sin(y) - 2*sin((x+y)/2)*cos((x-y)/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::maybe_exact_trig_equivalence_zero_scope_candidate(&mut ctx, expr));
}
#[test]
fn reject_scaled_surface_trig_power_vs_numeric_atom_before_default_simplify_preserves_special_angle(
) {
    let mut ctx = Context::new();
    let lhs = parse("cos(pi)^2", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs = parse("1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::super::reject_scaled_surface_trig_power_vs_numeric_atom_before_default_simplify(
            &mut ctx, lhs, rhs
        ),
        None
    );
}
#[test]
fn maybe_trig_double_angle_cos_variant_zero_scope_candidate_accepts_split_constants() {
    let mut ctx = Context::new();
    let expr = parse("3 - 4*sin(x)^2 - 2*cos(2*x) - 1", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::super::maybe_trig_double_angle_cos_variant_zero_scope_candidate(&mut ctx, expr));
}
#[test]
fn maybe_trig_double_angle_cos_variant_zero_scope_candidate_rejects_triple_sine_quotient_scope() {
    let mut ctx = Context::new();
    let expr = parse("sin(3*x)/sin(x) - 2*cos(2*x) - 1", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        !super::super::maybe_trig_double_angle_cos_variant_zero_scope_candidate(&mut ctx, expr)
    );
}
#[test]
fn maybe_trig_double_angle_cos_variant_zero_scope_candidate_rejects_mismatched_numeric_offset() {
    let mut ctx = Context::new();
    let expr =
        parse("3 - 4*sin(x)^2 - 2*cos(2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        !super::super::maybe_trig_double_angle_cos_variant_zero_scope_candidate(&mut ctx, expr)
    );
}
#[test]
fn maybe_trig_embedded_double_angle_factor_zero_scope_candidate_rejects_mixed_double_angle_core() {
    let mut ctx = Context::new();
    let expr = parse("2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x)-2*sin(x))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        !super::super::maybe_trig_embedded_double_angle_factor_zero_scope_candidate(&mut ctx, expr)
    );
}
#[test]
fn maybe_trig_embedded_double_angle_factor_zero_scope_candidate_rejects_triple_sine_quotient_scope()
{
    let mut ctx = Context::new();
    let expr = parse("sin(3*x)/sin(x) - 2*cos(2*x) - 1", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        !super::super::maybe_trig_embedded_double_angle_factor_zero_scope_candidate(&mut ctx, expr)
    );
}
#[test]
fn maybe_trig_embedded_double_angle_factor_zero_scope_candidate_rejects_numeric_offset_scope() {
    let mut ctx = Context::new();
    let expr =
        parse("3 - 4*sin(x)^2 - 2*cos(2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        !super::super::maybe_trig_embedded_double_angle_factor_zero_scope_candidate(&mut ctx, expr)
    );
}
#[test]
fn maybe_trig_sum_to_product_zero_candidate_rejects_mixed_trig_log_exp_fraction_scope() {
    let mut ctx = Context::new();
    let expr = parse(
        "(atanh((u^2 - 1)/(u^2 + 1)) - log(u)) + (log((p*q)^2) - 2*log(p) - 2*log(q)) + (exp(r*log(s)) - s^r) + (sin(a) + sin(b) - 2*sin((a+b)/2)*cos((a-b)/2)) + (2/(t^2 - 1) - 1/(t-1) + 1/(t+1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::maybe_trig_sum_to_product_zero_candidate(
        &ctx, expr
    ));
}
#[test]
fn direct_trig_cos_double_angle_polynomial_equivalence_matcher_matches() {
    let mut ctx = Context::new();
    let lhs = parse("2*cos(2*x)*cos(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs = parse("4*cos(x)^3-2*cos(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = crate::rules::arithmetic::try_build_direct_trig_cos_double_angle_polynomial_equivalence_rewrite(
        &mut ctx, lhs, rhs
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Double Angle Expansion");
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_trig_cos_double_angle_polynomial() {
    let mut ctx = Context::new();
    let expr = parse("2*cos(2*x)*cos(x) - (4*cos(x)^3-2*cos(x))", &mut ctx)
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
    assert!(
        rewrite.description.is_empty()
            || rewrite.description == "Product-to-Sum Identity"
            || rewrite.description == "Double Angle Expansion",
        "expected direct collapse or legacy label {:?}, got {:?}",
        "Product-to-Sum Identity",
        rewrite.description
    );
}
#[test]
fn exact_zero_cos_double_angle_polynomial_matches_factored_remaining_regression() {
    let mut ctx = Context::new();
    let expr = parse("2*cos(2*x)*cos(x) - 2*cos(x)*(2*cos(x)^2 - 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite =
        super::super::try_build_exact_zero_trig_cos_double_angle_polynomial_zero_scope_rewrite(
            &mut ctx, expr,
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
}
#[test]
fn direct_trig_mixed_double_angle_polynomial_equivalence_matcher_matches() {
    let mut ctx = Context::new();
    let lhs = parse("2*cos(2*x)*sin(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs =
        parse("4*cos(x)^2*sin(x)-2*sin(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = crate::rules::arithmetic::try_build_direct_trig_mixed_double_angle_polynomial_equivalence_rewrite(
        &mut ctx, lhs, rhs
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Double Angle Expansion");
}
#[test]
fn direct_trig_embedded_double_angle_expansion_equivalence_matcher_matches_scaled_sine_pair() {
    let mut ctx = Context::new();
    let lhs = parse("y*sin(2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs = parse("2*y*sin(x)*cos(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite =
        crate::rules::arithmetic::try_build_direct_trig_embedded_double_angle_expansion_equivalence_rewrite(
            &mut ctx, lhs, rhs,
        )
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Double Angle Expansion");
}
#[test]
fn exact_zero_embedded_double_angle_factor_matches_scaled_sine_pair() {
    let mut ctx = Context::new();
    let expr = parse("y*sin(2*x) - 2*y*sin(x)*cos(x)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite =
        super::super::try_build_exact_zero_trig_embedded_double_angle_factor_zero_scope_rewrite(
            &mut ctx, expr,
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
}
#[test]
fn exact_zero_embedded_double_angle_factor_rejects_partial_mixed_double_angle_scope() {
    let mut ctx = Context::new();
    let expr = parse("2*cos(2*x)*sin(x) - 4*cos(x)^2*sin(x)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        super::super::try_build_exact_zero_trig_embedded_double_angle_factor_zero_scope_rewrite(
            &mut ctx, expr,
        )
        .is_none()
    );
}
#[test]
fn small_direct_zero_core_rewrite_matches_trig_mixed_double_angle_core() {
    let mut ctx = Context::new();
    let expr = parse("2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x)-2*sin(x))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
fn small_direct_zero_core_rewrite_matches_trig_product_to_sum_sin_sin_core() {
    let mut ctx = Context::new();
    let expr = parse("2*sin(x)*sin(y) - cos(x-y) + cos(x+y)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
fn small_direct_zero_core_rewrite_matches_sec_tan_pythagorean_core() {
    let mut ctx = Context::new();
    let expr =
        parse("sec(x)^2 - tan(x)^2 - 1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 0);
}
#[test]
fn small_direct_zero_core_rewrite_matches_csc_cot_pythagorean_core() {
    let mut ctx = Context::new();
    let expr =
        parse("csc(x)^2 - cot(x)^2 - 1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 0);
}
#[test]
fn direct_small_zero_additive_combination_rewrite_matches_trig_and_sec_tan_pythagorean_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + (sec(x)^2 - tan(x)^2 - 1)",
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
    assert_eq!(rewrite.required_conditions.len(), 0);
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_trig_mixed_double_angle_polynomial() {
    let mut ctx = Context::new();
    let expr = parse("2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x)-2*sin(x))", &mut ctx)
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
    assert!(
        rewrite.description.is_empty() || rewrite.description == "Double Angle Expansion",
        "expected direct exact-zero collapse or the legacy double-angle label, got {:?}",
        rewrite.description
    );
}
#[test]
fn direct_trig_ratio_equivalence_rewrite_matches_half_angle_tan() {
    let mut ctx = Context::new();
    let lhs = parse("(1-cos(2*x))/sin(2*x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("tan(x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_trig_ratio_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn collapse_exact_zero_same_denominator_rule_matches_half_angle_tan_core() {
    let mut ctx = Context::new();
    let expr = parse("(((1-cos(2*x))/sin(x+x))/q) - ((tan(x))/q)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::super::try_build_exact_zero_same_denominator_rewrite(&mut ctx, expr)
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
fn collapse_exact_zero_same_denominator_rule_matches_trig_product_to_sum_cos_sin() {
    let mut ctx = Context::new();
    let expr = parse(
        "((2*cos(x)*sin(y))/q) - ((sin(x+y) - sin(x-y))/q)",
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
    assert_empty_or_legacy_description(&rewrite.description, "Product-to-Sum Identity");
}
#[test]
fn expand_trig_sine_product_triple_angle_to_enable_cancellation_rule_matches_cubic_cosine_residual()
{
    let mut ctx = Context::new();
    let expr = parse("2*sin(2*x)*sin(x) - (4*cos(x) - 4*cos(x)^3)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigSineProductTripleAngleToEnableCancellationRule;
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
        "Product-to-Sum and Triple-Angle Identity",
    );
    assert_eq!(rewrite.substeps.len(), 4);
}
#[test]
fn tan_triple_angle_contraction_matcher_matches_canonical_rational_form() {
    let mut ctx = Context::new();
    let expr = parse("((3*tan(x)-tan(x)^3)/(1-3*tan(x)^2))", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));
    let x = parse("x", &mut ctx).unwrap_or_else(|err| panic!("x: {err}"));

    let (matched_arg, _) = super::super::match_tan_triple_angle_contraction_arg(&mut ctx, expr)
        .unwrap_or_else(|| panic!("matcher"));

    assert_eq!(
        cas_ast::ordering::compare_expr(&ctx, matched_arg, x),
        std::cmp::Ordering::Equal
    );
}
#[test]
fn direct_multi_angle_equivalence_rewrite_matches_tangent_triple_angle_reverse() {
    let mut ctx = Context::new();
    let lhs = parse("((3*tan(x)-tan(x)^3)/(1-3*tan(x)^2))", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("tan(3*x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_multi_angle_equivalence_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Triple Angle Identity");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
    assert_eq!(rewrite.required_conditions.len(), 1);
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_tangent_triple_angle_reverse() {
    let mut ctx = Context::new();
    let expr = parse("((3*tan(x)-tan(x)^3)/(1-3*tan(x)^2)) - tan(3*x)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

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
    assert_empty_or_legacy_description(&rewrite.description, "Triple Angle Identity");
    assert_eq!(rewrite.required_conditions.len(), 1);
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_tangent_triple_angle_wrapper() {
    let mut ctx = Context::new();
    let expr = parse(
        "((tan(3*x)) + m) - (((3*tan(x)-tan(x)^3)/(1-3*tan(x)^2)) + m)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

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
}
