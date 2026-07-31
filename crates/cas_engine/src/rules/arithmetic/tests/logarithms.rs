//! Tests de las reglas aritméticas: familia `logarithms` (troceo P1).

use super::*;

#[test]
fn default_simplifier_collapses_div_add_common_factor_log_power_residual() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(x^2*ln(x^2-1)^4 - ln(x^2-1)^4)/(x^2-1) - ln(x^2-1)^4",
        &mut simplifier.context,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));
    let (simplified, _) = simplifier.simplify(expr);
    let zero = simplifier.context.num(0);
    assert_eq!(
        compare_expr(&simplifier.context, simplified, zero),
        Ordering::Equal
    );
}
#[test]
fn expand_log_abs_mul_div_to_enable_cancellation_rule_matches_scaled_ln_product() {
    let mut ctx = Context::new();
    let expr = parse("2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandLogAbsMulDivToEnableCancellationRule;
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
#[test]
fn expand_log_abs_mul_div_to_enable_cancellation_rule_matches_scaled_log_product() {
    let mut ctx = Context::new();
    let expr = parse("2*log(abs(x*y)) - 2*log(abs(x)) - 2*log(abs(y))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandLogAbsMulDivToEnableCancellationRule;
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
#[test]
fn expand_log_abs_mul_div_to_enable_cancellation_rule_rejects_nonexpandable_abs_log_sum() {
    let mut ctx = Context::new();
    let expr =
        parse("ln(abs(x)) + ln(abs(y)) - z", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandLogAbsMulDivToEnableCancellationRule;
    let rewrite = rule.apply(&mut ctx, expr, &parent_ctx);

    assert!(rewrite.is_none());
}
#[test]
fn maybe_log_abs_mul_div_zero_candidate_rejects_nonabs_log_product_scope() {
    let mut ctx = Context::new();
    let expr = parse("ln((p*q)^2) - 2*ln(p) - 2*ln(q)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::maybe_log_abs_mul_div_zero_candidate(
        &mut ctx, expr
    ));
}
#[test]
fn maybe_log_abs_mul_div_zero_candidate_rejects_mixed_nonlog_scope() {
    let mut ctx = Context::new();
    let expr = parse("2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y)) + z", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::maybe_log_abs_mul_div_zero_candidate(
        &mut ctx, expr
    ));
}
#[test]
fn expand_log_product_power_to_enable_cancellation_rule_matches_mixed_power_product() {
    let mut ctx = Context::new();
    let expr = parse("ln(x^3) + ln(y^2) - ln(x^3 * y^2)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandLogProductPowerToEnableCancellationRule;
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
fn expand_log_product_power_to_enable_cancellation_rule_rejects_nonexpandable_log_sum() {
    let mut ctx = Context::new();
    let expr = parse("ln(x) + ln(y) - z", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandLogProductPowerToEnableCancellationRule;
    let rewrite = rule.apply(&mut ctx, expr, &parent_ctx);

    assert!(rewrite.is_none());
}
#[test]
fn expand_log_product_power_cancellation_does_not_panic_on_mixed_ln_and_two_arg_log() {
    // Regression: `ln(2*x) - ln(x) - log(2,8)` mixes a natural log (base =
    // ln_base_sentinel(), a non-arena ExprId) with a two-arg log(2,8). The base
    // comparison in log_terms_match_up_to_abs_subject_for_cancellation used to call
    // compare_expr on the sentinel, panicking with an out-of-bounds Context::get.
    // The ln-base sentinel never matches the real base 2, so no cancellation applies.
    let mut ctx = Context::new();
    let expr =
        parse("ln(2*x) - ln(x) - log(2,8)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandLogProductPowerToEnableCancellationRule;
    // Must not panic; bases differ (ln vs log base 2) so the rule does not cancel.
    assert!(rule.apply(&mut ctx, expr, &parent_ctx).is_none());
}
#[test]
fn maybe_log_product_power_zero_candidate_rejects_mixed_nonlog_scope() {
    let mut ctx = Context::new();
    let expr = parse("ln(x^3) + ln(y^2) - ln(x^3 * y^2) + z", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::maybe_log_product_power_zero_candidate(
        &mut ctx, expr
    ));
}
#[test]
fn maybe_log_product_power_zero_candidate_rejects_insufficient_cancellation_components() {
    let mut ctx = Context::new();
    let expr =
        parse("ln((x*y)^2) - 2*ln(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::maybe_log_product_power_zero_candidate(
        &mut ctx, expr
    ));
}
#[test]
fn maybe_direct_small_zero_additive_combination_candidate_rejects_mixed_log_exp_fraction_scope() {
    let mut ctx = Context::new();
    let expr = parse("x^y - exp(y*log(x)) - 1/(x+1) - 2/(x^2 - 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}
#[test]
fn maybe_direct_small_zero_additive_combination_candidate_rejects_root_log_exp_fraction_scope() {
    let mut ctx = Context::new();
    let expr = parse(
        "sqrt(2 * sqrt(x - 1) + x) + exp(y*log(x)) + 1/(x + 1) + 2/(x^2 - 1) - x^y - 1/(x - 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}
#[test]
fn direct_core_equivalence_rewrite_matches_log_power_pair_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("ln((x+1)^2)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("2*ln(abs(x+1))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Log Expansion Identity");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_core_equivalence_rewrite_matches_log_abs_product_pair_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("ln(abs(x*y))", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("ln(abs(x)) + ln(abs(y))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Log Expansion Identity");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_core_equivalence_rewrite_matches_grouped_ln_power_pair_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("ln((x*y)^2)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("ln(x^2) + ln(y^2)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Log Expansion Identity");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_core_equivalence_rewrite_matches_scaled_log_abs_product_pair_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("2*ln(abs(x*y))", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs =
        parse("2*ln(abs(x)) + 2*ln(abs(y))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Log Expansion Identity");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn negated_log_nonreciprocal_reject_matches_power_and_abs_pairs_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("-ln(x^2)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("ln((x*y)^2)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    assert_eq!(
        super::super::reject_negated_log_pair_without_reciprocal_shape_before_default_simplify(
            &mut ctx, lhs, rhs
        ),
        Some(false)
    );

    let lhs = parse("-2*ln(abs(x))", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("2*ln(abs(x*y))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    assert_eq!(
        super::super::reject_negated_log_pair_without_reciprocal_shape_before_default_simplify(
            &mut ctx, lhs, rhs
        ),
        Some(false)
    );
}
#[test]
fn negated_log_nonreciprocal_reject_defers_reciprocal_shapes_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("-ln(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("ln(1/x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    assert_eq!(
        super::super::reject_negated_log_pair_without_reciprocal_shape_before_default_simplify(
            &mut ctx, lhs, rhs
        ),
        None
    );

    let lhs = parse("-ln(x^2)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("ln(x^(-2))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    assert_eq!(
        super::super::reject_negated_log_pair_without_reciprocal_shape_before_default_simplify(
            &mut ctx, lhs, rhs
        ),
        None
    );
}
#[test]
fn direct_core_equivalence_rewrite_rejects_grouped_general_base_log_power_pair() {
    let mut ctx = Context::new();
    let lhs = parse("log(b,(x*y)^2)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("2*log(b,x) + 2*log(b,y)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}
#[test]
fn direct_core_equivalence_rewrite_matches_log_chain_product_pair() {
    let mut ctx = Context::new();
    let lhs = parse("log(b,a)*log(a,c)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("log(b,c)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Log Chain Identity");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_log_expand_double_squared_passthrough() {
    let mut ctx = Context::new();
    let expr = parse(
        "((((ln(x^2-y^2))^2) + m)^2) - ((((ln(x-y)+ln(x+y))^2) + m)^2)",
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
}
#[test]
fn collapse_exact_zero_additive_subexpression_matches_log_contract_double_squared_passthrough() {
    let mut ctx = Context::new();
    let expr = parse(
        "((((ln(x^2)+ln(y^2))^2) + m)^2) - ((((ln((x*y)^2))^2) + m)^2)",
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
}
