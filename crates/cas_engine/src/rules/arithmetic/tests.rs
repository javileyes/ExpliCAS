//! Tests de las reglas aritméticas, extraídos del módulo (P2).
//!
//! Vivían como `mod tests` inline dentro de `rules/arithmetic.rs`, donde
//! eran 8.524 de sus 39.346 líneas.
//!
//! No confundir con `rules/arithmetic_tests.rs`, que es un fichero aparte y
//! mucho más pequeño (145 líneas) que ya existía.

use super::{
    canonicalize_nested_integer_powers, exprs_equal_up_to_add_term_order,
    extract_scaled_double_sine_product_for_cancellation, term_has_matrix_product_factor,
    try_build_direct_sum_diff_cubes_quotient_equivalence_rewrite,
    try_rewrite_hyperbolic_angle_sum_diff_for_cancellation, CollapseExactOneShiftedQuotientRule,
    CollapseExactZeroCommonScaledDifferenceRule, CollapseExactZeroProductFactorRule,
    CollapseExactZeroThreeTermSubsetRule, ExpandHyperbolicAngleSumDiffToEnableCancellationRule,
    ExpandHyperbolicPythagoreanFactorToEnableCancellationRule,
    ExpandLogAbsMulDivToEnableCancellationRule, ExpandLogProductPowerToEnableCancellationRule,
    ExpandOddHalfPowerToEnableCancellationRule, ExpandTrigPhaseShiftToEnableCancellationRule,
    ExpandTrigSineProductTripleAngleToEnableCancellationRule,
    ExpandTrigSquareIdentityToEnableCancellationRule,
    ExpandTrigSumToProductToEnableCancellationRule, SubSelfToZeroRule,
    SubtractExpandedSumDiffCubesQuotientRule,
};
use crate::parent_context::ParentContext;
use crate::rule::Rule;
use crate::DomainMode;
use cas_ast::ordering::compare_expr;
use cas_ast::{Context, Expr};
use cas_formatter::DisplayExpr;
use cas_parser::parse;
use std::cmp::Ordering;

fn assert_empty_or_legacy_description(actual: &str, expected: &str) {
    assert!(
        actual.is_empty() || actual == expected,
        "expected direct collapse or legacy label {:?}, got {:?}",
        expected,
        actual
    );
}

#[test]
fn term_has_matrix_product_factor_flags_only_matrix_products() {
    // A matrix literal used as a factor of a multiplication is a
    // non-commutative product: the cancellation/reorder machinery must not
    // treat such a term as a commutative-ring element.
    let flagged = [
        "[[1,2],[3,4]]*[[5,6],[7,8]]",
        "[[1,2],[3,4]]*[[5,6],[7,8]] - [[5,6],[7,8]]*[[1,2],[3,4]]",
        "2*[[1,2],[3,4]]", // scalar·matrix still has a matrix factor
        "[[1,2],[3,4]]^2", // matrix power expands to repeated products
        "x*([[1,0],[0,1]]*[[1,2],[3,4]])",
    ];
    for input in flagged {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx).unwrap_or_else(|err| panic!("parse {input}: {err}"));
        assert!(
            term_has_matrix_product_factor(&ctx, expr),
            "{input}: expected matrix product factor to be flagged"
        );
    }

    // Matrices that appear only as additive terms or only inside a function
    // argument are NOT non-commutative products, so they stay unflagged
    // (commutative add-reordering and scalar matchers remain enabled).
    let unflagged = [
        "[[1,2],[3,4]] - [[1,2],[3,4]]", // matrices in a difference, not a product
        "[[1,2],[3,4]] + [[5,6],[7,8]]",
        "det([[1,2],[3,4]])*x", // matrix only inside det(...) argument
        "x*y - y*x",            // no matrices at all
        "(x+1)*(x-1)",
    ];
    for input in unflagged {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx).unwrap_or_else(|err| panic!("parse {input}: {err}"));
        assert!(
            !term_has_matrix_product_factor(&ctx, expr),
            "{input}: expected NO matrix product factor"
        );
    }
}

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
fn subtraction_self_cancel_rule_matches_abs_sub_mirror_in_generic() {
    let mut ctx = Context::new();
    let expr = parse(
        "abs((2*u)/(u^2 - 1) - 1) - abs(1 - 2*u/(u^2 - 1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = SubSelfToZeroRule;
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
fn expand_odd_half_power_to_enable_cancellation_rule_matches_nonnegative_target() {
    let mut ctx = Context::new();
    let expr =
        parse("sqrt(x^5) - x^2*sqrt(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let expected = parse("x^2*sqrt(x) - x^2*sqrt(x)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse expected: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandOddHalfPowerToEnableCancellationRule;
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
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: expected
            }
        )
    );
    assert_eq!(rewrite.required_conditions.len(), 1);
}

#[test]
fn expand_odd_half_power_to_enable_cancellation_rule_matches_reversed_side() {
    let mut ctx = Context::new();
    let expr =
        parse("x^3*sqrt(x) - sqrt(x^7)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let expected = parse("x^3*sqrt(x) - x^3*sqrt(x)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse expected: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandOddHalfPowerToEnableCancellationRule;
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
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: expected
            }
        )
    );
    assert_eq!(rewrite.required_conditions.len(), 1);
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

    assert!(!super::maybe_log_abs_mul_div_zero_candidate(&mut ctx, expr));
}

#[test]
fn maybe_log_abs_mul_div_zero_candidate_rejects_mixed_nonlog_scope() {
    let mut ctx = Context::new();
    let expr = parse("2*ln(abs(x*y)) - 2*ln(abs(x)) - 2*ln(abs(y)) + z", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_log_abs_mul_div_zero_candidate(&mut ctx, expr));
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

    assert!(!super::maybe_log_product_power_zero_candidate(
        &mut ctx, expr
    ));
}

#[test]
fn maybe_log_product_power_zero_candidate_rejects_insufficient_cancellation_components() {
    let mut ctx = Context::new();
    let expr =
        parse("ln((x*y)^2) - 2*ln(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_log_product_power_zero_candidate(
        &mut ctx, expr
    ));
}

#[test]
fn maybe_two_term_trig_sum_to_product_equivalence_candidate_rejects_nontrig_partner_term() {
    let mut ctx = Context::new();
    let lhs_core =
        parse("sin(x) - 2*cos(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core = parse("1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        !super::maybe_two_term_trig_sum_to_product_equivalence_candidate(&ctx, lhs_core, rhs_core)
    );
}

#[test]
fn maybe_two_term_trig_product_to_sum_equivalence_candidate_accepts_sin_cos_pair() {
    let mut ctx = Context::new();
    let lhs_core = parse("2*sin(x)*cos(y)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core =
        parse("sin(x+y) + sin(x-y)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        super::maybe_two_term_trig_product_to_sum_equivalence_candidate(
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
        !super::maybe_two_term_trig_product_to_sum_equivalence_candidate(
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
        super::maybe_two_term_embedded_double_angle_expansion_candidate(
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
        !super::maybe_two_term_embedded_double_angle_expansion_candidate(
            &mut ctx, lhs_core, rhs_core
        )
    );
}

#[test]
fn maybe_two_term_hyperbolic_direct_core_equivalence_candidate_rejects_polynomial_partner() {
    let mut ctx = Context::new();
    let lhs_core = parse("sinh(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core = parse("x^2 - 1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        !super::maybe_two_term_hyperbolic_direct_core_equivalence_candidate(
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
        super::maybe_two_term_hyperbolic_direct_core_equivalence_candidate(
            &mut ctx, lhs_core, rhs_core
        )
    );
}

#[test]
fn maybe_two_term_hyperbolic_direct_identity_candidate_rejects_cross_swap_pair() {
    let mut ctx = Context::new();
    let lhs_core = parse("sinh(x)*cosh(y)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core = parse("cosh(x)*sinh(y)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_two_term_hyperbolic_direct_identity_candidate(
        &mut ctx, lhs_core, rhs_core
    ));
}

#[test]
fn maybe_two_term_hyperbolic_direct_identity_candidate_keeps_mul_vs_function_pair() {
    let mut ctx = Context::new();
    let lhs_core =
        parse("2*sinh(x)*cosh(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core = parse("sinh(2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::maybe_two_term_hyperbolic_direct_identity_candidate(
        &mut ctx, lhs_core, rhs_core
    ));
}

#[test]
fn scaled_single_hyperbolic_zero_scope_reject_preserves_double_angle_product_match() {
    let mut ctx = Context::new();
    let expr =
        parse("2*sinh(x)*cosh(x)-sinh(2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::reject_linear_hyperbolic_combination_before_zero_scope(&mut ctx, expr));
}

#[test]
fn linear_hyperbolic_zero_scope_reject_matches_symbolic_scale_mismatch() {
    let mut ctx = Context::new();
    let expr =
        parse("x*cosh(2*x+1) - sinh(2*x+1)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::reject_linear_hyperbolic_combination_before_zero_scope(&mut ctx, expr));
}

#[test]
fn linear_hyperbolic_zero_scope_reject_matches_expanded_scaled_mismatch() {
    let mut ctx = Context::new();
    let expr = parse("2*x*cosh(2*x+1) + 3*cosh(2*x+1) - sinh(2*x+1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::reject_linear_hyperbolic_combination_before_zero_scope(&mut ctx, expr));
}

#[test]
fn linear_hyperbolic_zero_scope_reject_preserves_same_family_numeric_cancellation() {
    let mut ctx = Context::new();
    let expr = parse("2*cosh(x) - cosh(x) - cosh(x)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::reject_linear_hyperbolic_combination_before_zero_scope(&mut ctx, expr));
}

#[test]
fn maybe_two_term_tanh_exp_equivalence_candidate_accepts_tanh_ratio_partner() {
    let mut ctx = Context::new();
    let lhs_core = parse("tanh(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core = parse("(e^x - e^(-x))/(e^x + e^(-x))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::maybe_two_term_tanh_exp_equivalence_candidate(
        &mut ctx, lhs_core, rhs_core
    ));
}

#[test]
fn maybe_two_term_tanh_exp_equivalence_candidate_rejects_missing_exp_partner() {
    let mut ctx = Context::new();
    let lhs_core = parse("tanh(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs_core =
        parse("sinh(x) + cosh(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_two_term_tanh_exp_equivalence_candidate(
        &mut ctx, lhs_core, rhs_core
    ));
}

#[test]
fn maybe_shifted_quotient_exact_zero_direct_residual_candidate_rejects_plain_multiplicative_pair() {
    let mut ctx = Context::new();
    let expr = parse("2*x - 3*x", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_shifted_quotient_exact_zero_direct_residual_candidate(&ctx, expr));
}

#[test]
fn maybe_shifted_quotient_exact_zero_direct_residual_candidate_accepts_hyperbolic_quotient_pair() {
    let mut ctx = Context::new();
    let expr = parse("cosh(x) - ((e^x + e^(-x))/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::maybe_shifted_quotient_exact_zero_direct_residual_candidate(&ctx, expr));
}

#[test]
fn additive_scope_contains_zero_term_detects_top_level_zero_regression() {
    let mut ctx = Context::new();
    let expr = parse("(ln(x) - ln(x)) + 0", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::additive_scope_contains_zero_term(&mut ctx, expr));
}

#[test]
fn additive_scope_contains_zero_term_rejects_abs_sqrt_pair_without_zero_regression() {
    let mut ctx = Context::new();
    let expr = parse("sqrt(x + 2*sqrt(x-1)) - abs(1 + sqrt(x-1))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::additive_scope_contains_zero_term(&mut ctx, expr));
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
        super::maybe_shifted_quotient_exact_zero_direct_residual_route_candidate(
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
        super::maybe_shifted_quotient_exact_zero_direct_residual_route_candidate(
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
        !super::maybe_shifted_quotient_exact_zero_direct_residual_route_candidate(
            &mut ctx,
            lhs_core,
            rhs_core,
            residual_expr,
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
fn collapse_exact_zero_additive_subexpression_matches_structural_cos_factor_triad() {
    let mut ctx = Context::new();
    let expr = parse("cos(x) + cos(x)^2 - cos(x)*(1+cos(x))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("expected structural three-term zero rewrite"));
    let zero = ctx.num(0);
    assert_eq!(compare_expr(&ctx, rewrite.new_expr, zero), Ordering::Equal);
}

#[test]
fn collapse_exact_zero_additive_subexpression_matches_special_sine_difference_to_product() {
    let mut ctx = Context::new();
    let expr = parse("sin(3*x) - sin(x) - 2*cos(2*x)*sin(x)", &mut ctx)
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
    assert_empty_or_legacy_description(&rewrite.description, "Expand sine difference to product");
}

#[test]
fn collapse_exact_zero_additive_subexpression_matches_recursive_six_sine() {
    let mut ctx = Context::new();
    let expr = parse("sin(6*x) - (sin(5*x)*cos(x)+cos(5*x)*sin(x))", &mut ctx)
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
    assert_empty_or_legacy_description(&rewrite.description, "Angle Sum/Diff Identity");
}

#[test]
fn fast_recursive_trig_angle_sum_diff_helper_matches_recursive_six_sine() {
    let mut ctx = Context::new();
    let expr = parse("sin(6*x) - (sin(5*x)*cos(x)+cos(5*x)*sin(x))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite =
        super::try_build_fast_recursive_trig_angle_sum_diff_zero_scope_rewrite(&mut ctx, expr)
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
fn collapse_exact_zero_additive_subexpression_matches_sine_sixth_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse(
        "sin(x)^6 - ((10-15*cos(2*x)+6*cos(4*x)-cos(6*x))/32)",
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
    assert_empty_or_legacy_description(&rewrite.description, "Power Reduction Identity");
}

#[test]
fn collapse_exact_zero_additive_subexpression_matches_scaled_sine_fourth_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse("8*sin(x)^4 - (3-4*cos(2*x)+cos(4*x))", &mut ctx)
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
    assert_empty_or_legacy_description(&rewrite.description, "Power Reduction Identity");
}

#[test]
fn collapse_exact_zero_additive_subexpression_matches_cosine_twenty_fourth_power_reduction() {
    let mut ctx = Context::new();
    let expr = parse(
        "cos(x)^24 - ((1352078+2496144*cos(2*x)+1961256*cos(4*x)+1307504*cos(6*x)+735471*cos(8*x)+346104*cos(10*x)+134596*cos(12*x)+42504*cos(14*x)+10626*cos(16*x)+2024*cos(18*x)+276*cos(20*x)+24*cos(22*x)+cos(24*x))/8388608)",
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
    assert_empty_or_legacy_description(&rewrite.description, "Power Reduction Identity");
}

#[test]
fn collapse_exact_zero_additive_subexpression_matches_recursive_six_cosine_with_passthrough() {
    let mut ctx = Context::new();
    let expr = parse(
        "((cos(5*x)*cos(x)-sin(5*x)*sin(x)) + m) - ((cos(6*x)) + m)",
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
    assert_empty_or_legacy_description(&rewrite.description, "Angle Sum/Diff Identity");
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    let rewrite = super::try_build_exact_zero_trig_double_angle_cos_variant_zero_scope_rewrite(
        &mut ctx, expr,
    )
    .unwrap_or_else(|| panic!("rewrite"));
    let zero = ctx.num(0);

    assert_empty_or_legacy_description(&rewrite.description, "Double Angle Expansion");
    assert!(super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.final_expr(),
        zero
    ));
}

#[test]
fn cancel_exact_additive_pairs_rule_matches_trig_and_numeric_pair_chain() {
    let mut ctx = Context::new();
    let expr = parse("2*cos(2*x) + 1 - 2*cos(2*x) - 1", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = super::CancelExactAdditivePairsRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let zero = ctx.num(0);

    assert_empty_or_legacy_description(&rewrite.description, "Cancel exact additive pairs");
    assert!(super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.final_expr(),
        zero
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
fn collapse_exact_zero_common_scaled_difference_preserves_assumed_abs_event() {
    let mut ctx = Context::new();
    let expr = parse("2*a - 2*abs(a)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Assume);

    let event = super::common_scale_abs_like_positive_assumption_event(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("expected common-scale abs assumption event"));
    assert_eq!(event.message, "a > 0");

    assert_eq!(event.expr_display, "a");
}

#[test]
fn direct_trig_double_angle_cos_variant_equivalence_rewrite_matches_scaled_one_minus_two_sin_sq() {
    let mut ctx = Context::new();
    let lhs = parse("2*cos(2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse lhs: {err}"));
    let rhs = parse("2 - 4*sin(x)^2", &mut ctx).unwrap_or_else(|err| panic!("parse rhs: {err}"));

    let rewrite = super::try_build_direct_trig_double_angle_cos_variant_equivalence_rewrite(
        &mut ctx, lhs, rhs,
    )
    .unwrap_or_else(|| panic!("rewrite"));
    let zero = ctx.num(0);

    assert_empty_or_legacy_description(&rewrite.description, "Double Angle Expansion");
    assert!(super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.final_expr(),
        zero
    ));
}

#[test]
fn direct_core_equivalence_rewrite_matches_exact_quarter_phase_shift_pair_sum() {
    let mut ctx = Context::new();
    let lhs = parse("sin(x)+cos(x)+sin(y)+cos(y)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse lhs: {err}"));
    let rhs = parse("sqrt(2)*sin(x+pi/4)+sqrt(2)*sin(y+pi/4)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse rhs: {err}"));

    let rewrite = super::try_build_direct_trig_exact_quarter_phase_shift_pair_equivalence_rewrite(
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
fn collapse_exact_zero_common_scaled_difference_rule_matches_recursive_six_cosine_shared_denominator(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "((cos(5*x)*cos(x)-sin(5*x)*sin(x))/q) - ((cos(6*x))/q)",
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
    assert_empty_or_legacy_description(&rewrite.description, "Angle Sum/Diff Identity");
    assert_eq!(rewrite.required_conditions.len(), 1);
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
        super::try_build_exact_trig_product_to_sum_sin_sin_three_term_zero_rewrite(&mut ctx, expr)
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
fn exact_trig_equivalence_zero_scope_rewrite_matches_raw_sin_sin_residual() {
    let mut ctx = Context::new();
    let expr = parse("2*sin(x)*sin(y) - cos(x-y) + cos(x+y)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_exact_trig_equivalence_zero_scope_rewrite(&mut ctx, expr)
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

    let rewrite = super::try_build_exact_trig_equivalence_zero_scope_rewrite(&mut ctx, expr)
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

    let rewrite = super::try_build_exact_trig_equivalence_zero_scope_rewrite(&mut ctx, expr)
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

    let rewrite = super::try_build_exact_trig_sum_to_product_zero_scope_rewrite(&mut ctx, expr)
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
fn maybe_exact_trig_equivalence_zero_scope_candidate_accepts_pure_trig_residual() {
    let mut ctx = Context::new();
    let expr =
        parse("sin(x)^2 + cos(x)^2 - 1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::maybe_exact_trig_equivalence_zero_scope_candidate(
        &mut ctx, expr
    ));
}

#[test]
fn maybe_exact_trig_equivalence_zero_scope_candidate_rejects_structural_trig_sum_to_product_scope()
{
    let mut ctx = Context::new();
    let expr = parse("sin(x) + sin(y) - 2*sin((x+y)/2)*cos((x-y)/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_exact_trig_equivalence_zero_scope_candidate(
        &mut ctx, expr
    ));
}

#[test]
fn maybe_exact_trig_equivalence_zero_scope_candidate_rejects_two_term_direct_trig_against_product_scope(
) {
    let mut ctx = Context::new();
    let expr = parse("sin(x) - 2*sin((x+y)/2)*cos((x-y)/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_exact_trig_equivalence_zero_scope_candidate(
        &mut ctx, expr
    ));
}

#[test]
fn maybe_exact_trig_equivalence_zero_scope_candidate_rejects_mixed_trig_log_exp_fraction_scope() {
    let mut ctx = Context::new();
    let expr = parse(
        "(atanh((u^2 - 1)/(u^2 + 1)) - log(u)) + (log((p*q)^2) - 2*log(p) - 2*log(q)) + (exp(r*log(s)) - s^r) + (sin(a) + sin(b) - 2*sin((a+b)/2)*cos((a-b)/2)) + (2/(t^2 - 1) - 1/(t-1) + 1/(t+1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_exact_trig_equivalence_zero_scope_candidate(
        &mut ctx, expr
    ));
}

#[test]
fn maybe_exact_trig_equivalence_zero_scope_candidate_rejects_reciprocal_trig_mixed_scope() {
    let mut ctx = Context::new();
    let expr = parse(
        "(tan(x)*cot(x) - 1) + (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_exact_trig_equivalence_zero_scope_candidate(
        &mut ctx, expr
    ));
}

#[test]
fn reject_scaled_surface_trig_power_vs_numeric_atom_before_default_simplify_matches_symbolic_power()
{
    let mut ctx = Context::new();
    let lhs = parse("4*cos(x)^2", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs = parse("1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::reject_scaled_surface_trig_power_vs_numeric_atom_before_default_simplify(
            &mut ctx, lhs, rhs
        ),
        Some(false)
    );
}

#[test]
fn reject_scaled_surface_trig_power_vs_numeric_atom_before_default_simplify_preserves_special_angle(
) {
    let mut ctx = Context::new();
    let lhs = parse("cos(pi)^2", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs = parse("1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::reject_scaled_surface_trig_power_vs_numeric_atom_before_default_simplify(
            &mut ctx, lhs, rhs
        ),
        None
    );
}

#[test]
fn reject_plain_surface_trig_power_gap_before_default_simplify_matches_scaled_gap() {
    let mut ctx = Context::new();
    let lhs = parse("2*sin(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs = parse("3*sin(x)^2", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::reject_plain_surface_trig_power_gap_before_default_simplify(&mut ctx, lhs, rhs),
        Some(false)
    );
}

#[test]
fn maybe_trig_double_angle_cos_variant_zero_scope_candidate_accepts_split_constants() {
    let mut ctx = Context::new();
    let expr = parse("3 - 4*sin(x)^2 - 2*cos(2*x) - 1", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::maybe_trig_double_angle_cos_variant_zero_scope_candidate(&mut ctx, expr));
}

#[test]
fn maybe_trig_double_angle_cos_variant_zero_scope_candidate_rejects_triple_sine_quotient_scope() {
    let mut ctx = Context::new();
    let expr = parse("sin(3*x)/sin(x) - 2*cos(2*x) - 1", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_trig_double_angle_cos_variant_zero_scope_candidate(&mut ctx, expr));
}

#[test]
fn maybe_trig_double_angle_cos_variant_zero_scope_candidate_rejects_mismatched_numeric_offset() {
    let mut ctx = Context::new();
    let expr =
        parse("3 - 4*sin(x)^2 - 2*cos(2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_trig_double_angle_cos_variant_zero_scope_candidate(&mut ctx, expr));
}

#[test]
fn maybe_trig_embedded_double_angle_factor_zero_scope_candidate_rejects_mixed_double_angle_core() {
    let mut ctx = Context::new();
    let expr = parse("2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x)-2*sin(x))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_trig_embedded_double_angle_factor_zero_scope_candidate(&mut ctx, expr));
}

#[test]
fn maybe_trig_embedded_double_angle_factor_zero_scope_candidate_rejects_triple_sine_quotient_scope()
{
    let mut ctx = Context::new();
    let expr = parse("sin(3*x)/sin(x) - 2*cos(2*x) - 1", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_trig_embedded_double_angle_factor_zero_scope_candidate(&mut ctx, expr));
}

#[test]
fn maybe_trig_embedded_double_angle_factor_zero_scope_candidate_rejects_numeric_offset_scope() {
    let mut ctx = Context::new();
    let expr =
        parse("3 - 4*sin(x)^2 - 2*cos(2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_trig_embedded_double_angle_factor_zero_scope_candidate(&mut ctx, expr));
}

#[test]
fn maybe_trig_sum_to_product_zero_candidate_rejects_mixed_trig_log_exp_fraction_scope() {
    let mut ctx = Context::new();
    let expr = parse(
        "(atanh((u^2 - 1)/(u^2 + 1)) - log(u)) + (log((p*q)^2) - 2*log(p) - 2*log(q)) + (exp(r*log(s)) - s^r) + (sin(a) + sin(b) - 2*sin((a+b)/2)*cos((a-b)/2)) + (2/(t^2 - 1) - 1/(t-1) + 1/(t+1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_trig_sum_to_product_zero_candidate(&ctx, expr));
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

    let rewrite = super::try_build_exact_hyperbolic_equivalence_zero_scope_rewrite(&mut ctx, expr)
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
        super::try_build_exact_hyperbolic_equivalence_zero_scope_rewrite(&mut ctx, expr).is_none()
    );
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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
fn direct_trig_sine_product_cubic_equivalence_matcher_matches() {
    let mut ctx = Context::new();
    let lhs = parse("2*sin(2*x)*sin(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs = parse("4*cos(x)-4*cos(x)^3", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite =
        crate::rules::arithmetic::try_build_direct_trig_sine_product_cubic_equivalence_rewrite(
            &mut ctx, lhs, rhs,
        )
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Product-to-Sum and Triple-Angle Identity",
    );
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

    let rewrite = super::try_build_exact_zero_trig_cos_double_angle_polynomial_zero_scope_rewrite(
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

    let rewrite = super::try_build_exact_zero_trig_embedded_double_angle_factor_zero_scope_rewrite(
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
        super::try_build_exact_zero_trig_embedded_double_angle_factor_zero_scope_rewrite(
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

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
fn small_direct_zero_core_rewrite_matches_signed_common_factor_core() {
    let mut ctx = Context::new();
    let expr = parse("2*cos(x) - x^2*cos(x) - cos(x)*(2-x^2)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
fn small_direct_zero_core_rewrite_matches_telescoping_fraction_core() {
    let mut ctx = Context::new();
    let expr =
        parse("1/(u*(u+1)) - 1/u + 1/(u+1)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
fn direct_fraction_telescoping_zero_scope_rewrite_rejects_mixed_nonfraction_three_term_shape() {
    let mut ctx = Context::new();
    let expr =
        parse("sin(x) - 1/u + 1/(u+1)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        super::try_build_direct_fraction_telescoping_zero_scope_rewrite(&mut ctx, expr).is_none()
    );
}

#[test]
fn small_direct_zero_core_rewrite_matches_reciprocal_nested_fraction_core() {
    let mut ctx = Context::new();
    let expr = parse("(1/x + 1/y)/(1/x - 1/y) - (x+y)/(y-x)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 4);
}

#[test]
fn small_direct_zero_core_rewrite_matches_tan_cot_sec_csc_core() {
    let mut ctx = Context::new();
    let expr = parse("tan(x) + cot(x) - sec(x)*csc(x)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 2);
}

#[test]
fn small_direct_zero_core_rewrite_matches_tan_cot_product_core() {
    let mut ctx = Context::new();
    let expr = parse("tan(x)*cot(x) - 1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 2);
}

#[test]
fn small_direct_zero_core_rewrite_matches_sec_tan_pythagorean_core() {
    let mut ctx = Context::new();
    let expr =
        parse("sec(x)^2 - tan(x)^2 - 1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
fn small_direct_zero_core_rewrite_matches_symmetric_partial_fraction_core() {
    let mut ctx = Context::new();
    let expr = parse("1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 3);
}

#[test]
fn small_direct_zero_core_rewrite_matches_rationalized_sum_of_sqrts_core() {
    let mut ctx = Context::new();
    let expr = parse(
        "1/(sqrt(a) + sqrt(b)) - (sqrt(a) - sqrt(b))/(a - b)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 4);
}

#[test]
fn small_direct_zero_core_rewrite_matches_gap_two_factorial_ratio_core() {
    let mut ctx = Context::new();
    let expr =
        parse("(n+1)!/(n-1)! - n*(n+1)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 1);
}

#[test]
fn small_direct_zero_core_rewrite_matches_odd_half_power_core() {
    let mut ctx = Context::new();
    let expr =
        parse("sqrt(x^5) - x^2*sqrt(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 1);
}

#[test]
fn small_direct_zero_core_rewrite_matches_symbolic_root_denesting_core() {
    let mut ctx = Context::new();
    let expr = parse(
        "sqrt(x + sqrt(x^2 - y^2)) - (sqrt(x+y) + sqrt(x-y))/sqrt(2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 2);
}

#[test]
fn direct_small_zero_additive_combination_rewrite_matches_trig_and_reciprocal_nested_fraction_sum()
{
    let mut ctx = Context::new();
    let expr = parse(
        "(2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + ((1/x + 1/y)/(1/x - 1/y) - (x+y)/(y-x))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 4);
}

#[test]
fn direct_small_zero_additive_combination_rewrite_matches_trig_and_tan_cot_sec_csc_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + (tan(x) + cot(x) - sec(x)*csc(x))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 2);
}

#[test]
fn direct_small_zero_additive_combination_rewrite_matches_trig_and_tan_cot_product_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + (tan(x)*cot(x) - 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 2);
}

#[test]
fn direct_small_zero_additive_combination_rewrite_matches_trig_and_sec_tan_pythagorean_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + (sec(x)^2 - tan(x)^2 - 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
fn direct_small_zero_additive_combination_rewrite_matches_trig_and_gap_two_factorial_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + ((n+1)!/(n-1)! - n*(n+1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 1);
}

#[test]
fn direct_small_zero_additive_combination_rewrite_matches_trig_and_symmetric_partial_fraction_sum()
{
    let mut ctx = Context::new();
    let expr = parse(
        "(2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + (1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 3);
}

#[test]
fn direct_small_zero_additive_combination_rewrite_matches_trig_and_rationalized_sum_of_sqrts_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + (1/(sqrt(a) + sqrt(b)) - (sqrt(a) - sqrt(b))/(a - b))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 4);
}

#[test]
fn direct_small_zero_additive_combination_rewrite_matches_trig_and_odd_half_power_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + (sqrt(x^5) - x^2*sqrt(x))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
    assert_eq!(rewrite.required_conditions.len(), 1);
}

#[test]
fn direct_small_zero_additive_combination_rewrite_matches_trig_and_telescoping_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x)-2*sin(x))) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
fn direct_small_zero_additive_combination_rewrite_flattens_signed_nested_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "2*cos(x) + 2*x*sin(x) - (2*x*sin(x)+(2-x^2)*cos(x)) - x^2*cos(x)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
fn direct_small_zero_additive_combination_rewrite_rejects_non_partition_core_pair() {
    let mut ctx = Context::new();
    let expr = parse("(2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + z", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr).is_none()
    );
}

#[test]
fn maybe_direct_small_zero_additive_combination_candidate_accepts_trig_and_telescoping_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x)-2*sin(x))) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}

#[test]
fn maybe_direct_small_zero_additive_combination_candidate_accepts_pure_trig_partition_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + (tan(x)*cot(x) - 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}

#[test]
fn maybe_direct_small_zero_additive_combination_candidate_accepts_solve_prep_and_nested_fraction_sum(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "(x^2 + 2*b*x - ((x+b)^2 - b^2)) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}

#[test]
fn direct_small_zero_additive_combination_rewrite_matches_solve_prep_and_nested_fraction_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(x^2 + 2*b*x - ((x+b)^2 - b^2)) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
fn maybe_direct_small_zero_additive_combination_candidate_accepts_integrate_prep_and_nested_fraction_sum(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "(cos(x)*cos(2*x)*cos(4*x) - sin(8*x)/(8*sin(x))) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}

#[test]
fn direct_small_zero_additive_combination_rewrite_matches_integrate_prep_and_nested_fraction_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(cos(x)*cos(2*x)*cos(4*x) - sin(8*x)/(8*sin(x))) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
    assert!(!rewrite.required_conditions.is_empty());
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

    assert!(super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}

#[test]
fn direct_small_zero_additive_combination_rewrite_matches_log_square_and_hyperbolic_cubic_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(ln((x*y)^2) - ln(x^2) - ln(y^2)) + (2*sinh(2*x)*sinh(x) - (4*cosh(x)^3 - 4*cosh(x)))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
fn small_direct_zero_core_rewrite_matches_dirichlet_raw_difference() {
    let mut ctx = Context::new();
    let expr = parse(
        "sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
    assert!(!rewrite.required_conditions.is_empty());
}

#[test]
fn integrate_prep_candidate_accepts_dirichlet_raw_difference() {
    let mut ctx = Context::new();
    let expr = parse(
        "sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::maybe_integrate_prep_exact_additive_candidate(
        &mut ctx, expr
    ));
}

#[test]
fn small_structural_poly_zero_core_rewrite_matches_factor_difference_squares() {
    let mut ctx = Context::new();
    let expr =
        parse("p^2-q^2 - (p-q)*(p+q)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_small_structural_poly_zero_core_rewrite(&mut ctx, expr)
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
fn small_structural_poly_zero_core_rewrite_matches_collect_common_factor() {
    let mut ctx = Context::new();
    let expr = parse("u*v + u*w - u*(v+w)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_small_structural_poly_zero_core_rewrite(&mut ctx, expr)
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
fn small_zero_combination_supported_core_accepts_dirichlet_and_factor_cores() {
    let mut ctx = Context::new();
    let dirichlet = parse(
        "sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));
    let factor =
        parse("p^2-q^2 - (p-q)*(p+q)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::small_zero_additive_combination_supported_partition_core(&mut ctx, dirichlet));
    assert!(super::small_zero_additive_combination_supported_partition_core(&mut ctx, factor));
}

#[test]
fn maybe_direct_small_zero_additive_combination_candidate_accepts_dirichlet_and_factor_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))) + (p^2-q^2 - (p-q)*(p+q))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::small_zero_additive_combination_max_terms(&ctx, expr),
        7
    );
    let terms = cas_math::expr_nary::AddView::from_expr(&ctx, expr).terms;
    assert_eq!(terms.len(), 7);
    let first_terms: Vec<_> = terms.iter().copied().take(4).collect();
    let second_terms: Vec<_> = terms.iter().copied().skip(4).collect();
    let first_expr = super::build_small_zero_partition_expr(&mut ctx, &first_terms);
    let second_expr = super::build_small_zero_partition_expr(&mut ctx, &second_terms);
    assert!(super::maybe_integrate_prep_exact_additive_candidate(
        &mut ctx, first_expr
    ));
    assert!(super::small_zero_additive_combination_supported_partition_core(&mut ctx, first_expr));
    assert!(super::small_zero_additive_combination_supported_partition_core(&mut ctx, second_expr));
    assert!(super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}

#[test]
fn direct_small_zero_additive_combination_rewrite_matches_dirichlet_and_factor_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))) + (p^2-q^2 - (p-q)*(p+q))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
    assert!(!rewrite.required_conditions.is_empty());
}

#[test]
fn direct_small_zero_additive_combination_rewrite_matches_dirichlet_and_collect_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))) + (u*v + u*w - u*(v+w))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
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
    assert!(!rewrite.required_conditions.is_empty());
}

#[test]
fn maybe_direct_small_zero_additive_combination_candidate_rejects_nontrig_polynomial_scope() {
    let mut ctx = Context::new();
    let expr = parse("(a+b-c) + (d+e-f)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}

#[test]
fn maybe_direct_small_zero_additive_combination_candidate_rejects_mixed_log_exp_fraction_scope() {
    let mut ctx = Context::new();
    let expr = parse("x^y - exp(y*log(x)) - 1/(x+1) - 2/(x^2 - 1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}

#[test]
fn maybe_direct_small_zero_additive_combination_candidate_rejects_root_log_exp_fraction_scope() {
    let mut ctx = Context::new();
    let expr = parse(
        "sqrt(2 * sqrt(x - 1) + x) + exp(y*log(x)) + 1/(x + 1) + 2/(x^2 - 1) - x^y - 1/(x - 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
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
fn collapse_exact_zero_additive_subexpression_skips_linear_trig_by_parts_residual() {
    let mut ctx = Context::new();
    let expr = parse("cos(2*x+1)*(2*x+3) - sin(2*x+1)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = CollapseExactZeroThreeTermSubsetRule;

    assert!(
        rule.apply(&mut ctx, expr, &parent_ctx).is_none(),
        "linear trig by-parts presentation residual is not an exact-zero identity"
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
        super::try_build_direct_safe_hyperbolic_core_equivalence_rewrite(&mut ctx, lhs, rhs)
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
        super::try_rewrite_hyperbolic_product_sum_sinh_cubic_polynomial_for_cancellation(
            &mut ctx, expr,
        )
        .unwrap_or_else(|| panic!("rewrite"));

    assert!(super::exprs_match_after_default_simplify(
        &mut ctx, rewritten, rhs
    ));
}

#[test]
fn direct_safe_hyperbolic_core_equivalence_rewrite_matches_sinh_cubic_polynomial_pair() {
    let mut ctx = Context::new();
    let lhs = parse("2*sinh(2*x)*cosh(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("4*sinh(x) + 4*sinh(x)^3", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::try_build_direct_safe_hyperbolic_core_equivalence_rewrite(&mut ctx, lhs, rhs)
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
        super::try_build_direct_safe_hyperbolic_core_equivalence_rewrite(&mut ctx, lhs, rhs)
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
        super::try_build_direct_safe_hyperbolic_core_equivalence_rewrite(&mut ctx, lhs, rhs)
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
        super::try_build_direct_safe_hyperbolic_core_equivalence_rewrite(&mut ctx, lhs, rhs)
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
        super::try_build_direct_safe_hyperbolic_core_equivalence_rewrite(&mut ctx, lhs, rhs);

    assert!(rewrite.is_none());
}

#[test]
fn direct_safe_hyperbolic_core_equivalence_rewrite_rejects_atanh_without_log_target() {
    let mut ctx = Context::new();
    let lhs =
        parse("atanh((x^2 - 1)/(x^2 + 1))", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sin(x) + cos(x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::try_build_direct_safe_hyperbolic_core_equivalence_rewrite(&mut ctx, lhs, rhs);

    assert!(rewrite.is_none());
}

#[test]
fn atanh_common_log_definition_mismatch_pair_detector_matches_scaled_forms() {
    let mut ctx = Context::new();
    let lhs =
        parse("1/2*log10((1 + x)/(1 - x))", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("atanh(x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    assert!(super::is_atanh_common_log_definition_mismatch_pair(
        &ctx, lhs, rhs
    ));
}

#[test]
fn exact_zero_identity_rewrite_rejects_atanh_common_log_pair() {
    let mut ctx = Context::new();
    let expr = parse("1/2*log10((1 + x)/(1 - x)) - atanh(x)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr);

    assert!(rewrite.is_none());
}

#[test]
fn exact_zero_identity_rewrite_matches_atanh_square_ratio_ln_pair() {
    let mut ctx = Context::new();
    let expr = parse("atanh((x^2 - 1)/(x^2 + 1)) - ln(x)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(rewrite.final_expr(), ctx.num(0));
    assert_empty_or_legacy_description(&rewrite.description, "Inverse Hyperbolic Log Definition");
}

#[test]
fn exact_zero_identity_rewrite_matches_symbolic_root_denesting_pair() {
    let mut ctx = Context::new();
    let expr = parse(
        "sqrt(x + sqrt(x^2 - y^2)) - (sqrt(x+y) + sqrt(x-y))/sqrt(2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
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
fn exact_zero_product_factor_rule_rejects_atanh_common_log_pair_factor() {
    let mut ctx = Context::new();
    let expr = parse("1/2*(log10((1 + x)/(1 - x)) - 2*atanh(x))", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));
    let parent_ctx = ParentContext::root().with_domain_mode(crate::DomainMode::Generic);

    let rewrite = super::try_build_exact_zero_product_factor_rewrite(&mut ctx, expr, &parent_ctx);

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

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Recognize Hyperbolic from Exponential",
    );
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn direct_core_equivalence_rewrite_matches_shifted_hyperbolic_pythagorean_pair() {
    let mut ctx = Context::new();
    let lhs = parse("cosh(x)^2 - 1", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sinh(x)^2", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Hyperbolic Pythagorean Identity");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn direct_core_equivalence_rewrite_matches_trig_binomial_square_diff_pair() {
    let mut ctx = Context::new();
    let lhs = parse("(sin(x)-cos(x))^2", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("1-sin(2*x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Trig Square Identity");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn direct_core_equivalence_rewrite_matches_log_power_pair_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("ln((x+1)^2)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("2*ln(abs(x+1))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
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

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
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

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
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

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
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
        super::reject_negated_log_pair_without_reciprocal_shape_before_default_simplify(
            &mut ctx, lhs, rhs
        ),
        Some(false)
    );

    let lhs = parse("-2*ln(abs(x))", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("2*ln(abs(x*y))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    assert_eq!(
        super::reject_negated_log_pair_without_reciprocal_shape_before_default_simplify(
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
        super::reject_negated_log_pair_without_reciprocal_shape_before_default_simplify(
            &mut ctx, lhs, rhs
        ),
        None
    );

    let lhs = parse("-ln(x^2)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("ln(x^(-2))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    assert_eq!(
        super::reject_negated_log_pair_without_reciprocal_shape_before_default_simplify(
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

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}

#[test]
fn direct_core_equivalence_rewrite_matches_log_chain_product_pair() {
    let mut ctx = Context::new();
    let lhs = parse("log(b,a)*log(a,c)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("log(b,c)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Log Chain Identity");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn direct_core_equivalence_rewrite_matches_grouped_symbolic_scale_sum_distribution_pair() {
    let mut ctx = Context::new();
    let lhs = parse("a*x^2 + c*x^2 + e*x^2 + b*x + d*x", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs =
        parse("x*(b + d) + x^2*(a + c + e)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn direct_core_equivalence_rewrite_matches_reordered_additive_noncall_pair() {
    let mut ctx = Context::new();
    let lhs = parse("x + y + z", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("z + x + y", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn direct_core_equivalence_rewrite_rejects_plain_cross_trig_pair_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("sin(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("-(cos(y)*sqrt(3))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}

#[test]
fn direct_core_equivalence_rewrite_rejects_noncall_vs_surface_symbolic_trig_before_default_simplify(
) {
    let mut ctx = Context::new();
    let lhs = parse("a", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs =
        parse("-(sin(x + pi/4)*sqrt(2))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}

#[test]
fn direct_core_equivalence_rewrite_rejects_atomic_noncall_pair_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("a", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("-b", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}

#[test]
fn direct_core_equivalence_rewrite_rejects_scaled_symbolic_atom_mismatch_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("x", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("2*x", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}

#[test]
fn direct_core_equivalence_rewrite_keeps_signed_scaled_symbolic_atom_match_before_default_simplify()
{
    let mut ctx = Context::new();
    let lhs = parse("-x", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("-1*x", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
}

#[test]
fn direct_core_equivalence_rewrite_rejects_product_division_shared_scale_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("a * x", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("-(a * d / c)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}

#[test]
fn direct_core_equivalence_rewrite_keeps_cancelable_product_division_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("a * x", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("(a * x * y) / y", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_some());
}

#[test]
fn direct_core_equivalence_rewrite_keeps_symbolic_scale_sum_pair_before_phase_shift_reject_regression(
) {
    let mut ctx = Context::new();
    let lhs = parse("a*x^2 + b*x + c", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("x*(c/x + a*x + b)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_some());
}

#[test]
fn direct_core_equivalence_rewrite_rejects_shifted_surface_trig_symbolic_base_mismatch_before_default_simplify(
) {
    let mut ctx = Context::new();
    let lhs = parse("sin(x + pi/4)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("-sin(y + pi/4)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}

#[test]
fn direct_core_equivalence_rewrite_keeps_atomic_direct_match_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("a", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("a", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_some());
}

#[test]
fn direct_core_equivalence_rewrite_rejects_plain_surface_trig_power_gap_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("cos(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("cos(x)^3", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}

#[test]
fn direct_core_equivalence_rewrite_keeps_exact_surface_trig_power_match_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("cos(pi)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("cos(pi)^3", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_some());
}

#[test]
fn direct_core_equivalence_rewrite_keeps_inverse_trig_surface_match_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("x", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sin(arcsin(x))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_some());
}

#[test]
fn direct_core_equivalence_rewrite_rejects_mismatched_symbolic_hyperbolic_pair_before_default_simplify(
) {
    let mut ctx = Context::new();
    let lhs = parse("cosh(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("cosh(y)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}

#[test]
fn direct_core_equivalence_rewrite_rejects_same_arg_hyperbolic_square_gap_before_default_simplify()
{
    let mut ctx = Context::new();
    let lhs = parse("cosh(x)^2", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sinh(x)^2", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}

#[test]
fn direct_core_equivalence_rewrite_keeps_hyperbolic_pythagorean_pair_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("cosh(x)^2 - 1", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sinh(x)^2", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_some());
}

#[test]
fn direct_core_equivalence_rewrite_rejects_hyperbolic_additive_atomic_tail_before_default_simplify()
{
    let mut ctx = Context::new();
    let lhs = parse("cosh(x)^2 + a", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sinh(x)^2", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}

#[test]
fn direct_core_equivalence_rewrite_keeps_hyperbolic_pythagorean_plus_one_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("sinh(x)^2 + 1", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("cosh(x)^2", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_some());
}

#[test]
fn shifted_quotient_power_merge_residual_rewrite_matches_symbolic_power_pair() {
    let mut ctx = Context::new();
    let lhs = parse("x^a * x^b", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("x^(a+b)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::try_build_shifted_quotient_power_merge_residual_rewrite(&mut ctx, lhs, rhs)
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
        super::try_build_shifted_quotient_power_merge_residual_rewrite(&mut ctx, lhs, rhs)
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
        super::try_build_shifted_quotient_power_merge_residual_rewrite(&mut ctx, lhs, rhs)
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
        super::try_build_shifted_quotient_power_merge_residual_rewrite(&mut ctx, lhs, rhs)
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
        super::try_build_shifted_quotient_power_merge_residual_rewrite(&mut ctx, lhs, rhs)
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
        super::try_build_shifted_quotient_power_merge_residual_rewrite(&mut ctx, lhs, rhs)
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

    let rewrite = super::try_build_shifted_quotient_cancel_common_factors_residual_rewrite(
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

    let rewrite = super::try_build_shifted_quotient_cancel_common_factors_residual_rewrite(
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

    let rewrite = super::try_build_shifted_quotient_cancel_common_factors_residual_rewrite(
        &mut ctx, lhs, rhs,
    );

    assert!(rewrite.is_none());
}

#[test]
fn shifted_quotient_fraction_combine_residual_rewrite_matches_fraction_pair() {
    let mut ctx = Context::new();
    let lhs = parse("a/x + b/y", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("(a*y + b*x)/(x*y)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::try_build_shifted_quotient_fraction_combine_residual_rewrite(&mut ctx, lhs, rhs)
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

    let rewrite =
        super::try_build_shifted_quotient_fraction_combine_residual_rewrite(&mut ctx, lhs, rhs)
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

    let rewrite =
        super::try_build_shifted_quotient_nested_fraction_residual_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Simplify Nested Fraction");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn direct_trig_reciprocal_equivalence_rewrite_matches_sec() {
    let mut ctx = Context::new();
    let lhs = parse("1/cos(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sec(x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_trig_reciprocal_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Reciprocal Quotient Identity");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn direct_trig_ratio_equivalence_rewrite_matches_tan() {
    let mut ctx = Context::new();
    let lhs = parse("sin(2*x)/cos(x+x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("tan(2*x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_trig_ratio_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Trigonometric Quotient Identity");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn direct_trig_ratio_equivalence_rewrite_matches_cot() {
    let mut ctx = Context::new();
    let lhs = parse("cos(x)/sin(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("cot(x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_trig_ratio_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Trigonometric Quotient Identity");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn direct_trig_ratio_equivalence_rewrite_matches_half_angle_tan() {
    let mut ctx = Context::new();
    let lhs = parse("(1-cos(2*x))/sin(2*x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("tan(x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_trig_ratio_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn direct_reciprocal_half_power_shared_denominator_rewrite_matches_root_form() {
    let mut ctx = Context::new();
    let lhs =
        parse("tan(x)^(-1/2)/(2*cos(x)^2)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs =
        parse("1/(2*cos(x)^2*sqrt(tan(x)))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_reciprocal_half_power_shared_denominator_rewrite(
        &mut ctx, lhs, rhs,
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Reciprocal Half-Power Cancellation");
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
fn direct_reciprocal_half_power_product_rewrite_matches_expanded_factor() {
    let mut ctx = Context::new();
    let lhs = parse("asinh(2*x+1)^(-1/2)*((2*x+1)^2+1)^(-1/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("(asinh(2*x+1)*(4*x^2+4*x+2))^(-1/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_reciprocal_half_power_product_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
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
fn direct_reciprocal_half_power_product_rewrite_matches_sqrt_denominator_form() {
    let mut ctx = Context::new();
    let lhs = parse("asinh(2*x+1)^(-1/2)*((2*x+1)^2+1)^(-1/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("1/(sqrt(asinh(2*x+1))*sqrt((2*x+1)^2+1))", &mut ctx)
        .unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_reciprocal_half_power_product_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
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
fn direct_reciprocal_half_power_product_rewrite_matches_sqrt_half_power_base_mix() {
    let mut ctx = Context::new();
    let lhs =
        parse("1/(sqrt(x)*sqrt(sqrt(x)-x))", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("(x*(x^(1/2)-x))^(-1/2)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_reciprocal_half_power_product_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
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
fn direct_reciprocal_half_power_product_rewrite_matches_scaled_sqrt_denominator_form() {
    let mut ctx = Context::new();
    let lhs = parse("(2*asinh(2*x+1)^(-1/2)/sqrt((2*x+1)^2+1))/2", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("1/(sqrt(asinh(2*x+1))*sqrt((2*x+1)^2+1))", &mut ctx)
        .unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_reciprocal_half_power_product_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
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
fn direct_negative_even_root_power_reciprocal_rewrite_matches_scaled_polynomial() {
    let mut ctx = Context::new();
    let lhs =
        parse("(2*x^2+2*x-3)^(-3/2)*(4*x+2)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs =
        parse("(4*x+2)/(2*x^2+2*x-3)^(3/2)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::try_build_direct_negative_even_root_power_reciprocal_rewrite(&mut ctx, lhs, rhs)
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
fn direct_negative_even_root_power_reciprocal_rewrite_matches_sqrt_power_denominator() {
    let mut ctx = Context::new();
    let lhs = parse("cos(x)*sin(x)^(-3/2)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("cos(x)/sqrt(sin(x)^3)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::try_build_direct_negative_even_root_power_reciprocal_rewrite(&mut ctx, lhs, rhs)
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
fn direct_negative_even_root_power_reciprocal_rewrite_matches_shifted_trig_ratio_cofactor() {
    let mut ctx = Context::new();
    let lhs = parse("tan(x)*cos(x)^(-1/2)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sin(x)/sqrt(cos(x)^3)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::try_build_direct_negative_even_root_power_reciprocal_rewrite(&mut ctx, lhs, rhs)
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
fn exact_zero_identity_rewrite_matches_negative_even_root_power_reciprocal_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "(2*x^2+2*x-3)^(-3/2)*(4*x+2) - (4*x+2)/(2*x^2+2*x-3)^(3/2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
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
}

#[test]
fn exact_zero_identity_rewrite_matches_negative_even_root_power_reciprocal_product_cofactor() {
    let mut ctx = Context::new();
    let expr = parse(
        "(y+1)*(2*x^2+2*x-3)^(-3/2)*(4*x+2) - ((y+1)*(4*x+2))/(2*x^2+2*x-3)^(3/2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
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
}

#[test]
fn exact_zero_identity_rewrite_matches_reciprocal_half_power_product_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "asinh(2*x+1)^(-1/2)*((2*x+1)^2+1)^(-1/2) - (asinh(2*x+1)*(4*x^2+4*x+2))^(-1/2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
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
}

#[test]
fn exact_zero_identity_rewrite_matches_scaled_reciprocal_half_power_product_residual() {
    let mut ctx = Context::new();
    let expr = parse("(a/(2*sqrt(x)*sqrt(1-x)))*2 - a*(x*(1-x))^(-1/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
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
}

#[test]
fn exact_zero_identity_rewrite_matches_reciprocal_half_power_quotient_product_one_residual() {
    let mut ctx = Context::new();
    let expr = parse("1 - x^(-1/2)*((x-1)/x)^(-1/2)*(x-1)^(1/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Quotient Product Cancellation",
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
}

#[test]
fn exact_zero_identity_rewrite_matches_reciprocal_half_power_quotient_over_base_residual() {
    let mut ctx = Context::new();
    let expr = parse("(x-4)^(1/2)/(x*(x-4)) - x^(-3/2)*(1-4/x)^(-1/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Quotient/Base Cancellation",
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
}

#[test]
fn exact_zero_identity_rewrite_matches_reciprocal_half_power_linear_residual() {
    let mut ctx = Context::new();
    let expr = parse("x^(-1/2) - (2*x-1)/(2*x^(3/2)) - 1/2*x^(-3/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));
    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Linear Residual Cancellation",
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
}

#[test]
fn exact_zero_identity_rewrite_matches_common_sqrt_denominator_fraction_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "-(3*x+1)/(2*sqrt(x)*(x*(x+1)^2+1)) + \
         (3*x+1)/(2*x^(1/2)+2*x^(3/2)*(x+1)^2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Sqrt Denominator Factor Cancellation",
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
}

#[test]
fn exact_zero_identity_rewrite_matches_scaled_common_sqrt_denominator_fraction_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "(-1/2)*((3*x+1)/(2*sqrt(x)*(x*(x+1)^2+1))) + \
         (3*x+1)/(4*x^(1/2)+4*x^(3/2)*(x+1)^2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Sqrt Denominator Factor Cancellation",
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
}

#[test]
fn exact_zero_identity_rewrite_matches_sqrt_over_base_fraction_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "(x^(-1/2) - 3*x^(3/2))/(2*(x^2+1)^2) - \
         (x^(1/2)-3*x^(5/2))/(2*x*(x^2+1)^2)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Sqrt/Base Fraction Cancellation");
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
fn exact_zero_identity_rewrite_matches_rationalized_common_sqrt_denominator_fraction_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "-(3*x+1)/((x*(x+1)^2+1)*sqrt(x)) + \
         (3*x+1)*(x^(1/2)-x^(3/2)*(x+1)^2)/(x-x^3*(x+1)^4)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Rationalized Sqrt Denominator Cancellation",
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
}

#[test]
fn exact_zero_identity_rewrite_matches_scaled_reciprocal_half_power_over_base_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "(x*(log2(x^2+1)+1)^(-1/2))/(ln(2)*(x^2+1)) - \
         (x*(log2(x^2+1)+1)^(1/2))/(ln(2)*(log2(x^2+1)+1)*(x^2+1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Reciprocal Half-Power Cancellation");
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
fn common_scaled_difference_rule_matches_scaled_reciprocal_half_power_over_base_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "(x*(log2(x^2+1)+1)^(-1/2))/(ln(2)*(x^2+1)) - \
         (x*(log2(x^2+1)+1)^(1/2))/(ln(2)*(log2(x^2+1)+1)*(x^2+1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_common_scaled_difference_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Reciprocal Half-Power Cancellation");
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
fn common_scaled_difference_rule_matches_scaled_reciprocal_half_power_over_base_residual_with_unfactored_common_denominators(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "((log2(x^2+1)+1)^(-1/2)*x*2)/(ln(2)*(2*q*x^2+2*q)) - \
         (x*(log2(x^2+1)+1)^(1/2))/(ln(2)*(x^2+1)*(q*log2(x^2+1)+q))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_common_scaled_difference_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Reciprocal Half-Power Cancellation");
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
fn common_scaled_difference_rule_matches_product_wrapped_scaled_reciprocal_half_power_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "2*(((log2(x^2+1)+1)^(-1/2)*x)/(ln(2)*(2*q*x^2+2*q))) - \
         (x*(log2(x^2+1)+1)^(1/2))/(ln(2)*(x^2+1)*(q*log2(x^2+1)+q))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_common_scaled_difference_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Reciprocal Half-Power Cancellation");
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
fn common_scaled_difference_rule_matches_scaled_reciprocal_half_power_shared_denominator_residual_with_unfactored_common_denominator(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "2*(x/sqrt(log2(x^2+1)+1))/(2*q*ln(2)+2*q*ln(2)*x^2) - \
         x/(q*ln(2)*sqrt(log2(x^2+1)+1)*(x^2+1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_common_scaled_difference_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Reciprocal Half-Power Cancellation");
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
fn collapse_exact_zero_product_factor_matches_scaled_reciprocal_half_power_product_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "1/2*((a/(2*sqrt(x)*sqrt(1-x)))*2 - a*(x*(1-x))^(-1/2))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let rewrite = super::try_build_exact_zero_product_factor_rewrite(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
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
}

#[test]
fn div_zero_rule_matches_scaled_reciprocal_half_power_product_residual_over_constant() {
    let mut ctx = Context::new();
    let expr = parse(
        "((a/(2*sqrt(x)*sqrt(1-x)))*2 - a*(x*(1-x))^(-1/2))/2",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite =
        super::try_build_exact_zero_radical_numerator_const_division_rewrite(&mut ctx, expr)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(
        &rewrite.description,
        "Reciprocal Half-Power Product Cancellation",
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
}

#[test]
fn collapse_exact_zero_same_denominator_rule_matches_fraction_difference_combination_pair() {
    let mut ctx = Context::new();
    let expr = parse(
        "((1/(x+b) - 1/(x+c))/q) - (((c-b)/(x^2+(b+c)*x+b*c))/q)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_same_denominator_rewrite(&mut ctx, expr)
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
fn collapse_exact_zero_same_denominator_rule_matches_trig_reciprocal_core() {
    let mut ctx = Context::new();
    let expr = parse("((1/cos(x))/q) - ((sec(x))/q)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_same_denominator_rewrite(&mut ctx, expr)
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
    assert_empty_or_legacy_description(&rewrite.description, "Reciprocal Quotient Identity");
}

#[test]
fn collapse_exact_zero_same_denominator_rule_matches_scaled_fraction_difference_combination_pair() {
    let mut ctx = Context::new();
    let expr = parse(
        "((1/(2*a)*(1/(x-a) - 1/(x+a)))/q) - ((1/(x^2-a^2))/q)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_same_denominator_rewrite(&mut ctx, expr)
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
fn collapse_exact_zero_same_denominator_rule_matches_trig_ratio_core_tan() {
    let mut ctx = Context::new();
    let expr = parse("((sin(2*x)/cos(x+x))/q) - ((tan(2*x))/q)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_same_denominator_rewrite(&mut ctx, expr)
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
    assert_empty_or_legacy_description(&rewrite.description, "Trigonometric Quotient Identity");
}

#[test]
fn collapse_exact_zero_same_denominator_rule_matches_trig_ratio_core_cot() {
    let mut ctx = Context::new();
    let expr = parse("((cos(x)/sin(x))/q) - ((cot(x))/q)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_same_denominator_rewrite(&mut ctx, expr)
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
    assert_empty_or_legacy_description(&rewrite.description, "Trigonometric Quotient Identity");
}

#[test]
fn small_zero_core_matches_unit_fraction_equivalent_trig_denominators() {
    let mut ctx = Context::new();
    let expr = parse(
        "1/(2*x*cos(x)+(x*x-2)*sin(x)+c) - 1/(2*x*cos(x)+(x^2-2)*sin(x)+c)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
    assert_empty_or_legacy_description(&rewrite.description, "Subtract Fractions");
    assert_eq!(rewrite.required_conditions.len(), 2);
}

#[test]
fn collapse_exact_zero_same_denominator_rule_matches_half_angle_tan_core() {
    let mut ctx = Context::new();
    let expr = parse("(((1-cos(2*x))/sin(x+x))/q) - ((tan(x))/q)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_same_denominator_rewrite(&mut ctx, expr)
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
fn direct_sub_fraction_combination_equivalence_rewrite_matches_symbolic_difference_squares_pair() {
    let mut ctx = Context::new();
    let lhs =
        parse("1/(2*a)*(1/(x-a) - 1/(x+a))", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("1/(x^2-a^2)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::try_build_direct_sub_fraction_combination_equivalence_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Subtract Fractions");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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
    let numerator_core = super::strip_positive_one_passthrough(&mut ctx, numerator)
        .unwrap_or_else(|| panic!("numerator core"));
    let denominator_core = super::strip_positive_one_passthrough(&mut ctx, denominator)
        .unwrap_or_else(|| panic!("denominator core"));

    let rewrite = super::try_build_direct_small_zero_core_shifted_quotient_rewrite(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        expected
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Phase Shift Identity");
}

#[test]
fn shared_passthrough_difference_shape_gate_rejects_additive_vs_single_term_residual() {
    let mut ctx = Context::new();
    let expr = parse("(x + x) - (2*x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::has_plausible_shared_additive_passthrough_difference_shape(&mut ctx, expr));
    assert!(
        super::try_build_exact_zero_shared_passthrough_difference_rewrite(&mut ctx, expr).is_none()
    );
}

#[test]
fn shared_passthrough_difference_shape_gate_keeps_known_passthrough_zero_regression() {
    let mut ctx = Context::new();
    let expr = parse("((sin(x)^2 + cos(x)^2) + m) - ((1) + m)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::has_plausible_shared_additive_passthrough_difference_shape(&mut ctx, expr));
    let rewrite = super::try_build_exact_zero_shared_passthrough_difference_rewrite(&mut ctx, expr)
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
fn shared_passthrough_square_base_equivalence_keeps_quartic_conditional_factor_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "(((a*x^4 + b*x^3 + c*x^2 + d)^2) + m) - (((x^2*(a*x^2 + b*x + c + d/x^2))^2) + m)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::has_plausible_shared_additive_passthrough_difference_shape(&mut ctx, expr));
    let rewrite = super::try_build_exact_zero_shared_passthrough_difference_rewrite(&mut ctx, expr)
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
fn collapse_exact_zero_additive_subexpression_matches_quartic_conditional_factor_square_difference()
{
    let mut ctx = Context::new();
    let expr = parse(
        "(a*x^4 + b*x^3 + c*x^2 + d)^2 - (x^2*(a*x^2 + b*x + c + d/x^2))^2",
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
fn collapse_exact_zero_additive_subexpression_matches_solve_prep_square_difference() {
    let mut ctx = Context::new();
    let expr = parse(
        "(a*x^2 + b*x + c)^2 - (a*(x + b/(2*a))^2 + c - b^2/(4*a))^2",
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
fn collapse_exact_zero_additive_subexpression_matches_solve_prep_squared_passthrough() {
    let mut ctx = Context::new();
    let expr = parse(
        "(((a*x^2 + b*x + c)^2) + m) - (((a*(x + b/(2*a))^2 + c - b^2/(4*a))^2) + m)",
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

#[test]
fn collapse_exact_zero_additive_subexpression_matches_telescoping_fraction_double_squared_passthrough(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "((((1/(x+b) - 1/(x+c))^2) + m)^2) - (((((c-b)/(x^2+(b+c)*x+b*c))^2) + m)^2)",
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
    assert_empty_or_legacy_description(&rewrite.description, "Subtract Fractions");
}

#[test]
fn collapse_exact_zero_additive_subexpression_matches_morrie_double_squared_passthrough() {
    let mut ctx = Context::new();
    let expr = parse(
        "((((cos(x)*cos(2*x)*cos(4*x))^2) + m)^2) - ((((sin(8*x)/(8*sin(x)))^2) + m)^2)",
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
    assert_empty_or_legacy_description(&rewrite.description, "Apply Morrie's law");
    assert!(!rewrite.required_conditions.is_empty());
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
}

#[test]
fn collapse_exact_zero_additive_subexpression_matches_dirichlet_passthrough() {
    let mut ctx = Context::new();
    let expr = parse(
        "((1 + 2*cos(x) + 2*cos(2*x)) + m) - ((sin(5*x/2)/sin(x/2)) + m)",
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
    assert_empty_or_legacy_description(&rewrite.description, "Dirichlet Kernel Identity");
}

#[test]
fn collapse_exact_zero_additive_subexpression_matches_reverse_dirichlet_raw_difference() {
    let mut ctx = Context::new();
    let expr = parse(
        "sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))",
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
    assert_empty_or_legacy_description(&rewrite.description, "Dirichlet Kernel Identity");
}

#[test]
fn collapse_exact_zero_common_scaled_difference_rule_matches_dirichlet_scaled_difference() {
    let mut ctx = Context::new();
    let expr = parse(
        "k*(1 + 2*cos(x) + 2*cos(2*x)) - k*(sin(5*x/2)/sin(x/2))",
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
    assert!(
        rewrite.description.contains("Dirichlet Kernel Identity"),
        "unexpected description: {}",
        rewrite.description
    );
}

#[test]
fn collapse_exact_zero_common_scaled_difference_rule_matches_dirichlet_same_denominator() {
    let mut ctx = Context::new();
    let expr = parse(
        "((1 + 2*cos(x) + 2*cos(2*x))/q) - ((sin(5*x/2)/sin(x/2))/q)",
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
    assert!(
        rewrite.description.contains("Dirichlet Kernel Identity"),
        "unexpected description: {}",
        rewrite.description
    );
}

#[test]
fn direct_finite_product_equivalence_rewrite_matches_shifted_factorized_telescoping_product() {
    let mut ctx = Context::new();
    let lhs = parse("product(1 - 1/(k+a)^2, k, m, n)", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("((m+a-1)*(n+a+1))/((m+a)*(n+a))", &mut ctx)
        .unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_finite_product_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Finite Telescoping Product");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn direct_finite_sum_equivalence_rewrite_matches_telescoping_sum() {
    let mut ctx = Context::new();
    let lhs =
        parse("sum(1/(k*(k+1)), k, 1, n)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("1 - 1/(n+1)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_finite_sum_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Finite Telescoping Sum");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn direct_core_equivalence_rewrite_matches_telescoping_sum_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs =
        parse("sum(1/(k*(k+1)), k, 1, n)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("1 - 1/(n+1)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Finite Telescoping Sum");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn direct_finite_sum_equivalence_matches_combined_telescoping_fraction_target() {
    let mut ctx = Context::new();
    let lhs = parse("sum(1/((a*k+b+c)*(a*k+b+c+a)), k, m, n)", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse(
        "(n + 1 - m) / \
         (m*n*a^2 + a*b*m + a*b*n + a*c*m + a*c*n + m*a^2 + b^2 + c^2 + 2*b*c + a*b + a*c)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_finite_sum_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Finite Telescoping Sum");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn direct_finite_sum_equivalence_matches_scaled_combined_telescoping_fraction_target() {
    let mut ctx = Context::new();
    let lhs = parse("k*sum(1/((a*k+b+c)*(a*k+b+c+a)), k, m, n)", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse(
        "(k*n + k - k*m) / \
         (m*n*a^2 + a*b*m + a*b*n + a*c*m + a*c*n + m*a^2 + b^2 + c^2 + 2*b*c + a*b + a*c)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_finite_sum_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Finite Telescoping Sum");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}

#[test]
fn exact_zero_identity_rewrite_matches_combined_telescoping_fraction_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "sum(1/((a*k+b+c)*(a*k+b+c+a)), k, m, n) - \
         ((1/(a*m+b+c) - 1/(a*n+a+b+c))*1)/a",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Finite Telescoping Sum");
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
fn exact_zero_identity_rewrite_matches_scaled_combined_telescoping_fraction_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "k*sum(1/((a*k+b+c)*(a*k+b+c+a)), k, m, n) - \
         (k*n + k - k*m) / \
         (m*n*a^2 + a*b*m + a*b*n + a*c*m + a*c*n + m*a^2 + b^2 + c^2 + 2*b*c + a*b + a*c)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("expr: {err}"));

    let rewrite = super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Finite Telescoping Sum");
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
fn collapse_exact_zero_common_scaled_difference_rule_matches_finite_telescoping_same_denominator() {
    let mut ctx = Context::new();
    let expr = parse(
        "((product(1 - 1/(k+a)^2, k, m, n))/q) - ((((m+a-1)*(n+a+1))/((m+a)*(n+a)))/q)",
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
    assert!(
        rewrite.description.contains("Finite Telescoping Product"),
        "unexpected description: {}",
        rewrite.description
    );
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

    assert!(super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Dirichlet Kernel Identity");
}

#[test]
fn collapse_exact_zero_additive_subexpression_matches_complete_square_with_passthrough() {
    let mut ctx = Context::new();
    let expr = parse(
        "((a*x^2 - b*x + c) + m) - ((a*(x - b/(2*a))^2 + c - b^2/(4*a)) + m)",
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
    assert_empty_or_legacy_description(&rewrite.description, "Complete the Square");
}

#[test]
fn collapse_exact_zero_additive_subexpression_matches_trinomial_square() {
    let mut ctx = Context::new();
    let expr = parse(
        "(a + b + c)^2 - (a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c)",
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
    assert_empty_or_legacy_description(&rewrite.description, "Expand binomial/trinomial power");
}

#[test]
fn small_polynomial_expand_helper_expands_trinomial_square() {
    let mut ctx = Context::new();
    let expr = parse("(a + b + c)^2", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let expanded = cas_math::expansion_rule_support::try_expand_small_pow_sum_expr(
        &mut ctx,
        expr,
        cas_math::expansion_rule_support::SmallPowExpandPolicy {
            max_vars: 3,
            ..cas_math::expansion_rule_support::SmallPowExpandPolicy::default()
        },
    )
    .unwrap_or_else(|| panic!("expanded"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: expanded
            }
        ),
        "a^2 + b^2 + c^2 + 2 * a * b + 2 * a * c + 2 * b * c"
    );
}

#[test]
fn fast_small_polynomial_expansion_zero_scope_rewrite_matches_trinomial_square() {
    let mut ctx = Context::new();
    let expr = parse(
        "(a + b + c)^2 - (a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite =
        super::try_build_fast_small_polynomial_expansion_zero_scope_rewrite(&mut ctx, expr)
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
    assert_empty_or_legacy_description(&rewrite.description, "Expand binomial/trinomial power");
}

#[test]
fn fast_small_polynomial_expansion_zero_scope_rewrite_matches_signed_trinomial_square() {
    let mut ctx = Context::new();
    let expr = parse(
        "(a - b + c)^2 - (a^2 + b^2 + c^2 - 2*a*b + 2*a*c - 2*b*c)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite =
        super::try_build_fast_small_polynomial_expansion_zero_scope_rewrite(&mut ctx, expr)
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
    assert_empty_or_legacy_description(&rewrite.description, "Expand binomial/trinomial power");
}

#[test]
fn small_polynomial_expand_helper_expands_signed_trinomial_square() {
    let mut ctx = Context::new();
    let expr = parse("(a - b + c)^2", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let expanded = cas_math::expansion_rule_support::try_expand_small_pow_sum_expr(
        &mut ctx,
        expr,
        cas_math::expansion_rule_support::SmallPowExpandPolicy {
            max_vars: 3,
            ..cas_math::expansion_rule_support::SmallPowExpandPolicy::default()
        },
    )
    .unwrap_or_else(|| panic!("expanded"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: expanded
            }
        ),
        "a^2 + b^2 + c^2 + -2 * a * b + -2 * b * c + 2 * a * c"
    );
}

#[test]
fn collapse_exact_zero_additive_subexpression_matches_trinomial_square_with_passthrough() {
    let mut ctx = Context::new();
    let expr = parse(
        "((a + b + c)^2 + m) - ((a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c) + m)",
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
    assert_empty_or_legacy_description(&rewrite.description, "Expand binomial/trinomial power");
}

#[test]
fn solve_prep_equivalence_helper_rewrites_fraction_with_common_denominator() {
    let mut ctx = Context::new();
    let expr = parse("(a*x^2 + b*x + c)/q", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rewrite_match =
        super::try_rewrite_exact_solve_prep_equivalence_for_cancellation(&mut ctx, expr)
            .unwrap_or_else(|| panic!("rewrite"));
    let expected = parse("(a*(x + b/(2*a))^2 + c - b^2/(4*a))/q", &mut ctx)
        .unwrap_or_else(|err| panic!("parse expected: {err}"));

    assert!(super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite_match.rewritten,
        expected
    ));
}

#[test]
fn collapse_exact_zero_common_scaled_difference_rule_matches_complete_square() {
    let mut ctx = Context::new();
    let expr = parse(
        "k*(a*x^2 - b*x + c) - k*(a*(x - b/(2*a))^2 + c - b^2/(4*a))",
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
    assert_empty_or_legacy_description(&rewrite.description, "Complete the Square");
}

#[test]
fn collapse_exact_zero_common_scaled_difference_rule_matches_trinomial_square() {
    let mut ctx = Context::new();
    let expr = parse(
        "k*(a + b + c)^2 - k*(a^2 + b^2 + c^2 + 2*a*b + 2*a*c + 2*b*c)",
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
    assert_empty_or_legacy_description(&rewrite.description, "Expand binomial/trinomial power");
}

#[test]
fn collapse_exact_zero_same_denominator_rule_matches_complete_square() {
    let mut ctx = Context::new();
    let expr = parse(
        "((a*x^2 + b*x + c)/q) - ((a*(x + b/(2*a))^2 + c - b^2/(4*a))/q)",
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
    assert_empty_or_legacy_description(&rewrite.description, "Complete the Square");
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
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

    assert!(super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.new_expr,
        one
    ));
    assert_empty_or_legacy_description(&rewrite.description, "Equivalent Residual Cancellation");
}

#[test]
fn solve_prep_exact_additive_candidate_detects_complete_square_pair() {
    let mut ctx = Context::new();
    let expr = parse(
        "a*x^2 - b*x + c - (a*(x - b/(2*a))^2 + c - b^2/(4*a))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::maybe_solve_prep_exact_additive_candidate(&ctx, expr));
}

#[test]
fn solve_prep_focus_remaining_variable_overlap_detects_complete_square_pair() {
    let mut ctx = Context::new();
    let focus_expr =
        parse("a*x^2 - b*x + c", &mut ctx).unwrap_or_else(|err| panic!("focus parse: {err}"));
    let remaining_expr = parse("-(a*(x - b/(2*a))^2 + c - b^2/(4*a))", &mut ctx)
        .unwrap_or_else(|err| panic!("remaining parse: {err}"));

    assert!(
        super::has_plausible_solve_prep_focus_remaining_variable_overlap(
            &ctx,
            focus_expr,
            remaining_expr
        )
    );
}

#[test]
fn solve_prep_focus_remaining_variable_overlap_rejects_mismatched_shifted_square_variable() {
    let mut ctx = Context::new();
    let focus_expr =
        parse("a*x^2 - b*x + c", &mut ctx).unwrap_or_else(|err| panic!("focus parse: {err}"));
    let remaining_expr = parse("-(a*(y - b/(2*a))^2 + c - b^2/(4*a))", &mut ctx)
        .unwrap_or_else(|err| panic!("remaining parse: {err}"));

    assert!(
        !super::has_plausible_solve_prep_focus_remaining_variable_overlap(
            &ctx,
            focus_expr,
            remaining_expr
        )
    );
}

#[test]
fn solve_prep_focus_remaining_variable_overlap_rejects_shift_parameter_only_overlap() {
    let mut ctx = Context::new();
    let focus_expr =
        parse("b^2/(4*a) - b + c", &mut ctx).unwrap_or_else(|err| panic!("focus parse: {err}"));
    let remaining_expr = parse("-(a*(x - b/(2*a))^2 + c - b^2/(4*a))", &mut ctx)
        .unwrap_or_else(|err| panic!("remaining parse: {err}"));

    assert!(
        !super::has_plausible_solve_prep_focus_remaining_variable_overlap(
            &ctx,
            focus_expr,
            remaining_expr
        )
    );
}

#[test]
fn solve_prep_candidate_variable_names_prefer_shifted_square_primary_variable() {
    let mut ctx = Context::new();
    let expr = parse("a*(x - b/(2*a))^2 + c - b^2/(4*a)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::collect_solve_prep_candidate_variable_names(&ctx, expr),
        vec!["x".to_string()]
    );
}

#[test]
fn solve_prep_candidate_variable_names_fall_back_to_raw_squared_variable() {
    let mut ctx = Context::new();
    let expr = parse("a*x^2 - b*x + c", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::collect_solve_prep_candidate_variable_names(&ctx, expr),
        vec!["x".to_string()]
    );
}

#[test]
fn solve_prep_candidate_variable_names_ignore_division_only_correction_square() {
    let mut ctx = Context::new();
    let expr =
        parse("b^2/(4*a) + a*x^2 - b*x + c", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::collect_solve_prep_candidate_variable_names(&ctx, expr),
        vec!["x".to_string()]
    );
}

#[test]
fn solve_prep_candidate_variable_names_reject_division_only_square_fallback() {
    let mut ctx = Context::new();
    let expr = parse("b^2/(4*a) - c", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::collect_solve_prep_candidate_variable_names(&ctx, expr),
        Vec::<String>::new()
    );
}

#[test]
fn solve_prep_candidate_variable_names_reject_square_without_linear_term() {
    let mut ctx = Context::new();
    let expr = parse("b^2 - c", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::collect_solve_prep_candidate_variable_names(&ctx, expr),
        Vec::<String>::new()
    );
}

#[test]
fn complete_square_binomial_expr_flips_negative_shift_for_positive_orientation() {
    let mut ctx = Context::new();
    let x = parse("x", &mut ctx).unwrap_or_else(|err| panic!("parse x: {err}"));
    let neg_shift = parse("-b/(2*a)", &mut ctx).unwrap_or_else(|err| panic!("parse shift: {err}"));

    let expr = super::build_complete_square_binomial_expr(
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

    let expr = super::build_complete_square_binomial_expr(
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
fn fast_solve_prep_zero_scope_rewrite_rejects_mismatched_shifted_square_variable() {
    let mut ctx = Context::new();
    let expr = parse(
        "a*x^2 - b*x + c - (a*(y - b/(2*a))^2 + c - b^2/(4*a))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr).is_none());
}

#[test]
fn solve_prep_equivalence_helper_rewrites_raw_quadratic() {
    let mut ctx = Context::new();
    let expr = parse("a*x^2 - b*x + c", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite_match =
        super::try_rewrite_exact_solve_prep_equivalence_for_cancellation(&mut ctx, expr)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite_match.rewritten
            }
        ),
        "a * (x - b / (2 * a))^2 + c - b^2 / (4 * a)"
    );
}

#[test]
fn solve_prep_zero_scope_rewrite_matches_raw_difference() {
    let mut ctx = Context::new();
    let expr = parse(
        "a*x^2 - b*x + c - (a*(x - b/(2*a))^2 + c - b^2/(4*a))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_exact_solve_prep_zero_scope_rewrite(&mut ctx, expr)
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
fn fast_solve_prep_zero_scope_rewrite_matches_raw_difference() {
    let mut ctx = Context::new();
    let expr = parse(
        "a*x^2 + b*x + c - (a*(x + b/(2*a))^2 + c - b^2/(4*a))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("fast rewrite"));
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
fn fast_solve_prep_zero_scope_rewrite_matches_negative_leading_difference() {
    let mut ctx = Context::new();
    let expr = parse(
        "-a*x^2 + b*x + c - (-a*(x - b/(2*a))^2 + c + b^2/(4*a))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = match super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr) {
        Some(rewrite) => rewrite,
        None => {
            let focus_expr = parse("-a*x^2 + b*x + c", &mut ctx)
                .unwrap_or_else(|err| panic!("focus parse: {err}"));
            let remaining_expr = parse("a*(x - b/(2*a))^2 - c - b^2/(4*a)", &mut ctx)
                .unwrap_or_else(|err| panic!("remaining parse: {err}"));
            let canonical_neg_remaining =
                super::normalize_additive_scope_expr(&mut ctx, remaining_expr);
            eprintln!(
                "canonical_neg_remaining={}",
                DisplayExpr {
                    context: &ctx,
                    id: canonical_neg_remaining
                }
            );
            for rewrite_match in
                super::collect_exact_solve_prep_equivalence_rewrites_for_cancellation(
                    &mut ctx, focus_expr,
                )
            {
                let canonical_rewritten =
                    super::normalize_additive_scope_expr(&mut ctx, rewrite_match.rewritten);
                eprintln!(
                    "canonical_rewritten={}",
                    DisplayExpr {
                        context: &ctx,
                        id: canonical_rewritten
                    }
                );
            }
            panic!("fast rewrite");
        }
    };
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
fn fast_solve_prep_zero_scope_rewrite_matches_negative_leading_difference_with_unsimplified_c() {
    let mut ctx = Context::new();
    let expr = parse(
        "-a*x^2 + b*x + (c + d - d) - (-a*(x - b/(2*a))^2 + (c + d - d) + b^2/(4*a))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("fast rewrite"));
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
fn fast_solve_prep_zero_scope_rewrite_matches_positive_symbolic_difference_with_unsimplified_c() {
    let mut ctx = Context::new();
    let expr = parse(
        "a*x^2 + b*x + (c + d - d) - (a*(x + b/(2*a))^2 + (c + d - d) - b^2/(4*a))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("fast rewrite"));
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
fn fast_solve_prep_zero_scope_rewrite_matches_fractional_symbolic_difference_with_unsimplified_c() {
    let mut ctx = Context::new();
    let expr = parse(
        "(a/2)*x^2 + b*x + (c + d - d) - ((a/2)*(x + b/a)^2 + (c + d - d) - b^2/(2*a))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("fast rewrite"));
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
fn solve_prep_equivalence_helper_rewrites_fractional_symbolic_leading_quadratic() {
    let mut ctx = Context::new();
    let expr = parse("(a/2)*x^2 + b*x + c", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let expected = parse("(a/2)*(x + b/a)^2 + c - b^2/(2*a)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse expected: {err}"));

    let rewrite_match =
        super::try_rewrite_exact_solve_prep_equivalence_for_cancellation(&mut ctx, expr)
            .unwrap_or_else(|| panic!("rewrite"));
    let residual = ctx.add(Expr::Sub(rewrite_match.rewritten, expected));

    assert!(super::is_zero_after_default_simplify(&mut ctx, residual));
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: rewrite_match.nonzero_expr
            }
        ),
        "a"
    );
}

#[test]
fn fast_solve_prep_zero_scope_rewrite_matches_fractional_symbolic_leading_difference() {
    let mut ctx = Context::new();
    let expr = parse(
        "(a/2)*x^2 + b*x + c - ((a/2)*(x + b/a)^2 + c - b^2/(2*a))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("fast rewrite"));
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
fn fast_solve_prep_zero_scope_rewrite_matches_generic_leading_difference() {
    let mut ctx = Context::new();
    let expr = parse(
        "(m + n)*x^2 + b*x + c - ((m + n)*(x + b/(2*(m + n)))^2 + c - b^2/(4*(m + n)))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("fast rewrite"));
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
fn fast_solve_prep_zero_scope_rewrite_matches_non_contiguous_passthrough_difference() {
    let mut ctx = Context::new();
    let expr = parse("b^2/(4*a) + a*x^2 - a*(x - b/(2*a))^2 - b*x", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr)
        .unwrap_or_else(|| panic!("fast rewrite"));
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
fn solve_prep_zero_scope_rewrite_matches_non_contiguous_passthrough_difference() {
    let mut ctx = Context::new();
    let expr = parse("b^2/(4*a) + a*x^2 - a*(x - b/(2*a))^2 - b*x", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::try_build_exact_solve_prep_zero_scope_rewrite(&mut ctx, expr)
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
fn solve_prep_rewritten_focus_plus_remaining_simplifies_to_zero() {
    let mut ctx = Context::new();
    let raw = parse("a*x^2 - b*x + c", &mut ctx).unwrap_or_else(|err| panic!("parse raw: {err}"));
    let target = parse("a*(x - b/(2*a))^2 + c - b^2/(4*a)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse target: {err}"));

    let rewrite_match =
        super::try_rewrite_exact_solve_prep_equivalence_for_cancellation(&mut ctx, raw)
            .unwrap_or_else(|| panic!("rewrite"));
    let candidate_total = ctx.add(Expr::Sub(rewrite_match.rewritten, target));

    assert!(super::is_zero_after_default_simplify(
        &mut ctx,
        candidate_total
    ));
}

#[test]
fn collapse_exact_zero_three_term_subset_rule_matches_sine_sum_with_passthrough_one() {
    let mut ctx = Context::new();
    let expr = parse(
        "sin(x) + sin(y) + 1 - 2*sin((x+y)/2)*cos((x-y)/2)",
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
    assert!(
        rewrite.description.is_empty() || rewrite.description == "Expand sine sum to product",
        "expected direct subset collapse or the legacy sine-sum expansion label, got {:?}",
        rewrite.description
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

    let rewrite_match = super::try_find_trig_phase_shift_cancellation_match(
        &mut ctx,
        focus_expr,
        target_expr,
        false,
    )
    .unwrap_or_else(|| panic!("rewrite match"));

    assert_eq!(
        rewrite_match.mode,
        super::TrigPhaseShiftCancellationMode::ShiftedToShifted
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
        super::resolve_surface_shifted_candidate_vs_plain_trig_target_for_phase_shift(
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
        super::extract_general_phase_shift_term_data_for_cancellation(&mut ctx, focus_expr)
            .unwrap_or_else(|| panic!("focus data"));
    let (arg, sin_coeff, cos_coeff, sin_sign, cos_sign) =
        super::extract_general_phase_shift_linear_signature_for_cancellation(&mut ctx, focus_data)
            .unwrap_or_else(|| panic!("focus linear signature"));
    let target_data =
        super::extract_general_phase_shift_term_data_for_cancellation(&mut ctx, target_expr)
            .unwrap_or_else(|| panic!("target data"));

    assert!(
        super::matches_general_phase_shift_shifted_term_candidate_for_cancellation(
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
        super::extract_weighted_phase_shift_linear_combination_for_cancellation(
            &mut ctx, focus_expr,
        )
        .unwrap_or_else(|| panic!("focus linear signature"));
    let target_data =
        super::extract_general_phase_shift_term_data_for_cancellation(&mut ctx, target_expr)
            .unwrap_or_else(|| panic!("target data"));

    assert!(
        super::matches_general_phase_shift_shifted_term_candidate_for_cancellation(
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

    let rewrite = super::try_build_exact_trig_phase_shift_zero_scope_rewrite(&mut ctx, expr)
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

    let rewrite = super::try_build_exact_trig_phase_shift_zero_scope_rewrite(&mut ctx, expr)
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

    let rewrite = super::try_build_exact_trig_phase_shift_zero_scope_rewrite(&mut ctx, expr)
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

    let rewrite_match = super::try_find_trig_phase_shift_cancellation_match(
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

    let rewrite_match = super::try_find_trig_phase_shift_cancellation_match(
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

    let rewrite_match = super::try_find_trig_phase_shift_cancellation_match(
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

    let rewrite_match = super::try_find_trig_phase_shift_cancellation_match(
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
        super::binary_add_pair_is_surface_plain_trig_against_shift_signal_for_phase_shift(
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

    let rewrite_match = super::try_find_trig_phase_shift_cancellation_match(
        &mut ctx,
        focus_expr,
        target_expr,
        false,
    )
    .unwrap_or_else(|| panic!("rewrite match"));

    assert_eq!(
        rewrite_match.mode,
        super::TrigPhaseShiftCancellationMode::LinearToShifted
    );
}

#[test]
fn extract_structural_exact_phase_shift_term_data_accepts_mul_one_pi_over_four_regression() {
    let mut ctx = Context::new();
    let expr = parse("sqrt(2)*sin((1*pi)/4 + x)", &mut ctx)
        .unwrap_or_else(|err| panic!("expr parse: {err}"));
    let x = parse("x", &mut ctx).unwrap_or_else(|err| panic!("x parse: {err}"));

    let (arg, coeff, kind, sin_sign, cos_sign) =
        super::extract_structural_exact_phase_shift_term_data_for_cancellation(&mut ctx, expr)
            .unwrap_or_else(|| panic!("extract"));
    let one = ctx.num(1);

    assert_eq!(compare_expr(&ctx, arg, x), Ordering::Equal);
    assert_eq!(compare_expr(&ctx, coeff, one), Ordering::Equal);
    assert!(matches!(
        kind,
        super::PhaseShiftKindForCancellation::Quarter
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

#[test]
fn expand_trig_square_identity_to_enable_cancellation_rule_matches_binomial_square() {
    let mut ctx = Context::new();
    let expr = parse("(sin(x) + cos(x))^2 - (1 + sin(2*x))", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = ExpandTrigSquareIdentityToEnableCancellationRule;
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
    assert_empty_or_legacy_description(&rewrite.description, "Trig Square Identity");
    assert_eq!(rewrite.substeps.len(), 4);
}

#[test]
fn extract_scaled_double_sine_product_for_cancellation_matches_canonical_order() {
    let mut ctx = Context::new();
    let expr = parse("2*sin(2*x)*sin(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let (scale, arg) = extract_scaled_double_sine_product_for_cancellation(&mut ctx, expr)
        .unwrap_or_else(|| panic!("extract"));

    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: scale
            }
        ),
        "1"
    );
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &ctx,
                id: arg
            }
        ),
        "x"
    );
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

#[test]
fn subtract_expanded_sum_diff_cubes_quotient_rule_matches_trig_square_cube_residual() {
    let mut ctx = Context::new();
    let expr = parse(
        "((sin(u)^2)^3 - 1)/((sin(u)^2) - 1) - ((sin(u)^2)^2 + (sin(u)^2) + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let (lhs, rhs) = match ctx.get(expr) {
        Expr::Sub(lhs, rhs) => (*lhs, *rhs),
        Expr::Add(lhs, rhs) => match (ctx.get(*lhs), ctx.get(*rhs)) {
            (_, Expr::Neg(inner)) => (*lhs, *inner),
            (Expr::Neg(inner), _) => (*rhs, *inner),
            _ => panic!("unexpected add form"),
        },
        other => panic!("unexpected root: {other:?}"),
    };
    let (num, den) = match ctx.get(lhs) {
        Expr::Div(num, den) => (*num, *den),
        other => panic!("unexpected lhs: {other:?}"),
    };
    let plan = crate::rules::algebra::fractions::try_plan_sum_diff_of_cubes_in_num(
        &mut ctx, num, den, false,
    )
    .unwrap_or_else(|| panic!("plan"));
    let cancelled = canonicalize_nested_integer_powers(&mut ctx, plan.cancelled_result);
    let rhs = canonicalize_nested_integer_powers(&mut ctx, rhs);
    assert!(
        cas_math::expr_domain::exprs_equivalent(&ctx, cancelled, rhs)
            || exprs_equal_up_to_add_term_order(&ctx, cancelled, rhs),
        "cancelled={} rhs={}",
        DisplayExpr {
            context: &ctx,
            id: cancelled
        },
        DisplayExpr {
            context: &ctx,
            id: rhs
        }
    );

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = SubtractExpandedSumDiffCubesQuotientRule;
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
fn subtract_expanded_sum_diff_cubes_quotient_rule_matches_trig_square_cube_plain_fourth_power_residual(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "((sin(u)^2)^3 - 1)/((sin(u)^2) - 1) - (sin(u)^4 + sin(u)^2 + 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = SubtractExpandedSumDiffCubesQuotientRule;
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
fn direct_sum_diff_cubes_core_equivalence_rewrite_matches_difference_pair() {
    let mut ctx = Context::new();
    let lhs = parse("(a^3-b^3)/(a-b)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("a^2 + a*b + b^2", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = try_build_direct_sum_diff_cubes_quotient_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        rewrite.description,
        "Subtract Expanded Sum/Difference of Cubes Quotient"
    );
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
    assert_eq!(rewrite.required_conditions.len(), 1);
}

#[test]
fn direct_sum_diff_cubes_core_equivalence_rewrite_matches_sum_pair() {
    let mut ctx = Context::new();
    let lhs = parse("(a^3+b^3)/(a+b)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("a^2 - a*b + b^2", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = try_build_direct_sum_diff_cubes_quotient_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_eq!(
        rewrite.description,
        "Subtract Expanded Sum/Difference of Cubes Quotient"
    );
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
    assert_eq!(rewrite.required_conditions.len(), 1);
}

#[test]
fn tan_triple_angle_contraction_matcher_matches_canonical_rational_form() {
    let mut ctx = Context::new();
    let expr = parse("((3*tan(x)-tan(x)^3)/(1-3*tan(x)^2))", &mut ctx)
        .unwrap_or_else(|err| panic!("expr: {err}"));
    let x = parse("x", &mut ctx).unwrap_or_else(|err| panic!("x: {err}"));

    let (matched_arg, _) = super::match_tan_triple_angle_contraction_arg(&mut ctx, expr)
        .unwrap_or_else(|| panic!("matcher"));

    assert_eq!(
        cas_ast::ordering::compare_expr(&ctx, matched_arg, x),
        std::cmp::Ordering::Equal
    );
}

#[test]
fn direct_trig_cos_diff_sin_diff_quotient_equivalence_rewrite_matches() {
    let mut ctx = Context::new();
    let lhs = parse("(cos(x)-cos(3*x))/(sin(3*x)-sin(x))", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("tan(2*x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_trig_cos_diff_sin_diff_quotient_equivalence_rewrite(
        &mut ctx, lhs, rhs,
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Cos-Diff / Sin-Diff Quotient");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
    assert_eq!(rewrite.required_conditions.len(), 1);
}

#[test]
fn collapse_exact_zero_additive_subexpression_matches_cos_diff_sin_diff_quotient() {
    let mut ctx = Context::new();
    let expr = parse("(cos(x)-cos(3*x))/(sin(3*x)-sin(x)) - tan(2*x)", &mut ctx)
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
    assert_empty_or_legacy_description(&rewrite.description, "Cos-Diff / Sin-Diff Quotient");
    assert_eq!(rewrite.required_conditions.len(), 1);
}

#[test]
fn direct_cos_product_telescoping_equivalence_rewrite_matches_morrie_target() {
    let mut ctx = Context::new();
    let lhs =
        parse("cos(x)*cos(2*x)*cos(4*x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("(sin(8*x)/(8*sin(x)))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::try_build_direct_cos_product_telescoping_equivalence_rewrite(&mut ctx, lhs, rhs)
            .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Apply Morrie's law");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert!(!rewrite.required_conditions.is_empty());
}

#[test]
fn collapse_exact_zero_additive_subexpression_matches_morrie_raw_difference() {
    let mut ctx = Context::new();
    let expr = parse("cos(x)*cos(2*x)*cos(4*x) - (sin(8*x)/(8*sin(x)))", &mut ctx)
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
    assert_empty_or_legacy_description(&rewrite.description, "Apply Morrie's law");
}

#[test]
fn direct_multi_angle_equivalence_rewrite_matches_tangent_triple_angle_reverse() {
    let mut ctx = Context::new();
    let lhs = parse("((3*tan(x)-tan(x)^3)/(1-3*tan(x)^2))", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("tan(3*x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::try_build_direct_multi_angle_equivalence_rewrite(&mut ctx, lhs, rhs)
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
