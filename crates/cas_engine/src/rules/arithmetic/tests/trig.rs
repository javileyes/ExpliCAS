//! Tests de las reglas aritméticas: familia `trig` (troceo P1).

use super::*;

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
fn cancel_exact_additive_pairs_rule_matches_trig_and_numeric_pair_chain() {
    let mut ctx = Context::new();
    let expr = parse("2*cos(2*x) + 1 - 2*cos(2*x) - 1", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Generic);
    let rule = super::super::CancelExactAdditivePairsRule;
    let rewrite = rule
        .apply(&mut ctx, expr, &parent_ctx)
        .unwrap_or_else(|| panic!("rewrite"));
    let zero = ctx.num(0);

    assert_empty_or_legacy_description(&rewrite.description, "Cancel exact additive pairs");
    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite.final_expr(),
        zero
    ));
}
#[test]
fn exact_trig_equivalence_zero_scope_rewrite_matches_raw_sin_sin_residual() {
    let mut ctx = Context::new();
    let expr = parse("2*sin(x)*sin(y) - cos(x-y) + cos(x+y)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

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
fn maybe_exact_trig_equivalence_zero_scope_candidate_accepts_pure_trig_residual() {
    let mut ctx = Context::new();
    let expr =
        parse("sin(x)^2 + cos(x)^2 - 1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::super::maybe_exact_trig_equivalence_zero_scope_candidate(&mut ctx, expr));
}
#[test]
fn maybe_exact_trig_equivalence_zero_scope_candidate_rejects_two_term_direct_trig_against_product_scope(
) {
    let mut ctx = Context::new();
    let expr = parse("sin(x) - 2*sin((x+y)/2)*cos((x-y)/2)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::maybe_exact_trig_equivalence_zero_scope_candidate(&mut ctx, expr));
}
#[test]
fn maybe_exact_trig_equivalence_zero_scope_candidate_rejects_mixed_trig_log_exp_fraction_scope() {
    let mut ctx = Context::new();
    let expr = parse(
        "(atanh((u^2 - 1)/(u^2 + 1)) - log(u)) + (log((p*q)^2) - 2*log(p) - 2*log(q)) + (exp(r*log(s)) - s^r) + (sin(a) + sin(b) - 2*sin((a+b)/2)*cos((a-b)/2)) + (2/(t^2 - 1) - 1/(t-1) + 1/(t+1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::maybe_exact_trig_equivalence_zero_scope_candidate(&mut ctx, expr));
}
#[test]
fn maybe_exact_trig_equivalence_zero_scope_candidate_rejects_reciprocal_trig_mixed_scope() {
    let mut ctx = Context::new();
    let expr = parse(
        "(tan(x)*cot(x) - 1) + (2*sin(x)*sin(y) - cos(x-y) + cos(x+y))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::maybe_exact_trig_equivalence_zero_scope_candidate(&mut ctx, expr));
}
#[test]
fn reject_scaled_surface_trig_power_vs_numeric_atom_before_default_simplify_matches_symbolic_power()
{
    let mut ctx = Context::new();
    let lhs = parse("4*cos(x)^2", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs = parse("1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::super::reject_scaled_surface_trig_power_vs_numeric_atom_before_default_simplify(
            &mut ctx, lhs, rhs
        ),
        Some(false)
    );
}
#[test]
fn reject_plain_surface_trig_power_gap_before_default_simplify_matches_scaled_gap() {
    let mut ctx = Context::new();
    let lhs = parse("2*sin(x)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rhs = parse("3*sin(x)^2", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::super::reject_plain_surface_trig_power_gap_before_default_simplify(
            &mut ctx, lhs, rhs
        ),
        Some(false)
    );
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
fn small_direct_zero_core_rewrite_matches_tan_cot_sec_csc_core() {
    let mut ctx = Context::new();
    let expr = parse("tan(x) + cot(x) - sec(x)*csc(x)", &mut ctx)
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
    assert_eq!(rewrite.required_conditions.len(), 2);
}
#[test]
fn small_direct_zero_core_rewrite_matches_tan_cot_product_core() {
    let mut ctx = Context::new();
    let expr = parse("tan(x)*cot(x) - 1", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

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
    assert_eq!(rewrite.required_conditions.len(), 2);
}
#[test]
fn direct_small_zero_additive_combination_rewrite_matches_trig_and_gap_two_factorial_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + ((n+1)!/(n-1)! - n*(n+1))",
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
fn maybe_direct_small_zero_additive_combination_candidate_accepts_trig_and_telescoping_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(2*cos(2*x)*sin(x) - (4*cos(x)^2*sin(x)-2*sin(x))) + (1/(u*(u+1)) - 1/u + 1/(u+1))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}
#[test]
fn maybe_direct_small_zero_additive_combination_candidate_accepts_pure_trig_partition_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + (tan(x)*cot(x) - 1)",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
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
fn direct_core_equivalence_rewrite_matches_trig_binomial_square_diff_pair() {
    let mut ctx = Context::new();
    let lhs = parse("(sin(x)-cos(x))^2", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("1-sin(2*x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Trig Square Identity");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn direct_core_equivalence_rewrite_rejects_plain_cross_trig_pair_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("sin(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("-(cos(y)*sqrt(3))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}
#[test]
fn direct_core_equivalence_rewrite_rejects_noncall_vs_surface_symbolic_trig_before_default_simplify(
) {
    let mut ctx = Context::new();
    let lhs = parse("a", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs =
        parse("-(sin(x + pi/4)*sqrt(2))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}
#[test]
fn direct_core_equivalence_rewrite_rejects_plain_surface_trig_power_gap_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("cos(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("cos(x)^3", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_none());
}
#[test]
fn direct_core_equivalence_rewrite_keeps_exact_surface_trig_power_match_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("cos(pi)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("cos(pi)^3", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_some());
}
#[test]
fn direct_core_equivalence_rewrite_keeps_inverse_trig_surface_match_before_default_simplify() {
    let mut ctx = Context::new();
    let lhs = parse("x", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sin(arcsin(x))", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite = super::super::try_build_direct_core_equivalence_rewrite(&mut ctx, lhs, rhs);
    assert!(rewrite.is_some());
}
#[test]
fn direct_trig_reciprocal_equivalence_rewrite_matches_sec() {
    let mut ctx = Context::new();
    let lhs = parse("1/cos(x)", &mut ctx).unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("sec(x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_trig_reciprocal_equivalence_rewrite(&mut ctx, lhs, rhs)
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

    let rewrite = super::super::try_build_direct_trig_ratio_equivalence_rewrite(&mut ctx, lhs, rhs)
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

    let rewrite = super::super::try_build_direct_trig_ratio_equivalence_rewrite(&mut ctx, lhs, rhs)
        .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Trigonometric Quotient Identity");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert_eq!(rewrite.after_local, Some(rhs));
}
#[test]
fn collapse_exact_zero_same_denominator_rule_matches_trig_reciprocal_core() {
    let mut ctx = Context::new();
    let expr = parse("((1/cos(x))/q) - ((sec(x))/q)", &mut ctx)
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
    assert_empty_or_legacy_description(&rewrite.description, "Reciprocal Quotient Identity");
}
#[test]
fn collapse_exact_zero_same_denominator_rule_matches_trig_ratio_core_tan() {
    let mut ctx = Context::new();
    let expr = parse("((sin(2*x)/cos(x+x))/q) - ((tan(2*x))/q)", &mut ctx)
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
    assert_empty_or_legacy_description(&rewrite.description, "Trigonometric Quotient Identity");
}
#[test]
fn collapse_exact_zero_same_denominator_rule_matches_trig_ratio_core_cot() {
    let mut ctx = Context::new();
    let expr = parse("((cos(x)/sin(x))/q) - ((cot(x))/q)", &mut ctx)
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

    let rewrite = super::super::try_build_small_direct_zero_core_rewrite(&mut ctx, expr)
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
fn direct_trig_cos_diff_sin_diff_quotient_equivalence_rewrite_matches() {
    let mut ctx = Context::new();
    let lhs = parse("(cos(x)-cos(3*x))/(sin(3*x)-sin(x))", &mut ctx)
        .unwrap_or_else(|err| panic!("lhs: {err}"));
    let rhs = parse("tan(2*x)", &mut ctx).unwrap_or_else(|err| panic!("rhs: {err}"));

    let rewrite =
        super::super::try_build_direct_trig_cos_diff_sin_diff_quotient_equivalence_rewrite(
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

    let rewrite = super::super::try_build_direct_cos_product_telescoping_equivalence_rewrite(
        &mut ctx, lhs, rhs,
    )
    .unwrap_or_else(|| panic!("rewrite"));

    assert_empty_or_legacy_description(&rewrite.description, "Apply Morrie's law");
    assert_eq!(rewrite.before_local, Some(lhs));
    assert!(!rewrite.required_conditions.is_empty());
}
