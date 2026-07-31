//! Tests de las reglas aritméticas: familia `fractions` (troceo P1).

use super::*;

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
fn small_direct_zero_core_rewrite_matches_telescoping_fraction_core() {
    let mut ctx = Context::new();
    let expr =
        parse("1/(u*(u+1)) - 1/u + 1/(u+1)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

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
fn direct_fraction_telescoping_zero_scope_rewrite_rejects_mixed_nonfraction_three_term_shape() {
    let mut ctx = Context::new();
    let expr =
        parse("sin(x) - 1/u + 1/(u+1)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        super::super::try_build_direct_fraction_telescoping_zero_scope_rewrite(&mut ctx, expr)
            .is_none()
    );
}
#[test]
fn small_direct_zero_core_rewrite_matches_reciprocal_nested_fraction_core() {
    let mut ctx = Context::new();
    let expr = parse("(1/x + 1/y)/(1/x - 1/y) - (x+y)/(y-x)", &mut ctx)
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
    assert_eq!(rewrite.required_conditions.len(), 4);
}
#[test]
fn small_direct_zero_core_rewrite_matches_symmetric_partial_fraction_core() {
    let mut ctx = Context::new();
    let expr = parse("1/(x - 1) - 1/(x + 1) - 2/(x^2 - 1)", &mut ctx)
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
    assert_eq!(rewrite.required_conditions.len(), 3);
}
#[test]
fn direct_small_zero_additive_combination_rewrite_flattens_signed_nested_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "2*cos(x) + 2*x*sin(x) - (2*x*sin(x)+(2-x^2)*cos(x)) - x^2*cos(x)",
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
fn maybe_direct_small_zero_additive_combination_candidate_accepts_integrate_prep_and_nested_fraction_sum(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "(cos(x)*cos(2*x)*cos(4*x) - sin(8*x)/(8*sin(x))) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}
#[test]
fn direct_small_zero_additive_combination_rewrite_matches_integrate_prep_and_nested_fraction_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(cos(x)*cos(2*x)*cos(4*x) - sin(8*x)/(8*sin(x))) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
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
    assert!(!rewrite.required_conditions.is_empty());
}
#[test]
fn collapse_exact_zero_same_denominator_rule_matches_fraction_difference_combination_pair() {
    let mut ctx = Context::new();
    let expr = parse(
        "((1/(x+b) - 1/(x+c))/q) - (((c-b)/(x^2+(b+c)*x+b*c))/q)",
        &mut ctx,
    )
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
fn collapse_exact_zero_same_denominator_rule_matches_scaled_fraction_difference_combination_pair() {
    let mut ctx = Context::new();
    let expr = parse(
        "((1/(2*a)*(1/(x-a) - 1/(x+a)))/q) - ((1/(x^2-a^2))/q)",
        &mut ctx,
    )
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

    let rewrite = super::super::try_build_direct_finite_sum_equivalence_rewrite(&mut ctx, lhs, rhs)
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

    let rewrite = super::super::try_build_direct_finite_sum_equivalence_rewrite(&mut ctx, lhs, rhs)
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

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
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

    let rewrite = super::super::try_build_exact_zero_identity_rewrite(&mut ctx, expr)
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
