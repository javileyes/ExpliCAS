//! Tests de las reglas aritméticas: familia `zero_collapse` (troceo P1).

use super::*;

#[test]
fn additive_scope_contains_zero_term_detects_top_level_zero_regression() {
    let mut ctx = Context::new();
    let expr = parse("(ln(x) - ln(x)) + 0", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::super::additive_scope_contains_zero_term(
        &mut ctx, expr
    ));
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
fn collapse_exact_zero_common_scaled_difference_preserves_assumed_abs_event() {
    let mut ctx = Context::new();
    let expr = parse("2*a - 2*abs(a)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let parent_ctx = ParentContext::root().with_domain_mode(DomainMode::Assume);

    let event =
        super::super::common_scale_abs_like_positive_assumption_event(&mut ctx, expr, &parent_ctx)
            .unwrap_or_else(|| panic!("expected common-scale abs assumption event"));
    assert_eq!(event.message, "a > 0");

    assert_eq!(event.expr_display, "a");
}
#[test]
fn small_direct_zero_core_rewrite_matches_signed_common_factor_core() {
    let mut ctx = Context::new();
    let expr = parse("2*cos(x) - x^2*cos(x) - cos(x)*(2-x^2)", &mut ctx)
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
fn small_direct_zero_core_rewrite_matches_gap_two_factorial_ratio_core() {
    let mut ctx = Context::new();
    let expr =
        parse("(n+1)!/(n-1)! - n*(n+1)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

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
    assert_eq!(rewrite.required_conditions.len(), 1);
}
#[test]
fn direct_small_zero_additive_combination_rewrite_rejects_non_partition_core_pair() {
    let mut ctx = Context::new();
    let expr = parse("(2*sin(x)*cos(y) - sin(x+y) - sin(x-y)) + z", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        super::super::try_build_direct_small_zero_additive_combination_rewrite(&mut ctx, expr)
            .is_none()
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
    assert!(!rewrite.required_conditions.is_empty());
}
#[test]
fn small_structural_poly_zero_core_rewrite_matches_collect_common_factor() {
    let mut ctx = Context::new();
    let expr = parse("u*v + u*w - u*(v+w)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite = super::super::try_build_small_structural_poly_zero_core_rewrite(&mut ctx, expr)
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

    assert!(
        super::super::small_zero_additive_combination_supported_partition_core(&mut ctx, dirichlet)
    );
    assert!(
        super::super::small_zero_additive_combination_supported_partition_core(&mut ctx, factor)
    );
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
        super::super::small_zero_additive_combination_max_terms(&ctx, expr),
        7
    );
    let terms = cas_math::expr_nary::AddView::from_expr(&ctx, expr).terms;
    assert_eq!(terms.len(), 7);
    let first_terms: Vec<_> = terms.iter().copied().take(4).collect();
    let second_terms: Vec<_> = terms.iter().copied().skip(4).collect();
    let first_expr = super::super::build_small_zero_partition_expr(&mut ctx, &first_terms);
    let second_expr = super::super::build_small_zero_partition_expr(&mut ctx, &second_terms);
    assert!(super::super::maybe_integrate_prep_exact_additive_candidate(
        &mut ctx, first_expr
    ));
    assert!(
        super::super::small_zero_additive_combination_supported_partition_core(
            &mut ctx, first_expr
        )
    );
    assert!(
        super::super::small_zero_additive_combination_supported_partition_core(
            &mut ctx,
            second_expr
        )
    );
    assert!(super::super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}
#[test]
fn direct_small_zero_additive_combination_rewrite_matches_dirichlet_and_factor_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))) + (p^2-q^2 - (p-q)*(p+q))",
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
fn direct_small_zero_additive_combination_rewrite_matches_dirichlet_and_collect_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))) + (u*v + u*w - u*(v+w))",
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
fn maybe_direct_small_zero_additive_combination_candidate_rejects_nontrig_polynomial_scope() {
    let mut ctx = Context::new();
    let expr = parse("(a+b-c) + (d+e-f)", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(!super::super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}
#[test]
fn shared_passthrough_difference_shape_gate_keeps_known_passthrough_zero_regression() {
    let mut ctx = Context::new();
    let expr = parse("((sin(x)^2 + cos(x)^2) + m) - ((1) + m)", &mut ctx)
        .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(
        super::super::has_plausible_shared_additive_passthrough_difference_shape(&mut ctx, expr)
    );
    let rewrite =
        super::super::try_build_exact_zero_shared_passthrough_difference_rewrite(&mut ctx, expr)
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
