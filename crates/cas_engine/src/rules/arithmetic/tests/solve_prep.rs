//! Tests de las reglas aritméticas: familia `solve_prep` (troceo P1).

use super::*;

#[test]
fn maybe_direct_small_zero_additive_combination_candidate_accepts_solve_prep_and_nested_fraction_sum(
) {
    let mut ctx = Context::new();
    let expr = parse(
        "(x^2 + 2*b*x - ((x+b)^2 - b^2)) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::super::maybe_direct_small_zero_additive_combination_candidate(&mut ctx, expr));
}
#[test]
fn direct_small_zero_additive_combination_rewrite_matches_solve_prep_and_nested_fraction_sum() {
    let mut ctx = Context::new();
    let expr = parse(
        "(x^2 + 2*b*x - ((x+b)^2 - b^2)) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
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
fn solve_prep_equivalence_helper_rewrites_fraction_with_common_denominator() {
    let mut ctx = Context::new();
    let expr = parse("(a*x^2 + b*x + c)/q", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));
    let rewrite_match =
        super::super::try_rewrite_exact_solve_prep_equivalence_for_cancellation(&mut ctx, expr)
            .unwrap_or_else(|| panic!("rewrite"));
    let expected = parse("(a*(x + b/(2*a))^2 + c - b^2/(4*a))/q", &mut ctx)
        .unwrap_or_else(|err| panic!("parse expected: {err}"));

    assert!(super::super::exprs_match_after_default_simplify(
        &mut ctx,
        rewrite_match.rewritten,
        expected
    ));
}
#[test]
fn solve_prep_exact_additive_candidate_detects_complete_square_pair() {
    let mut ctx = Context::new();
    let expr = parse(
        "a*x^2 - b*x + c - (a*(x - b/(2*a))^2 + c - b^2/(4*a))",
        &mut ctx,
    )
    .unwrap_or_else(|err| panic!("parse: {err}"));

    assert!(super::super::maybe_solve_prep_exact_additive_candidate(
        &ctx, expr
    ));
}
#[test]
fn solve_prep_focus_remaining_variable_overlap_detects_complete_square_pair() {
    let mut ctx = Context::new();
    let focus_expr =
        parse("a*x^2 - b*x + c", &mut ctx).unwrap_or_else(|err| panic!("focus parse: {err}"));
    let remaining_expr = parse("-(a*(x - b/(2*a))^2 + c - b^2/(4*a))", &mut ctx)
        .unwrap_or_else(|err| panic!("remaining parse: {err}"));

    assert!(
        super::super::has_plausible_solve_prep_focus_remaining_variable_overlap(
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
        !super::super::has_plausible_solve_prep_focus_remaining_variable_overlap(
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
        !super::super::has_plausible_solve_prep_focus_remaining_variable_overlap(
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
        super::super::collect_solve_prep_candidate_variable_names(&ctx, expr),
        vec!["x".to_string()]
    );
}
#[test]
fn solve_prep_candidate_variable_names_fall_back_to_raw_squared_variable() {
    let mut ctx = Context::new();
    let expr = parse("a*x^2 - b*x + c", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::super::collect_solve_prep_candidate_variable_names(&ctx, expr),
        vec!["x".to_string()]
    );
}
#[test]
fn solve_prep_candidate_variable_names_ignore_division_only_correction_square() {
    let mut ctx = Context::new();
    let expr =
        parse("b^2/(4*a) + a*x^2 - b*x + c", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::super::collect_solve_prep_candidate_variable_names(&ctx, expr),
        vec!["x".to_string()]
    );
}
#[test]
fn solve_prep_candidate_variable_names_reject_division_only_square_fallback() {
    let mut ctx = Context::new();
    let expr = parse("b^2/(4*a) - c", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::super::collect_solve_prep_candidate_variable_names(&ctx, expr),
        Vec::<String>::new()
    );
}
#[test]
fn solve_prep_candidate_variable_names_reject_square_without_linear_term() {
    let mut ctx = Context::new();
    let expr = parse("b^2 - c", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    assert_eq!(
        super::super::collect_solve_prep_candidate_variable_names(&ctx, expr),
        Vec::<String>::new()
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

    assert!(
        super::super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr).is_none()
    );
}
#[test]
fn solve_prep_equivalence_helper_rewrites_raw_quadratic() {
    let mut ctx = Context::new();
    let expr = parse("a*x^2 - b*x + c", &mut ctx).unwrap_or_else(|err| panic!("parse: {err}"));

    let rewrite_match =
        super::super::try_rewrite_exact_solve_prep_equivalence_for_cancellation(&mut ctx, expr)
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

    let rewrite = super::super::try_build_exact_solve_prep_zero_scope_rewrite(&mut ctx, expr)
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

    let rewrite = super::super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr)
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

    let rewrite =
        match super::super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr) {
            Some(rewrite) => rewrite,
            None => {
                let focus_expr = parse("-a*x^2 + b*x + c", &mut ctx)
                    .unwrap_or_else(|err| panic!("focus parse: {err}"));
                let remaining_expr = parse("a*(x - b/(2*a))^2 - c - b^2/(4*a)", &mut ctx)
                    .unwrap_or_else(|err| panic!("remaining parse: {err}"));
                let canonical_neg_remaining =
                    super::super::normalize_additive_scope_expr(&mut ctx, remaining_expr);
                eprintln!(
                    "canonical_neg_remaining={}",
                    DisplayExpr {
                        context: &ctx,
                        id: canonical_neg_remaining
                    }
                );
                for rewrite_match in
                    super::super::collect_exact_solve_prep_equivalence_rewrites_for_cancellation(
                        &mut ctx, focus_expr,
                    )
                {
                    let canonical_rewritten = super::super::normalize_additive_scope_expr(
                        &mut ctx,
                        rewrite_match.rewritten,
                    );
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

    let rewrite = super::super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr)
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

    let rewrite = super::super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr)
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

    let rewrite = super::super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr)
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
        super::super::try_rewrite_exact_solve_prep_equivalence_for_cancellation(&mut ctx, expr)
            .unwrap_or_else(|| panic!("rewrite"));
    let residual = ctx.add(Expr::Sub(rewrite_match.rewritten, expected));

    assert!(super::super::is_zero_after_default_simplify(
        &mut ctx, residual
    ));
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

    let rewrite = super::super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr)
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

    let rewrite = super::super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr)
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

    let rewrite = super::super::try_build_fast_solve_prep_exact_zero_scope_rewrite(&mut ctx, expr)
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

    let rewrite = super::super::try_build_exact_solve_prep_zero_scope_rewrite(&mut ctx, expr)
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
        super::super::try_rewrite_exact_solve_prep_equivalence_for_cancellation(&mut ctx, raw)
            .unwrap_or_else(|| panic!("rewrite"));
    let candidate_total = ctx.add(Expr::Sub(rewrite_match.rewritten, target));

    assert!(super::super::is_zero_after_default_simplify(
        &mut ctx,
        candidate_total
    ));
}
