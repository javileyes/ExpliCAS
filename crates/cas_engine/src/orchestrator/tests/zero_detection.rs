//! Tests del orquestador: familia `zero_detection` (troceo P1).

use super::*;

#[test]
fn standard_exact_additive_pair_chain_shortcut_cancels_symbolic_passthrough_tail() {
    let mut ctx = Context::new();
    let expr = parse("m + 1 - m - 1", &mut ctx).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let (rewritten, _) = try_standard_exact_additive_pair_chain_shortcut(
        &crate::SimplifyOptions::default(),
        &mut ctx,
        expr,
        false,
    )
    .unwrap_or_else(|| panic!("shortcut should cancel symbolic passthrough tail"));
    assert_eq!(render(&ctx, rewritten), "0");
}
#[test]
fn simplify_pipeline_handles_nested_additive_zero_sum_case21_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3) + ln(y^2) - ln(x^3 * y^2)) + (sin(x)^2 - (1 - cos(2*x))/2)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn child_isolated_exact_zero_reinterns_function_names_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(diff(sin(e^(x^2)),x) - 2*x*cos(e^(x^2))*e^(x^2)) + (u*v+u*w-u*(v+w))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(child_isolated_exact_zero(
        &options,
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn detects_direct_geometric_difference_zero_identity_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "x^6 - 1 - (x-1)*(x^5+x^4+x^3+x^2+x+1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        matches_direct_geometric_difference_zero_identity_root(&mut simplifier.context, expr),
        "expr={}",
        render(&simplifier.context, expr)
    );
    assert!(matches_direct_small_zero_or_known_pair_base_root(
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn detects_two_factor_product_pair_zero_difference_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((1/x + 1/(x+1)) * cos(2*x)) - (((2*x+1)/(x*(x+1))) * (2*cos(x)^2 - 1))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let view = AddView::from_expr(&ctx, expr);
    let lhs_factors = flatten_mul_chain(&mut ctx, view.terms[0].0)
        .into_iter()
        .map(|factor| render(&ctx, factor))
        .collect::<Vec<_>>();
    let rhs_factors = flatten_mul_chain(&mut ctx, view.terms[1].0)
        .into_iter()
        .map(|factor| render(&ctx, factor))
        .collect::<Vec<_>>();
    let lhs_term = view.terms[0].0;
    let rhs_term = view.terms[1].0;
    let lhs_factor_ids = flatten_mul_chain(&mut ctx, lhs_term);
    let rhs_factor_ids = flatten_mul_chain(&mut ctx, rhs_term);
    let pair_00 = factors_match_by_equality_or_direct_pair_root(
        &mut ctx,
        lhs_factor_ids[0],
        rhs_factor_ids[0],
    );
    let pair_01 = factors_match_by_equality_or_direct_pair_root(
        &mut ctx,
        lhs_factor_ids[0],
        rhs_factor_ids[1],
    );
    let pair_10 = factors_match_by_equality_or_direct_pair_root(
        &mut ctx,
        lhs_factor_ids[1],
        rhs_factor_ids[0],
    );
    let pair_11 = factors_match_by_equality_or_direct_pair_root(
        &mut ctx,
        lhs_factor_ids[1],
        rhs_factor_ids[1],
    );
    assert!(
        matches_direct_two_factor_product_pair_zero_difference_root(&mut ctx, expr),
        "lhs factors = {:?}, rhs factors = {:?}, pair00 = {}, pair01 = {}, pair10 = {}, pair11 = {}",
        lhs_factors,
        rhs_factors,
        pair_00,
        pair_01,
        pair_10,
        pair_11
    );
}
#[test]
fn detects_two_factor_product_pair_zero_difference_tangent_addition_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((sin(5*x)) * (tan(x) + tan(y))) - ((sin(5*x)) * (sin(x+y)/(cos(x)*cos(y))))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn detects_two_factor_product_pair_zero_difference_factoring_tangent_addition_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((z^2 + 2*z) * (tan(u) + tan(v))) - (((z*(z+2)) * (sin(u+v)/(cos(u)*cos(v)))))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn simplify_pipeline_handles_quartic_gcf_times_chebyshev_zero_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((z^4 - z^2) * (cos(2*u))) - (((z^2*(z-1)*(z+1))) * (2*cos(u)^2 - 1))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn detects_two_factor_product_pair_zero_difference_quintuple_chebyshev_regression() {
    let mut ctx = Context::new();
    let expr = parse(
        "((sin(5*x)) * (cos(2*x))) - (((16*sin(x)^5 - 20*sin(x)^3 + 5*sin(x)) * (2*cos(x)^2 - 1)))",
        &mut ctx,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(matches_direct_two_factor_product_pair_zero_difference_root(
        &mut ctx, expr
    ));
}
#[test]
fn direct_small_zero_core_group_requires_cancellation_marker() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let positive_expr =
        parse("x + y", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let positive_terms = AddView::from_expr(&simplifier.context, positive_expr).terms;
    assert!(!super::matches_direct_small_zero_core_group_root(
        &mut simplifier.context,
        positive_terms.as_slice(),
    ));

    let zero_expr = parse("p^2 - q^2 - (p-q)*(p+q)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let zero_terms = AddView::from_expr(&simplifier.context, zero_expr).terms;
    assert!(super::matches_direct_small_zero_core_group_root(
        &mut simplifier.context,
        zero_terms.as_slice(),
    ));
}
#[test]
fn direct_small_zero_core_groups_require_enough_cancellation_markers_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let sparse_expr = parse("a + b + c + d - e - f", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let sparse_terms = AddView::from_expr(&simplifier.context, sparse_expr).terms;
    assert_eq!(
        super::direct_small_zero_cancellation_marker_count_root(
            &simplifier.context,
            sparse_terms.as_slice(),
        ),
        2
    );
    assert!(
        !super::has_enough_direct_small_zero_cancellation_markers_root(
            &simplifier.context,
            sparse_terms.as_slice(),
            3,
        )
    );
    assert!(!super::matches_direct_three_small_zero_cores_terms_root(
        &mut simplifier.context,
        sparse_terms.as_slice(),
    ));

    let dense_expr = parse("a - b + c - d + e - f", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let dense_terms = AddView::from_expr(&simplifier.context, dense_expr).terms;
    assert!(
        super::has_enough_direct_small_zero_cancellation_markers_root(
            &simplifier.context,
            dense_terms.as_slice(),
            3,
        )
    );
}
#[test]
fn direct_small_zero_plain_three_core_gate_requires_two_exact_cancellation_pairs() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let noisy_plain_expr = parse(
        "a*x^2 + b*x + c*x^2 + d*x + e*x^2 + f - ((a + c + e)*x^2 + (b + d)*x + f)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let noisy_plain_terms = AddView::from_expr(&simplifier.context, noisy_plain_expr).terms;
    assert_eq!(
        super::direct_small_zero_opposite_sign_exact_pair_count_root(
            &simplifier.context,
            noisy_plain_terms.as_slice(),
        ),
        1
    );
    assert!(
        !super::should_try_direct_three_small_zero_cores_root(
            &simplifier.context,
            noisy_plain_expr,
            noisy_plain_terms.as_slice(),
        ),
        "plain collect misses should not enter the recursive three-core partitioner"
    );

    let passthrough_expr = parse("c + m + a*b + b*a - c - m - 2*a*b", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let passthrough_terms = AddView::from_expr(&simplifier.context, passthrough_expr).terms;
    assert_eq!(
        super::direct_small_zero_opposite_sign_exact_pair_count_root(
            &simplifier.context,
            passthrough_terms.as_slice(),
        ),
        2
    );
    assert!(super::should_try_direct_three_small_zero_cores_root(
        &simplifier.context,
        passthrough_expr,
        passthrough_terms.as_slice(),
    ));
}
#[test]
fn direct_small_zero_three_core_gate_requires_remaining_anchor_capacity() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let impossible_three_core_expr = parse(
        "((a*x^2 + b*x + c) + m) - ((x*(a*x + b + c/x)) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let impossible_terms =
        AddView::from_expr(&simplifier.context, impossible_three_core_expr).terms;
    assert_eq!(impossible_terms.len(), 6);
    assert_eq!(
        super::direct_small_zero_opposite_sign_exact_pair_count_root(
            &simplifier.context,
            impossible_terms.as_slice(),
        ),
        1
    );
    assert!(
        !super::has_enough_direct_small_zero_remaining_anchor_terms_root(
            &simplifier.context,
            impossible_terms.as_slice(),
            3,
        ),
        "one passthrough pair and one negated structured term cannot form three zero groups"
    );
    assert!(!super::should_try_direct_three_small_zero_cores_root(
        &simplifier.context,
        impossible_three_core_expr,
        impossible_terms.as_slice(),
    ));

    let three_core_expr = parse(
        "(a^2-b^2 - (a-b)*(a+b)) + (sec(y) - 1/cos(y)) + (u*v + u*w - u*(v+w))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let three_core_terms = AddView::from_expr(&simplifier.context, three_core_expr).terms;
    assert!(
        super::has_enough_direct_small_zero_remaining_anchor_terms_root(
            &simplifier.context,
            three_core_terms.as_slice(),
            3,
        )
    );
    assert!(super::should_try_direct_three_small_zero_cores_root(
        &simplifier.context,
        three_core_expr,
        three_core_terms.as_slice(),
    ));
}
#[test]
fn direct_small_zero_three_core_matcher_handles_eight_term_composition_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(a^2-b^2 - (a-b)*(a+b)) + (sec(y) - 1/cos(y)) + (u*v + u*w - u*(v+w))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let terms = AddView::from_expr(&simplifier.context, expr).terms;
    assert_eq!(terms.len(), 8);
    assert_eq!(
        super::direct_small_zero_opposite_sign_exact_pair_count_root(
            &simplifier.context,
            terms.as_slice(),
        ),
        0
    );
    assert!(super::should_try_direct_three_small_zero_cores_root(
        &simplifier.context,
        expr,
        terms.as_slice(),
    ));
    assert!(
        super::matches_direct_three_small_zero_cores_terms_root(
            &mut simplifier.context,
            terms.as_slice(),
        ),
        "three-core compositions can contain eight additive terms"
    );

    let options = SimplifyOptions::default();
    let (rewritten, _steps) = super::try_standard_direct_small_zero_pair_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        options.collect_steps,
    )
    .unwrap_or_else(|| panic!("expected direct small zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn direct_small_zero_four_core_gate_requires_remaining_anchor_capacity() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let impossible_expr = parse(
        "(((a^3-b^3)/(a-b)+c) + m) - ((a^2+a*b+b^2+c) + m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let impossible_terms = AddView::from_expr(&simplifier.context, impossible_expr).terms;
    assert_eq!(impossible_terms.len(), 8);
    assert_eq!(
        super::direct_small_zero_opposite_sign_exact_pair_count_root(
            &simplifier.context,
            impossible_terms.as_slice(),
        ),
        2
    );
    assert!(
        !super::has_enough_direct_small_zero_remaining_anchor_terms_root(
            &simplifier.context,
            impossible_terms.as_slice(),
            4,
        ),
        "two exact passthrough pairs leave too few anchors for two more groups"
    );
    assert!(
        !super::matches_direct_four_or_five_small_zero_core_groups_terms_root(
            &mut simplifier.context,
            impossible_terms.as_slice(),
        )
    );

    let four_core_expr = parse(
        "(sec(a)-1/cos(a)) + (csc(b)-1/sin(b)) + (tan(c)-sin(c)/cos(c)) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let four_core_terms = AddView::from_expr(&simplifier.context, four_core_expr).terms;
    assert_eq!(four_core_terms.len(), 8);
    assert!(
        super::has_enough_direct_small_zero_remaining_anchor_terms_root(
            &simplifier.context,
            four_core_terms.as_slice(),
            4,
        )
    );
    assert!(
        super::matches_direct_four_or_five_small_zero_core_groups_terms_root(
            &mut simplifier.context,
            four_core_terms.as_slice(),
        )
    );
}
#[test]
fn direct_small_zero_additive_combination_shortcut_handles_dirichlet_vs_factor_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))) + (p^2-q^2 - (p-q)*(p+q))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = super::try_standard_direct_small_zero_additive_combination_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        true,
    )
    .unwrap_or_else(|| panic!("expected direct small zero additive combination shortcut"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn targeted_early_small_zero_additive_combination_accepts_dirichlet_vs_factor_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(sin(5*x/2)/sin(x/2) - (1 + 2*cos(x) + 2*cos(2*x))) + (p^2-q^2 - (p-q)*(p+q))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(
        super::is_targeted_early_small_zero_additive_combination_candidate_root(
            &mut simplifier.context,
            expr,
        )
    );
}
#[test]
fn direct_small_zero_profile_sample_includes_term_tags_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(ln(x^3)+ln(y^2)-ln(x^3*y^2)) + (sec(z)-1/cos(z)) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    let sample = super::render_direct_small_zero_profile_sample_root(&simplifier.context, expr);
    assert!(sample.contains("terms="), "{sample}");
    assert!(sample.contains(":log"), "{sample}");
    assert!(sample.contains(":trig"), "{sample}");
    assert!(sample.contains(":div"), "{sample}");
}
#[test]
fn direct_small_zero_pair_shortcut_handles_four_two_term_core_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    for core in [
        "1/(sqrt(a)+sqrt(b)) - (sqrt(a)-sqrt(b))/(a-b)",
        "sec(z)-1/cos(z)",
        "csc(w)-1/sin(w)",
        "1/(1 + 1/(1+u)) - (1+u)/(2+u)",
    ] {
        let core_expr =
            parse(core, &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        assert!(
            super::matches_direct_small_zero_or_known_pair_base_root(
                &mut simplifier.context,
                core_expr,
            ),
            "expected core to match: {core}"
        );
    }
    let expr = parse(
        "(1/(sqrt(a)+sqrt(b)) - (sqrt(a)-sqrt(b))/(a-b)) + (sec(z)-1/cos(z)) + (csc(w)-1/sin(w)) + (1/(1 + 1/(1+u)) - (1+u)/(2+u))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = super::try_standard_direct_small_zero_pair_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        options.collect_steps,
    )
    .unwrap_or_else(|| panic!("expected direct small zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn direct_small_zero_pair_shortcut_handles_five_two_term_core_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    for core in [
        "sec(a)-1/cos(a)",
        "csc(b)-1/sin(b)",
        "tan(c)-sin(c)/cos(c)",
        "sin(2*d)-2*sin(d)*cos(d)",
        "2*sinh(e)*cosh(e)-sinh(2*e)",
    ] {
        let core_expr = parse(core, &mut simplifier.context)
            .unwrap_or_else(|err| panic!("parse failed: {err:?}"));
        assert!(
            super::matches_direct_small_zero_or_known_pair_base_root(
                &mut simplifier.context,
                core_expr,
            ),
            "expected core to match: {core}"
        );
    }
    let expr = parse(
        "(sec(a)-1/cos(a)) + (csc(b)-1/sin(b)) + (tan(c)-sin(c)/cos(c)) + (sin(2*d)-2*sin(d)*cos(d)) + (2*sinh(e)*cosh(e)-sinh(2*e))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|err| panic!("parse failed: {err:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = super::try_standard_direct_small_zero_pair_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        options.collect_steps,
    )
    .unwrap_or_else(|| panic!("expected direct small zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn direct_small_zero_pair_shortcut_handles_five_core_sum_with_one_three_term_core_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    for core in [
        "sec(a)-1/cos(a)",
        "csc(b)-1/sin(b)",
        "tan(c)-sin(c)/cos(c)",
        "sin(2*d)-2*sin(d)*cos(d)",
        "cosh(e)^2-sinh(e)^2-1",
    ] {
        let core_expr = parse(core, &mut simplifier.context)
            .unwrap_or_else(|err| panic!("parse failed: {err:?}"));
        assert!(
            super::matches_direct_small_zero_or_known_pair_base_root(
                &mut simplifier.context,
                core_expr,
            ),
            "expected core to match: {core}"
        );
    }
    let expr = parse(
        "(sec(a)-1/cos(a)) + (csc(b)-1/sin(b)) + (tan(c)-sin(c)/cos(c)) + (sin(2*d)-2*sin(d)*cos(d)) + (cosh(e)^2-sinh(e)^2-1)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|err| panic!("parse failed: {err:?}"));
    let options = SimplifyOptions::default();
    let (rewritten, _steps) = super::try_standard_direct_small_zero_pair_shortcut(
        &options,
        &mut simplifier.context,
        expr,
        options.collect_steps,
    )
    .unwrap_or_else(|| panic!("expected direct small zero shortcut to match"));
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn isolated_simplify_rewrites_to_zero_handles_polynomial_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(isolated_simplify_rewrites_to_zero(
        &options,
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn supported_nested_zero_partner_rewrites_to_zero_handles_polynomial_partner_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((x^4 - 2*x^2*y^2 + y^4)/(x-y) - x^3 - x^2*y + x*y^2 + y^3)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(supported_nested_zero_partner_rewrites_to_zero(
        &options,
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn supported_nested_zero_child_partner_rejects_flat_nonlog_additive_noise_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr =
        parse("x + 1", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(!is_supported_nested_zero_child_partner(
        &simplifier.context,
        expr
    ));
}
#[test]
fn supported_nested_zero_partner_rewrites_to_zero_rejects_plain_division_difference_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("1/(x+1) - 1/(x-1)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = SimplifyOptions::default();
    assert!(!supported_nested_zero_partner_rewrites_to_zero(
        &options,
        &mut simplifier.context,
        expr
    ));
}
#[test]
fn supported_nested_zero_child_partner_keeps_nested_additive_cancellation_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(x - y) + (y - x)", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    assert!(is_supported_nested_zero_child_partner(
        &simplifier.context,
        expr
    ));
}
#[test]
fn simplify_pipeline_revisits_after_exact_additive_pair_chain_cancellation_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("arctan(3) + arctan(1/3) + 10 - 10", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "1/2 * pi");
}
#[test]
fn simplify_pipeline_revisits_constant_residual_after_additive_zero_cancellation_regression() {
    // Fixpoint gap caught by the dsolve O3 verification gate: the ±0
    // cancellation shortcut left a VARIABLE-FREE residual (`1/e^0 - 1`,
    // `ln(1)`, `sin(0)`, `sqrt(1) - 1`) un-folded because the resimplify
    // trigger was an enumerated shape list. A constant, non-literal
    // residual must always take the (cheap, single) re-pass.
    for (input, expected) in [
        ("(1/e^0 + 0 - 1) - 0", "0"),
        ("(ln(1) + 0) - 0", "0"),
        ("(sin(0) + 0) - 0", "0"),
        ("(sqrt(1) + 0 - 1) - 0", "0"),
        ("(1/ln(e) + 0 - 1) - 0", "0"),
        // Non-foldable constant residuals stay put (the re-pass is a
        // no-op, never a fabrication).
        ("(pi - e + 0) - 0", "pi - e"),
    ] {
        let mut simplifier = crate::Simplifier::with_default_rules();
        let expr =
            parse(input, &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
        let mut orchestrator = Orchestrator::new();
        let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
        assert_eq!(render(&simplifier.context, rewritten), expected, "{input}");
    }
    // Variable-bearing forms keep their existing path (no new trigger).
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse("(x + 0) - 0", &mut simplifier.context)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "x");
}
#[test]
fn simplify_pipeline_revisits_diff_residual_after_exact_additive_pair_cancellation_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(diff(exp(sin(x)),x)+m) - (cos(x)*e^sin(x)+m)",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    let (rewritten, _steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
}
#[test]
fn simplify_pipeline_steps_on_decomposes_partitioned_direct_small_zero_sum_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cos(x))^3 - (3*cos(x) + cos(3*x))/4) + (tan(x) + 1/tan(x) - 2/sin(2*x))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = true;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
    assert_eq!(steps.len(), 2);
}
#[test]
fn simplify_pipeline_steps_on_decomposes_full_mixed_identity_zero_chunks_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "((cos(x))^3 - (3*cos(x) + cos(3*x))/4) + (tan(x) + 1/tan(x) - 2/sin(2*x)) + (atanh((x^2 - 1)/(x^2 + 1)) - log(x)) + (sqrt(x + sqrt(x^2 - y^2)) - (sqrt(x+y) + sqrt(x-y))/sqrt(2))",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let mut orchestrator = Orchestrator::new();
    orchestrator.options.collect_steps = true;
    orchestrator.options.shared.context_mode = crate::options::ContextMode::Standard;
    orchestrator.options.shared.semantics.domain_mode = crate::DomainMode::Generic;
    let (rewritten, steps, _stats) = orchestrator.simplify_pipeline(expr, &mut simplifier);
    assert_eq!(render(&simplifier.context, rewritten), "0");
    assert_eq!(steps.len(), 4);
}
#[test]
fn small_exact_zero_leaf_guard_rejects_positive_linear_sum_without_zero_family_markers() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr =
        parse("x + y", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let options = crate::phase::SimplifyOptions::default();
    assert!(!child_is_small_exact_zero_leaf_root(
        &options,
        &mut simplifier.context,
        expr,
    ));
}
#[test]
fn small_exact_zero_leaf_prefilter_matches_child_guard_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let markerless_sum =
        parse("x + y", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let non_additive =
        parse("2*x", &mut simplifier.context).unwrap_or_else(|e| panic!("parse failed: {e:?}"));
    let trig_leaf = parse(
        "(cos(x))^3 - (3*cos(x) + cos(3*x))/4",
        &mut simplifier.context,
    )
    .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    assert!(!is_potential_small_exact_zero_leaf_root(
        &simplifier.context,
        markerless_sum,
    ));
    assert!(!is_potential_small_exact_zero_leaf_root(
        &simplifier.context,
        non_additive,
    ));
    assert!(is_potential_small_exact_zero_leaf_root(
        &simplifier.context,
        trig_leaf,
    ));
}
#[test]
fn nested_zero_direct_pair_family_candidate_gate_rejects_division_vs_common_factor_regression() {
    let mut simplifier = crate::Simplifier::with_default_rules();
    let expr = parse(
        "(((a^3-b^3)/(a-b) - (a^2 + a*b + b^2)) + 1)/((x*y + x*z - x*(y+z)) + 1)",
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

    assert!(!is_potential_nested_zero_direct_pair_family_pair_root(
        &mut simplifier.context,
        numerator_core,
        denominator_core,
    ));
}
