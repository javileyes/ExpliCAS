//! `metamorphic_simplification_tests`: familia `prove_zero`.
//!
//! Ver la cabecera de `metamorphic_simplification_tests.rs` para el contexto.

use super::*;

pub(super) fn near_numeric_guard_zero(
    ctx: &Context,
    guard: ExprId,
    var_map: &HashMap<String, f64>,
) -> bool {
    match eval_f64(ctx, guard, var_map) {
        Some(v) if v.is_finite() => v.abs() <= NUMERIC_DENOM_GUARD_ATOL,
        _ => true,
    }
}

fn expr_is_zero(ctx: &Context, expr: ExprId) -> bool {
    let zero = num_rational::BigRational::from_integer(0.into());
    matches!(ctx.get(expr), cas_ast::Expr::Number(n) if *n == zero)
}

fn prove_zero_from_diff_text(lhs: &str, rhs: &str) -> bool {
    let d_str = format!("({lhs}) - ({rhs})");
    let mut sd = Simplifier::with_default_rules();
    let Ok(dp) = parse(&d_str, &mut sd.context) else {
        return false;
    };

    let (mut dr_simp, _) = sd.simplify(dp);
    dr_simp = fold_constants_safe(&mut sd.context, dr_simp);
    if expr_is_zero(&sd.context, dr_simp) {
        return true;
    }

    let (mut dr_expand, _) = sd.expand(dp);
    dr_expand = fold_constants_safe(&mut sd.context, dr_expand);
    if expr_is_zero(&sd.context, dr_expand) {
        return true;
    }

    let (mut dr_expand_simp, _) = sd.simplify(dr_expand);
    dr_expand_simp = fold_constants_safe(&mut sd.context, dr_expand_simp);
    if expr_is_zero(&sd.context, dr_expand_simp) {
        return true;
    }

    let (mut dr_simp_expand, _) = sd.expand(dr_simp);
    dr_simp_expand = fold_constants_safe(&mut sd.context, dr_simp_expand);
    if expr_is_zero(&sd.context, dr_simp_expand) {
        return true;
    }

    let (mut dr_simp_expand_simp, _) = sd.simplify(dr_simp_expand);
    dr_simp_expand_simp = fold_constants_safe(&mut sd.context, dr_simp_expand_simp);
    expr_is_zero(&sd.context, dr_simp_expand_simp)
}

fn prove_zero_from_expanded_operands_text(lhs: &str, rhs: &str) -> bool {
    let mut sd = Simplifier::with_default_rules();
    let Ok(lhs_expr) = parse(lhs, &mut sd.context) else {
        return false;
    };
    let Ok(rhs_expr) = parse(rhs, &mut sd.context) else {
        return false;
    };

    let (mut lhs_expand, _) = sd.expand(lhs_expr);
    lhs_expand = fold_constants_safe(&mut sd.context, lhs_expand);
    let (mut rhs_expand, _) = sd.expand(rhs_expr);
    rhs_expand = fold_constants_safe(&mut sd.context, rhs_expand);

    if cas_solver::runtime::compare_expr(&sd.context, lhs_expand, rhs_expand)
        == std::cmp::Ordering::Equal
    {
        return true;
    }

    let (mut lhs_expand_simp, _) = sd.simplify(lhs_expand);
    lhs_expand_simp = fold_constants_safe(&mut sd.context, lhs_expand_simp);
    let (mut rhs_expand_simp, _) = sd.simplify(rhs_expand);
    rhs_expand_simp = fold_constants_safe(&mut sd.context, rhs_expand_simp);

    if cas_solver::runtime::compare_expr(&sd.context, lhs_expand_simp, rhs_expand_simp)
        == std::cmp::Ordering::Equal
    {
        return true;
    }

    let d = sd
        .context
        .add(cas_ast::Expr::Sub(lhs_expand_simp, rhs_expand_simp));
    let (mut ds_simp, _) = sd.simplify(d);
    ds_simp = fold_constants_safe(&mut sd.context, ds_simp);
    if expr_is_zero(&sd.context, ds_simp) {
        return true;
    }

    let (mut ds_expand, _) = sd.expand(ds_simp);
    ds_expand = fold_constants_safe(&mut sd.context, ds_expand);
    if expr_is_zero(&sd.context, ds_expand) {
        return true;
    }

    let (mut ds_expand_simp, _) = sd.simplify(ds_expand);
    ds_expand_simp = fold_constants_safe(&mut sd.context, ds_expand_simp);
    expr_is_zero(&sd.context, ds_expand_simp)
}

pub(super) fn prove_zero_from_engine_texts(lhs: &str, rhs: &str) -> bool {
    prove_zero_from_diff_text(lhs, rhs)
        || prove_zero_via_wire_eval(lhs, rhs)
        || prove_zero_from_expanded_operands_text(lhs, rhs)
}

pub(super) fn prove_zero_from_engine_texts_child_hint(lhs: &str, rhs: &str) -> bool {
    three_linear_shift_anchor_partner_identity_matches(lhs, rhs)
        || small_pow_anchor_partner_identity_matches(lhs, rhs)
        || sum_of_squares_anchor_partner_identity_matches(lhs, rhs)
        || prove_zero_via_wire_eval(lhs, rhs)
        || prove_equiv_expr_texts_fresh(lhs, rhs)
        || prove_zero_from_diff_text(lhs, rhs)
        || prove_zero_from_expanded_operands_text(lhs, rhs)
}

fn prove_zero_from_curated_text_shortcuts(lhs: &str, rhs: &str) -> bool {
    prove_zero_from_contextual_block_strategies_text(lhs, rhs)
        || prove_zero_from_curated_pair_corpus_text(lhs, rhs)
}

fn prove_zero_from_expr_texts(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    let lhs_str = expr_text(ctx, lhs);
    let rhs_str = expr_text(ctx, rhs);
    prove_zero_from_curated_pair_corpus_text(&lhs_str, &rhs_str)
        || prove_zero_from_engine_texts(&lhs_str, &rhs_str)
}

fn prove_zero_from_expr_texts_uncurated(ctx: &Context, lhs: ExprId, rhs: ExprId) -> bool {
    let lhs_str = expr_text(ctx, lhs);
    let rhs_str = expr_text(ctx, rhs);
    prove_zero_from_engine_texts(&lhs_str, &rhs_str)
}

fn prove_equiv_exprs(simplifier: &mut Simplifier, lhs: ExprId, rhs: ExprId) -> bool {
    if cas_solver::runtime::compare_expr(&simplifier.context, lhs, rhs) == std::cmp::Ordering::Equal
    {
        return true;
    }

    let mut lhs_folded = fold_constants_safe(&mut simplifier.context, lhs);
    let mut rhs_folded = fold_constants_safe(&mut simplifier.context, rhs);
    if cas_solver::runtime::compare_expr(&simplifier.context, lhs_folded, rhs_folded)
        == std::cmp::Ordering::Equal
    {
        return true;
    }
    if prove_zero_from_expr_texts(&simplifier.context, lhs_folded, rhs_folded) {
        return true;
    }

    let (lhs_simp_raw, _) = simplifier.simplify(lhs_folded);
    lhs_folded = fold_constants_safe(&mut simplifier.context, lhs_simp_raw);
    let (rhs_simp_raw, _) = simplifier.simplify(rhs_folded);
    rhs_folded = fold_constants_safe(&mut simplifier.context, rhs_simp_raw);
    if cas_solver::runtime::compare_expr(&simplifier.context, lhs_folded, rhs_folded)
        == std::cmp::Ordering::Equal
    {
        return true;
    }

    let (lhs_expand_raw, _) = simplifier.expand(lhs_folded);
    let lhs_expand = fold_constants_safe(&mut simplifier.context, lhs_expand_raw);
    let (rhs_expand_raw, _) = simplifier.expand(rhs_folded);
    let rhs_expand = fold_constants_safe(&mut simplifier.context, rhs_expand_raw);
    if cas_solver::runtime::compare_expr(&simplifier.context, lhs_expand, rhs_expand)
        == std::cmp::Ordering::Equal
    {
        return true;
    }

    let (lhs_expand_simp_raw, _) = simplifier.simplify(lhs_expand);
    let lhs_expand_simp = fold_constants_safe(&mut simplifier.context, lhs_expand_simp_raw);
    let (rhs_expand_simp_raw, _) = simplifier.simplify(rhs_expand);
    let rhs_expand_simp = fold_constants_safe(&mut simplifier.context, rhs_expand_simp_raw);
    if cas_solver::runtime::compare_expr(&simplifier.context, lhs_expand_simp, rhs_expand_simp)
        == std::cmp::Ordering::Equal
    {
        return true;
    }
    if prove_zero_from_expr_texts(&simplifier.context, lhs_expand_simp, rhs_expand_simp) {
        return true;
    }

    prove_zero_from_residual(simplifier, lhs_expand_simp, rhs_expand_simp)
}

fn prove_additive_partition_rec(
    simplifier: &mut Simplifier,
    lhs_terms: &[ExprId],
    rhs_terms: &[ExprId],
) -> bool {
    if lhs_terms.is_empty() || rhs_terms.is_empty() {
        return lhs_terms.is_empty() && rhs_terms.is_empty();
    }

    if lhs_terms.len() == 1 && rhs_terms.len() == 1 {
        return prove_equiv_exprs(simplifier, lhs_terms[0], rhs_terms[0]);
    }

    if lhs_terms.len() > 4 || rhs_terms.len() > 4 || lhs_terms.len() + rhs_terms.len() > 6 {
        return false;
    }

    let lhs_limit = 1u32 << lhs_terms.len();
    let rhs_limit = 1u32 << rhs_terms.len();
    for lhs_mask in 1..lhs_limit {
        if !mask_includes(lhs_mask, 0) {
            continue;
        }
        for rhs_mask in 1..rhs_limit {
            let lhs_group = build_group_from_mask(&mut simplifier.context, lhs_terms, lhs_mask);
            let rhs_group = build_group_from_mask(&mut simplifier.context, rhs_terms, rhs_mask);
            if !prove_equiv_exprs(simplifier, lhs_group, rhs_group) {
                continue;
            }

            let lhs_rest = filter_terms_by_mask(lhs_terms, lhs_mask, false);
            let rhs_rest = filter_terms_by_mask(rhs_terms, rhs_mask, false);
            if prove_additive_partition_rec(simplifier, &lhs_rest, &rhs_rest) {
                return true;
            }
        }
    }
    false
}

fn prove_zero_from_additive_partitions_text(lhs: &str, rhs: &str) -> bool {
    let mut simplifier = Simplifier::with_default_rules();
    let Ok(lhs_expr) = parse(lhs, &mut simplifier.context) else {
        return false;
    };
    let Ok(rhs_expr) = parse(rhs, &mut simplifier.context) else {
        return false;
    };

    let lhs_terms = collect_addends(&simplifier.context, lhs_expr);
    let rhs_terms = collect_addends(&simplifier.context, rhs_expr);
    if lhs_terms.len() < 2 || rhs_terms.is_empty() {
        return false;
    }

    prove_additive_partition_rec(&mut simplifier, &lhs_terms, &rhs_terms)
}

fn prove_zero_from_shallow_additive_partitions_text(lhs: &str, rhs: &str) -> bool {
    let mut simplifier = Simplifier::with_default_rules();
    let Ok(lhs_expr) = parse(lhs, &mut simplifier.context) else {
        return false;
    };
    let Ok(rhs_expr) = parse(rhs, &mut simplifier.context) else {
        return false;
    };

    let lhs_terms = collect_shallow_addends(&simplifier.context, lhs_expr);
    let rhs_terms = collect_shallow_addends(&simplifier.context, rhs_expr);
    if lhs_terms.len() < 2 || rhs_terms.len() < 2 {
        return false;
    }

    prove_additive_partition_rec(&mut simplifier, &lhs_terms, &rhs_terms)
}

pub(super) fn prove_equiv_expr_texts_fresh(lhs: &str, rhs: &str) -> bool {
    let mut simplifier = Simplifier::with_default_rules();
    let Ok(lhs_expr) = parse(lhs, &mut simplifier.context) else {
        return false;
    };
    let Ok(rhs_expr) = parse(rhs, &mut simplifier.context) else {
        return false;
    };
    prove_equiv_exprs(&mut simplifier, lhs_expr, rhs_expr)
}

pub(super) fn prove_zero_via_wire_eval(lhs: &str, rhs: &str) -> bool {
    let diff_expr = format!("({lhs}) - ({rhs})");
    let Ok(out) = serde_json::from_str::<Value>(&eval_str_to_wire(&diff_expr, "{}")) else {
        return false;
    };
    out.get("ok").and_then(Value::as_bool) == Some(true)
        && out.get("result").and_then(Value::as_str) == Some("0")
}

fn prove_equiv_block_texts(lhs: &str, rhs: &str) -> bool {
    prove_zero_from_diff_text(lhs, rhs)
        || prove_zero_from_expanded_operands_text(lhs, rhs)
        || prove_equiv_expr_texts_fresh(lhs, rhs)
        || prove_zero_via_wire_eval(lhs, rhs)
}

fn prove_block_pairings_rec(
    simplifier: &mut Simplifier,
    lhs_terms: &[ExprId],
    rhs_terms: &[ExprId],
    used: &mut [bool],
) -> bool {
    if lhs_terms.is_empty() {
        return true;
    }

    let lhs_head = lhs_terms[0];
    for rhs_idx in 0..rhs_terms.len() {
        if used[rhs_idx] {
            continue;
        }
        if !prove_equiv_exprs(simplifier, lhs_head, rhs_terms[rhs_idx]) {
            continue;
        }
        used[rhs_idx] = true;
        if prove_block_pairings_rec(simplifier, &lhs_terms[1..], rhs_terms, used) {
            return true;
        }
        used[rhs_idx] = false;
    }
    false
}

fn prove_zero_from_top_level_block_pairings_text(lhs: &str, rhs: &str) -> bool {
    let mut simplifier = Simplifier::with_default_rules();
    let Ok(lhs_expr) = parse(lhs, &mut simplifier.context) else {
        return false;
    };
    let Ok(rhs_expr) = parse(rhs, &mut simplifier.context) else {
        return false;
    };

    let lhs_terms = collect_shallow_addends(&simplifier.context, lhs_expr);
    let rhs_terms = collect_shallow_addends(&simplifier.context, rhs_expr);
    if lhs_terms.len() != rhs_terms.len() || !(2..=3).contains(&lhs_terms.len()) {
        return false;
    }

    let mut used = vec![false; rhs_terms.len()];
    prove_block_pairings_rec(&mut simplifier, &lhs_terms, &rhs_terms, &mut used)
}

pub(super) fn prove_zero_from_contextual_block_strategies_text(lhs: &str, rhs: &str) -> bool {
    prove_zero_from_diff_text(lhs, rhs)
        || prove_zero_from_expanded_operands_text(lhs, rhs)
        || prove_equiv_expr_texts_fresh(lhs, rhs)
        || prove_zero_from_top_level_block_pairings_text(lhs, rhs)
        || prove_zero_from_shallow_additive_partitions_text(lhs, rhs)
        || prove_zero_from_additive_partitions_text(lhs, rhs)
        || prove_zero_via_wire_eval(lhs, rhs)
}

pub(super) fn prove_zero_from_additive_abs_square_passthrough_text(lhs: &str, rhs: &str) -> bool {
    let mut simplifier = Simplifier::with_default_rules();
    let Ok(lhs_expr) = parse(lhs, &mut simplifier.context) else {
        return false;
    };
    let Ok(rhs_expr) = parse(rhs, &mut simplifier.context) else {
        return false;
    };

    let lhs_terms = collect_signed_shallow_addends(&simplifier.context, lhs_expr);
    let rhs_terms = collect_signed_shallow_addends(&simplifier.context, rhs_expr);
    if lhs_terms.len() != 2 || rhs_terms.len() != 2 {
        return false;
    }

    for lhs_idx in 0..2 {
        for rhs_idx in 0..2 {
            if lhs_terms[lhs_idx].0 != rhs_terms[rhs_idx].0 {
                continue;
            }

            let lhs_term = expr_text(&simplifier.context, lhs_terms[lhs_idx].1);
            let rhs_term = expr_text(&simplifier.context, rhs_terms[rhs_idx].1);
            if !abs_square_identity_matches(&lhs_term, &rhs_term) {
                continue;
            }

            if lhs_terms[1 - lhs_idx].0 != rhs_terms[1 - rhs_idx].0 {
                continue;
            }

            let lhs_other = expr_text(&simplifier.context, lhs_terms[1 - lhs_idx].1);
            let rhs_other = expr_text(&simplifier.context, rhs_terms[1 - rhs_idx].1);
            if prove_equiv_expr_texts_fresh(&lhs_other, &rhs_other) {
                return true;
            }
        }
    }

    false
}

pub(super) fn prove_zero_from_residual_pair_corpus_text(lhs: &str, rhs: &str) -> bool {
    let lhs = normalize_pair_text(lhs);
    let rhs = normalize_pair_text(rhs);
    if residual_pair_corpus()
        .raw
        .contains(&(lhs.clone(), rhs.clone()))
    {
        return true;
    }

    let (Some(lhs_alpha), Some(rhs_alpha)) = (
        alpha_normalize_pair_text(&lhs),
        alpha_normalize_pair_text(&rhs),
    ) else {
        return false;
    };
    residual_pair_corpus()
        .alpha
        .contains(&(lhs_alpha, rhs_alpha))
}

fn prove_zero_from_expr_variants(simplifier: &mut Simplifier, lhs: ExprId, rhs: ExprId) -> bool {
    if prove_zero_from_expr_texts(&simplifier.context, lhs, rhs) {
        return true;
    }

    let (lhs_expand_raw, _) = simplifier.expand(lhs);
    let lhs_expand = fold_constants_safe(&mut simplifier.context, lhs_expand_raw);
    let (rhs_expand_raw, _) = simplifier.expand(rhs);
    let rhs_expand = fold_constants_safe(&mut simplifier.context, rhs_expand_raw);
    if prove_zero_from_expr_texts(&simplifier.context, lhs_expand, rhs_expand) {
        return true;
    }

    let (lhs_expand_simp_raw, _) = simplifier.simplify(lhs_expand);
    let lhs_expand_simp = fold_constants_safe(&mut simplifier.context, lhs_expand_simp_raw);
    let (rhs_expand_simp_raw, _) = simplifier.simplify(rhs_expand);
    let rhs_expand_simp = fold_constants_safe(&mut simplifier.context, rhs_expand_simp_raw);
    prove_zero_from_expr_texts(&simplifier.context, lhs_expand_simp, rhs_expand_simp)
}

pub(super) fn pair_is_symbolically_proved(pair: &IdentityPair) -> bool {
    if abs_square_identity_matches(&pair.exp, &pair.simp) {
        return true;
    }
    prove_zero_from_curated_pair_corpus_text(&pair.exp, &pair.simp)
        || prove_zero_from_contextual_block_strategies_text(&pair.exp, &pair.simp)
}

pub(super) fn pair_is_raw_pressure_proved(pair: &IdentityPair) -> bool {
    if abs_square_identity_matches(&pair.exp, &pair.simp) {
        return true;
    }
    prove_zero_from_engine_texts_in_child_process(&pair.exp, &pair.simp)
}

pub(super) fn prove_zero_from_safe_window_parametrized_texts(
    lhs_text: &str,
    rhs_text: &str,
) -> bool {
    safe_window_parametrized_pair_texts(lhs_text, rhs_text).is_some()
}

fn prove_zero_from_expr_variants_with_flavor(
    simplifier: &mut Simplifier,
    lhs: ExprId,
    rhs: ExprId,
    flavor: MetamorphicProofFlavor,
) -> bool {
    let prove_expr_texts = |ctx: &Context, lhs: ExprId, rhs: ExprId| match flavor {
        MetamorphicProofFlavor::Curated => prove_zero_from_expr_texts(ctx, lhs, rhs),
        MetamorphicProofFlavor::RawPressure => prove_zero_from_expr_texts_uncurated(ctx, lhs, rhs),
    };

    if prove_expr_texts(&simplifier.context, lhs, rhs) {
        return true;
    }

    let (lhs_expand_raw, _) = simplifier.expand(lhs);
    let lhs_expand = fold_constants_safe(&mut simplifier.context, lhs_expand_raw);
    let (rhs_expand_raw, _) = simplifier.expand(rhs);
    let rhs_expand = fold_constants_safe(&mut simplifier.context, rhs_expand_raw);
    if prove_expr_texts(&simplifier.context, lhs_expand, rhs_expand) {
        return true;
    }

    let (lhs_expand_simp_raw, _) = simplifier.simplify(lhs_expand);
    let lhs_expand_simp = fold_constants_safe(&mut simplifier.context, lhs_expand_simp_raw);
    let (rhs_expand_simp_raw, _) = simplifier.simplify(rhs_expand);
    let rhs_expand_simp = fold_constants_safe(&mut simplifier.context, rhs_expand_simp_raw);
    if prove_expr_texts(&simplifier.context, lhs_expand_simp, rhs_expand_simp) {
        return true;
    }

    prove_zero_from_residual(simplifier, lhs_expand_simp, rhs_expand_simp)
}

pub(super) fn prove_zero_from_metamorphic_texts_with_flavor(
    simplifier: &mut Simplifier,
    lhs_text: &str,
    rhs_text: &str,
    lhs_simp: ExprId,
    rhs_simp: ExprId,
    flavor: MetamorphicProofFlavor,
) -> bool {
    match flavor {
        MetamorphicProofFlavor::Curated => {
            prove_zero_from_curated_text_shortcuts(lhs_text, rhs_text)
                || prove_zero_from_expr_variants_with_flavor(simplifier, lhs_simp, rhs_simp, flavor)
                || prove_zero_from_residual(simplifier, lhs_simp, rhs_simp)
        }
        // Pressure mode intentionally skips harness-level curated shortcuts.
        // It keeps only proof paths that still go through the engine itself.
        MetamorphicProofFlavor::RawPressure => {
            prove_zero_from_engine_texts(lhs_text, rhs_text)
                || prove_zero_from_expr_variants_with_flavor(simplifier, lhs_simp, rhs_simp, flavor)
                || prove_zero_from_residual(simplifier, lhs_simp, rhs_simp)
        }
    }
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI keeps the quadratic contextual representative plus dedicated sum-of-squares partner matchers"
)]
fn top_level_block_pairings_proves_multivar_plus_cubic_context() {
    let lhs = "((x^2 + y^2)*(a^2 + b^2)) + ((u+1)*(u+2)*(u+3))";
    let rhs = "((x*a + y*b)^2 + (x*b - y*a)^2) + (u^3 + 6*u^2 + 11*u + 6)";
    assert!(prove_zero_from_contextual_block_strategies_text(lhs, rhs));
}

#[test]
fn top_level_block_pairings_proves_multivar_plus_quadratic_context() {
    let lhs = "((x^2 + y^2)*(a^2 + b^2)) + ((u+2)*(u+3))";
    let rhs = "((x*a + y*b)^2 + (x*b - y*a)^2) + (u^2 + 5*u + 6)";
    assert!(prove_zero_from_contextual_block_strategies_text(lhs, rhs));
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "Debug CI keeps cheaper factor and embedded-corpus coverage for this heavy symbolic Vandermonde identity"
)]
fn engine_proves_alternating_cubic_vandermonde_identity() {
    let lhs = "a^3*(b-c) + b^3*(c-a) + c^3*(a-b)";
    let rhs = "(a-b)*(a-c)*(b-c)*(a+b+c)";
    assert!(prove_zero_from_engine_texts(lhs, rhs));
    assert!(prove_zero_from_engine_texts(rhs, lhs));
}

#[test]
fn engine_proves_sophie_germain_symbolic_identity() {
    let lhs = "x^4 + 4*y^4";
    let rhs = "(x^2 - 2*x*y + 2*y^2)*(x^2 + 2*x*y + 2*y^2)";
    assert!(prove_zero_from_engine_texts(lhs, rhs));
    assert!(prove_zero_from_engine_texts(rhs, lhs));
}

#[test]
fn engine_proves_unary_log_product_contraction_identity() {
    let lhs = "log(x^3) + log(y^2)";
    let rhs = "log(x^3*y^2)";
    assert!(prove_zero_from_engine_texts(lhs, rhs));
    assert!(prove_zero_from_engine_texts(rhs, lhs));
}

#[test]
fn engine_proves_consecutive_factorial_ratio_identity() {
    let lhs = "(n + 1)! / n!";
    let rhs = "n + 1";
    assert!(prove_zero_from_engine_texts(lhs, rhs));
    assert!(prove_zero_from_engine_texts(rhs, lhs));
}

#[test]
fn engine_proves_cos_diff_over_sin_diff_tan_identity() {
    let lhs = "(cos(x) - cos(3*x)) / (sin(3*x) - sin(x))";
    let rhs = "tan(2*x)";
    assert!(prove_zero_from_engine_texts(lhs, rhs));
    assert!(prove_zero_from_engine_texts(rhs, lhs));
}

#[test]
fn engine_proves_small_geometric_product_difference_identity() {
    let lhs = "(x - 1)*(x^5 + x^4 + x^3 + x^2 + x + 1)";
    let rhs = "x^6 - 1";
    assert!(prove_zero_from_engine_texts(lhs, rhs));
    assert!(prove_zero_from_engine_texts(rhs, lhs));
}

#[test]
fn curated_pair_corpus_proves_contextual_pair_both_directions() {
    let lhs = "(1/(x - 1) + 1/(x + 1)) + ((u+1)^2)";
    let rhs = "(2*x/(x^2 - 1)) + (u^2 + 2*u + 1)";
    assert!(prove_zero_from_curated_pair_corpus_text(lhs, rhs));
    assert!(prove_zero_from_curated_pair_corpus_text(rhs, lhs));
}

#[test]
fn curated_pair_corpus_proves_contextual_pair_with_alpha_renaming() {
    let lhs = "(1/(t - 1) + 1/(t + 1)) + ((z+1)^2)";
    let rhs = "(2*t/(t^2 - 1)) + (z^2 + 2*z + 1)";
    assert!(prove_zero_from_curated_pair_corpus_text(lhs, rhs));
    assert!(prove_zero_from_curated_pair_corpus_text(rhs, lhs));
}

fn prove_zero_from_residual(
    simplifier: &mut Simplifier,
    lhs_simp: ExprId,
    rhs_simp: ExprId,
) -> bool {
    let d = simplifier
        .context
        .add(cas_ast::Expr::Sub(lhs_simp, rhs_simp));
    let (mut ds_simp, _) = simplifier.simplify(d);
    ds_simp = fold_constants_safe(&mut simplifier.context, ds_simp);
    if expr_is_zero(&simplifier.context, ds_simp) {
        return true;
    }

    let (mut ds_expand, _) = simplifier.expand(ds_simp);
    ds_expand = fold_constants_safe(&mut simplifier.context, ds_expand);
    if expr_is_zero(&simplifier.context, ds_expand) {
        return true;
    }

    let (mut ds_expand_simp, _) = simplifier.simplify(ds_expand);
    ds_expand_simp = fold_constants_safe(&mut simplifier.context, ds_expand_simp);
    expr_is_zero(&simplifier.context, ds_expand_simp)
}

pub(super) fn top_proved_symbolic_contributors(
    metrics: &[ComboMetrics],
    limit: usize,
) -> Vec<(String, usize, usize, usize, usize)> {
    let mut rows: Vec<_> = metrics
        .iter()
        .filter_map(|m| {
            let proved = m.proved_symbolic();
            (proved > 0).then(|| {
                (
                    m.op.clone(),
                    proved,
                    m.proved_quotient,
                    m.proved_difference,
                    m.proved_composed,
                )
            })
        })
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    rows.truncate(limit);
    rows
}

#[test]
fn numeric_promotion_requires_supported_op_two_proved_sources_and_safe_cause() {
    assert!(should_promote_numeric_to_composed(
        CombineOp::Mul,
        true,
        "multivar-context"
    ));
    assert!(should_promote_numeric_to_composed(
        CombineOp::Sub,
        true,
        "multivar-context"
    ));
    assert!(should_promote_numeric_to_composed(
        CombineOp::Add,
        true,
        "multivar-context"
    ));
    assert!(should_promote_numeric_to_composed(
        CombineOp::Sub,
        true,
        "sampling-weak"
    ));
    assert!(!should_promote_numeric_to_composed(
        CombineOp::Div,
        true,
        "multivar-context"
    ));
    assert!(!should_promote_numeric_to_composed(
        CombineOp::Mul,
        false,
        "multivar-context"
    ));
    assert!(!should_promote_numeric_to_composed(
        CombineOp::Mul,
        true,
        "domain-sensitive"
    ));
}

#[test]
fn top_proved_symbolic_contributors_prefers_largest_counts_then_name() {
    let mk = |op: &str, quotient: usize, diff: usize, composed: usize| ComboMetrics {
        op: op.to_string(),
        pairs: 0,
        families: 0,
        combos: 0,
        nf_convergent: 0,
        proved_quotient: quotient,
        proved_difference: diff,
        proved_composed: composed,
        numeric_only: 0,
        inconclusive: 0,
        failed: 0,
        skipped: 0,
        timeouts: 0,
        cycle_events_total: 0,
        known_symbolic_residuals: 0,
        numeric_only_causes: HashMap::new(),
        inconclusive_causes: HashMap::new(),
        domain_frontier_examples: Vec::new(),
    };

    let top = top_proved_symbolic_contributors(
        &[
            mk("⇄ctx", 13, 0, 0),
            mk("mul", 2253, 502, 76),
            mk("⇄sub", 441, 0, 0),
            mk("sub", 104, 0, 0),
            mk("div", 253, 5, 0),
            mk("add", 129, 0, 0),
            mk("zero", 0, 0, 0),
        ],
        4,
    );

    assert_eq!(
        top,
        vec![
            ("mul".to_string(), 2831, 2253, 502, 76),
            ("⇄sub".to_string(), 441, 441, 0, 0),
            ("div".to_string(), 258, 253, 5, 0),
            ("add".to_string(), 129, 129, 0, 0),
        ]
    );
}

#[test]
fn trig_square_cube_substitution_now_proves_zero_symbolically() {
    let mut simplifier = Simplifier::with_default_rules();
    // This is a symbolic-closure tracker, not a didactic trace test. Exercise
    // the retained full-expression zero route instead of pre-simplifying both
    // sides separately, which is a known debug-only trace/proof blow-up path.
    simplifier.set_steps_mode(StepsMode::Off);
    let expr = parse(
        "(((sin(u)^2)^3 - 1)/((sin(u)^2) - 1)) - (sin(u)^4 + sin(u)^2 + 1)",
        &mut simplifier.context,
    )
    .expect("expr");

    let (diff_simp, _) = simplifier.simplify(expr);
    assert_eq!(
        format!(
            "{}",
            DisplayExpr {
                context: &simplifier.context,
                id: diff_simp
            }
        ),
        "0"
    );
}
