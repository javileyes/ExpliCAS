//! Orquestador: familia `zero_detection` (troceo P1).
//!
//! Ver la cabecera de `orchestrator.rs` para el contexto.

use super::*;

fn direct_small_zero_profile_tags_root(ctx: &Context, expr: ExprId) -> String {
    let flags = scan_hot_direct_small_zero_family_flags_root(ctx, expr);
    let mut tags = Vec::new();
    if flags.has_log {
        tags.push("log");
    }
    if flags.has_trig {
        tags.push("trig");
    }
    if flags.has_hyperbolic {
        tags.push("hyp");
    }
    if flags.has_division {
        tags.push("div");
    }
    if expr_contains_sqrt_or_half_power_local(ctx, expr) {
        tags.push("sqrt");
    }
    if expr_contains_factorial_call_local(ctx, expr) {
        tags.push("fact");
    }

    if tags.is_empty() {
        "plain".to_string()
    } else {
        tags.join("+")
    }
}

pub(super) fn render_direct_small_zero_profile_sample_root(ctx: &Context, expr: ExprId) -> String {
    match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Sub(_, _) => {
            let terms = AddView::from_expr(ctx, expr).terms;
            let mut pieces = terms
                .iter()
                .take(6)
                .map(|(term, sign)| {
                    let sign_label = match sign {
                        Sign::Pos => "+",
                        Sign::Neg => "-",
                    };
                    format!(
                        "{}{}:{}",
                        sign_label,
                        render_expr_for_orchestrator_profile(ctx, *term),
                        direct_small_zero_profile_tags_root(ctx, *term)
                    )
                })
                .collect::<Vec<_>>();
            if terms.len() > pieces.len() {
                pieces.push("...".to_string());
            }
            format!("terms={} [{}]", terms.len(), pieces.join(" "))
        }
        Expr::Mul(lhs, rhs) => format!(
            "product_pair lhs={} rhs={}",
            render_direct_small_zero_profile_sample_root(ctx, *lhs),
            render_direct_small_zero_profile_sample_root(ctx, *rhs)
        ),
        _ => render_expr_for_orchestrator_profile(ctx, expr),
    }
}

pub(super) fn is_symbolic_pow_zero_root(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return false;
    };

    is_symbolic_atom(ctx, *base) && matches!(ctx.get(*exp), Expr::Number(n) if n.is_zero())
}

pub(super) fn is_symbolic_atom_plus_nonzero_literal_root(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return false;
    };
    (is_symbolic_atom(ctx, *left) && matches!(ctx.get(*right), Expr::Number(n) if !n.is_zero()))
        || (matches!(ctx.get(*left), Expr::Number(n) if !n.is_zero())
            && is_symbolic_atom(ctx, *right))
}

pub(super) fn try_div_add_common_factor_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        || !expr_contains_division_node_local(ctx, expr)
    {
        return None;
    }

    let (lhs_core, rhs_core) =
        crate::rules::arithmetic::extract_two_term_core_difference(ctx, expr)?;

    for (quotient_side, residual_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if !expr_contains_division_node_local(ctx, quotient_side) {
            continue;
        }

        let Some(rewrite) =
            cas_math::div_add_common_factor_from_den_support::try_rewrite_div_add_common_factor_from_den_expr(
                ctx,
                quotient_side,
            )
        else {
            continue;
        };

        if compare_expr(ctx, rewrite.rewritten, residual_side) == Ordering::Equal
            || SemanticEqualityChecker::new(ctx).are_equal(rewrite.rewritten, residual_side)
        {
            return Some(ctx.num(0));
        }
    }

    None
}

pub(super) fn scan_hot_direct_small_zero_family_flags_root(
    ctx: &Context,
    root: ExprId,
) -> HotDirectSmallZeroFamilyFlags {
    let mut flags = HotDirectSmallZeroFamilyFlags::default();
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        match ctx.get(expr) {
            Expr::Div(lhs, rhs) => {
                flags.has_division = true;
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Pow(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Function(fn_id, args) => {
                if ctx.is_builtin(*fn_id, BuiltinFn::Ln)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Log)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Log10)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Abs)
                {
                    flags.has_log = true;
                } else if ctx.is_builtin(*fn_id, BuiltinFn::Sin)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Cos)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Tan)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Cot)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Sec)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Csc)
                {
                    flags.has_trig = true;
                } else if ctx.is_builtin(*fn_id, BuiltinFn::Sinh)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Cosh)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Tanh)
                {
                    flags.has_hyperbolic = true;
                }
                stack.extend(args.iter().copied());
            }
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => {}
        }

        if flags.has_log && flags.has_trig && flags.has_hyperbolic && flags.has_division {
            break;
        }
    }
    flags
}

fn expr_contains_guarded_small_zero_family_local(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_division_node_local(ctx, expr)
        || expr_contains_sqrt_or_half_power_local(ctx, expr)
        || expr_contains_factorial_call_local(ctx, expr)
}

fn expr_contains_direct_small_zero_multicore_marker_root(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
        || expr_contains_log_builtin_local(ctx, expr)
        || expr_contains_guarded_small_zero_family_local(ctx, expr)
}

pub(super) fn matches_guarded_small_zero_pair_root(
    ctx: &Context,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    (expr_contains_trig_or_hyperbolic_builtin_local(ctx, lhs)
        && expr_contains_guarded_small_zero_family_local(ctx, rhs))
        || (expr_contains_trig_or_hyperbolic_builtin_local(ctx, rhs)
            && expr_contains_guarded_small_zero_family_local(ctx, lhs))
}

pub(super) fn matches_hot_direct_small_zero_family_with_flags_root(
    ctx: &mut Context,
    expr: ExprId,
    flags: HotDirectSmallZeroFamilyFlags,
) -> bool {
    let has_log = flags.has_log;
    let has_trig = flags.has_trig;
    let has_hyperbolic = flags.has_hyperbolic;
    let has_division = flags.has_division;

    if has_log && !has_trig && !has_hyperbolic {
        return matches_direct_log_product_contract_zero_identity_root(ctx, expr)
            || matches_direct_log_square_product_split_zero_identity_root(ctx, expr)
            || matches_direct_ln_abs_product_split_zero_identity_root(ctx, expr)
            || matches_direct_log_difference_squares_split_zero_identity_root(ctx, expr);
    }

    if has_trig && !has_hyperbolic && !has_log {
        let trig_three_term = AddView::from_expr(ctx, expr).terms.len() == 3;
        return matches_direct_half_angle_square_zero_identity_root(ctx, expr)
            || matches_direct_trig_binomial_square_zero_identity_root(ctx, expr)
            || matches_direct_symbolic_trig_sum_to_product_zero_identity_root(ctx, expr)
            || (trig_three_term
                && matches_direct_trig_product_to_sum_zero_identity_root(ctx, expr))
            || (trig_three_term
                && matches_narrow_trig_mixed_double_angle_zero_candidate_root(ctx, expr)
                && matches_direct_trig_mixed_double_angle_zero_identity_root(ctx, expr))
            || matches_direct_tan_cot_sec_csc_zero_identity_root(ctx, expr)
            || matches_direct_trig_cubic_cosine_zero_identity_root(ctx, expr)
            || (is_potential_direct_three_term_phase_shift_zero_subset_root(ctx, expr)
                && (matches_direct_numeric_general_phase_shift_zero_identity_root(ctx, expr)
                    || matches_direct_three_term_phase_shift_zero_subset_root(ctx, expr)));
    }

    if has_hyperbolic && !has_trig && !has_log {
        return matches_direct_hyperbolic_exp_sum_zero_identity_root(ctx, expr)
            || matches_direct_recursive_hyperbolic_sinh_sum_zero_identity_root(ctx, expr)
            || matches_direct_recursive_hyperbolic_cosh_sum_zero_identity_root(ctx, expr)
            || matches_direct_hyperbolic_cosh_cubic_zero_identity_root(ctx, expr)
            || matches_direct_hyperbolic_pythagorean_zero_identity_root(ctx, expr)
            || matches_direct_exp_hyperbolic_double_identity_root(ctx, expr);
    }

    if has_division && !has_trig && !has_hyperbolic && !has_log {
        return matches_structural_same_denominator_distribution_zero_identity_root(ctx, expr)
            || matches_direct_affine_common_denominator_zero_identity_root(ctx, expr)
            || matches_direct_consecutive_telescoping_fraction_zero_identity_root(ctx, expr)
            || matches_direct_depth_three_unit_continued_fraction_zero_identity_root(ctx, expr)
            || matches_direct_nested_fraction_simplified_zero_identity_root(ctx, expr)
            || matches_direct_small_rational_zero_identity_root(ctx, expr)
            || extract_small_quotient_cancel_zero_candidate_root(ctx, expr).is_some();
    }

    !has_log
        && !has_trig
        && !has_hyperbolic
        && (matches_direct_sophie_germain_zero_identity_root(ctx, expr)
            || matches_direct_sqrt_perfect_square_abs_zero_identity_root(ctx, expr)
            || matches_direct_odd_half_power_zero_identity_root(ctx, expr))
}

fn matches_hot_direct_small_zero_family_root(ctx: &mut Context, expr: ExprId) -> bool {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }

    let flags = scan_hot_direct_small_zero_family_flags_root(ctx, expr);
    matches_hot_direct_small_zero_family_with_flags_root(ctx, expr, flags)
}

pub(super) fn matches_direct_small_zero_pair_root(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    matches!(ctx.get(lhs), Expr::Add(_, _) | Expr::Sub(_, _))
        && matches!(ctx.get(rhs), Expr::Add(_, _) | Expr::Sub(_, _))
        && (matches_direct_small_zero_log_split_division_hot_pair_root(ctx, lhs, rhs)
            || matches_direct_small_zero_nested_fraction_division_hot_pair_root(ctx, lhs, rhs)
            || (matches_hot_direct_small_zero_family_root(ctx, lhs)
                && matches_hot_direct_small_zero_family_root(ctx, rhs))
            || (matches_direct_small_zero_or_known_pair_base_root(ctx, lhs)
                && matches_direct_small_zero_or_known_pair_base_root(ctx, rhs)))
}

fn direct_small_zero_term_has_cancellation_marker_root(
    ctx: &Context,
    term: ExprId,
    sign: Sign,
) -> bool {
    sign == Sign::Neg || expr_contains_explicit_negative_marker_local(ctx, term)
}

pub(super) fn direct_small_zero_cancellation_marker_count_root(
    ctx: &Context,
    terms: &[(ExprId, Sign)],
) -> usize {
    terms
        .iter()
        .filter(|(term, sign)| {
            direct_small_zero_term_has_cancellation_marker_root(ctx, *term, *sign)
        })
        .count()
}

pub(super) fn direct_small_zero_opposite_sign_exact_pair_count_root(
    ctx: &Context,
    terms: &[(ExprId, Sign)],
) -> usize {
    direct_small_zero_opposite_sign_exact_pair_stats_root(ctx, terms).0
}

fn direct_small_zero_opposite_sign_exact_pair_stats_root(
    ctx: &Context,
    terms: &[(ExprId, Sign)],
) -> (usize, usize) {
    let mut paired = vec![false; terms.len()];
    let mut pair_count = 0;
    let mut non_marker_terms_in_pairs = 0;
    for left_index in 0..terms.len() {
        if paired[left_index] {
            continue;
        }
        let (left_expr, left_sign) = terms[left_index];
        for right_index in (left_index + 1)..terms.len() {
            if paired[right_index] {
                continue;
            }
            let (right_expr, right_sign) = terms[right_index];
            if left_sign != right_sign
                && compare_expr(ctx, left_expr, right_expr) == Ordering::Equal
            {
                paired[left_index] = true;
                paired[right_index] = true;
                pair_count += 1;
                if !direct_small_zero_term_has_cancellation_marker_root(ctx, left_expr, left_sign) {
                    non_marker_terms_in_pairs += 1;
                }
                if !direct_small_zero_term_has_cancellation_marker_root(ctx, right_expr, right_sign)
                {
                    non_marker_terms_in_pairs += 1;
                }
                break;
            }
        }
    }
    (pair_count, non_marker_terms_in_pairs)
}

pub(super) fn has_enough_direct_small_zero_cancellation_markers_root(
    ctx: &Context,
    terms: &[(ExprId, Sign)],
    group_count: usize,
) -> bool {
    direct_small_zero_cancellation_marker_count_root(ctx, terms) >= group_count
}

pub(super) fn has_enough_direct_small_zero_remaining_anchor_terms_root(
    ctx: &Context,
    terms: &[(ExprId, Sign)],
    group_count: usize,
) -> bool {
    let marker_count = direct_small_zero_cancellation_marker_count_root(ctx, terms);
    if marker_count < group_count {
        return false;
    }

    let non_marker_count = terms.len().saturating_sub(marker_count);
    let (exact_pair_count, paired_non_marker_count) =
        direct_small_zero_opposite_sign_exact_pair_stats_root(ctx, terms);
    let remaining_groups = group_count.saturating_sub(exact_pair_count);
    let remaining_non_markers = non_marker_count.saturating_sub(paired_non_marker_count);
    remaining_non_markers >= remaining_groups
}

pub(super) fn should_try_direct_three_small_zero_cores_root(
    ctx: &Context,
    expr: ExprId,
    terms: &[(ExprId, Sign)],
) -> bool {
    let exact_pair_count = direct_small_zero_opposite_sign_exact_pair_count_root(ctx, terms);
    if (8..=9).contains(&terms.len()) && exact_pair_count >= 2 {
        return false;
    }
    if !has_enough_direct_small_zero_remaining_anchor_terms_root(ctx, terms, 3) {
        return false;
    }

    if expr_contains_direct_small_zero_multicore_marker_root(ctx, expr) {
        return true;
    }

    exact_pair_count >= 2
}

pub(super) fn matches_direct_three_small_zero_cores_terms_root(
    ctx: &mut Context,
    terms: &[(ExprId, Sign)],
) -> bool {
    if (8..=9).contains(&terms.len()) {
        return matches_direct_small_zero_core_groups_root(ctx, terms, 3);
    }

    if !(6..=7).contains(&terms.len()) {
        return false;
    }
    if !has_enough_direct_small_zero_cancellation_markers_root(ctx, terms, 3) {
        return false;
    }

    for second_index in 1..terms.len() {
        let first_group = [terms[0], terms[second_index]];
        if matches_direct_small_zero_core_group_root(ctx, &first_group) {
            let remaining: Vec<_> = terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| (index != 0 && index != second_index).then_some(term))
                .collect();
            if matches_direct_two_small_zero_core_groups_root(ctx, &remaining) {
                return true;
            }
        }

        if terms.len() == 7 {
            for third_index in (second_index + 1)..terms.len() {
                let first_group = [terms[0], terms[second_index], terms[third_index]];
                if !matches_direct_small_zero_core_group_root(ctx, &first_group) {
                    continue;
                }

                let remaining: Vec<_> = terms
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, term)| {
                        (index != 0 && index != second_index && index != third_index)
                            .then_some(term)
                    })
                    .collect();
                if matches_direct_two_small_zero_core_groups_root(ctx, &remaining) {
                    return true;
                }
            }
        }
    }

    false
}

pub(super) fn matches_direct_four_or_five_small_zero_core_groups_terms_root(
    ctx: &mut Context,
    terms: &[(ExprId, Sign)],
) -> bool {
    if !(8..=11).contains(&terms.len()) {
        return false;
    }

    let group_count = match terms.len() {
        8 | 9 => 4,
        10 | 11 => 5,
        _ => return false,
    };
    if !has_enough_direct_small_zero_remaining_anchor_terms_root(ctx, terms, group_count) {
        return false;
    }

    matches_direct_small_zero_core_groups_root(ctx, terms, group_count)
}

fn matches_direct_small_zero_core_groups_root(
    ctx: &mut Context,
    terms: &[(ExprId, Sign)],
    group_count: usize,
) -> bool {
    if group_count == 0 {
        return terms.is_empty();
    }
    if terms.len() < group_count * 2 || terms.len() > group_count * 3 {
        return false;
    }
    if !has_enough_direct_small_zero_cancellation_markers_root(ctx, terms, group_count) {
        return false;
    }

    for second_index in 1..terms.len() {
        let first_group = [terms[0], terms[second_index]];
        if matches_direct_small_zero_core_group_root(ctx, &first_group) {
            let remaining: Vec<_> = terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| (index != 0 && index != second_index).then_some(term))
                .collect();
            if matches_direct_small_zero_core_groups_root(ctx, &remaining, group_count - 1) {
                return true;
            }
        }

        for third_index in (second_index + 1)..terms.len() {
            let first_group = [terms[0], terms[second_index], terms[third_index]];
            if !matches_direct_small_zero_core_group_root(ctx, &first_group) {
                continue;
            }

            let remaining: Vec<_> = terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index != 0 && index != second_index && index != third_index).then_some(term)
                })
                .collect();
            if matches_direct_small_zero_core_groups_root(ctx, &remaining, group_count - 1) {
                return true;
            }
        }
    }

    false
}

fn matches_direct_two_small_zero_core_groups_root(
    ctx: &mut Context,
    terms: &[(ExprId, Sign)],
) -> bool {
    if !(4..=5).contains(&terms.len()) {
        return false;
    }
    if !has_enough_direct_small_zero_cancellation_markers_root(ctx, terms, 2) {
        return false;
    }

    for second_index in 1..terms.len() {
        let first_group = [terms[0], terms[second_index]];
        if matches_direct_small_zero_core_group_root(ctx, &first_group) {
            let tail: Vec<_> = terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| (index != 0 && index != second_index).then_some(term))
                .collect();
            if matches_direct_small_zero_core_group_root(ctx, &tail) {
                return true;
            }
        }

        if terms.len() == 5 {
            for third_index in (second_index + 1)..terms.len() {
                let first_group = [terms[0], terms[second_index], terms[third_index]];
                if !matches_direct_small_zero_core_group_root(ctx, &first_group) {
                    continue;
                }

                let tail: Vec<_> = terms
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, term)| {
                        (index != 0 && index != second_index && index != third_index)
                            .then_some(term)
                    })
                    .collect();
                if matches_direct_small_zero_core_group_root(ctx, &tail) {
                    return true;
                }
            }
        }
    }

    false
}

pub(super) fn matches_direct_small_zero_core_group_root(
    ctx: &mut Context,
    terms: &[(ExprId, Sign)],
) -> bool {
    if !(2..=3).contains(&terms.len()) {
        return false;
    }

    if !has_enough_direct_small_zero_cancellation_markers_root(ctx, terms, 1) {
        return false;
    }

    let expr = build_signed_sum_expr_root(ctx, terms);
    matches_direct_small_zero_or_known_pair_base_root(ctx, expr)
}

pub(super) fn is_direct_small_zero_composition_candidate_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let profile_direct_small_zero =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let sample =
        profile_direct_small_zero.then(|| render_direct_small_zero_profile_sample_root(ctx, expr));
    macro_rules! profiled_direct_small_zero_bool {
        ($name:literal, $body:expr) => {{
            if profile_direct_small_zero {
                run_profiled_orchestrator_bool_section_with_sample($name, sample.clone(), || $body)
            } else {
                $body
            }
        }};
    }

    let direct_small_zero_shape = match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => Some((true, *lhs, *rhs)),
        Expr::Mul(lhs, rhs) => Some((false, *lhs, *rhs)),
        _ => None,
    };

    match direct_small_zero_shape {
        Some((true, lhs, rhs)) => {
            profiled_direct_small_zero_bool!(
                "root.direct_small_zero_composition.candidate.pair",
                matches_direct_small_zero_pair_root(ctx, lhs, rhs)
            ) || {
                let terms = AddView::from_expr(ctx, expr).terms;
                match terms.len() {
                    6..=9
                        if should_try_direct_three_small_zero_cores_root(
                            ctx,
                            expr,
                            terms.as_slice(),
                        ) =>
                    {
                        profiled_direct_small_zero_bool!(
                            "root.direct_small_zero_composition.candidate.three_core_groups",
                            matches_direct_three_small_zero_cores_terms_root(ctx, terms.as_slice())
                        ) || if (8..=9).contains(&terms.len())
                            && expr_contains_direct_small_zero_multicore_marker_root(ctx, expr)
                        {
                            profiled_direct_small_zero_bool!(
                                "root.direct_small_zero_composition.candidate.four_or_five_core_groups",
                                matches_direct_four_or_five_small_zero_core_groups_terms_root(
                                    ctx,
                                    terms.as_slice()
                                )
                            )
                        } else {
                            false
                        }
                    }
                    8..=11 if expr_contains_direct_small_zero_multicore_marker_root(ctx, expr) => {
                        profiled_direct_small_zero_bool!(
                            "root.direct_small_zero_composition.candidate.four_or_five_core_groups",
                            matches_direct_four_or_five_small_zero_core_groups_terms_root(
                                ctx,
                                terms.as_slice()
                            )
                        )
                    }
                    _ => false,
                }
            }
        }
        Some((false, lhs, rhs)) => profiled_direct_small_zero_bool!(
            "root.direct_small_zero_composition.candidate.product_pair",
            matches_direct_small_zero_pair_root(ctx, lhs, rhs)
        ),
        None => false,
    }
}

pub(super) fn is_guarded_small_zero_composition_candidate_root(
    ctx: &Context,
    expr: ExprId,
) -> bool {
    match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
            matches_guarded_small_zero_pair_root(ctx, *lhs, *rhs)
        }
        _ => false,
    }
}

pub(super) fn finish_common_scale_zero_shortcut_with_domain_meta(
    ctx: &mut Context,
    before: ExprId,
    parent_ctx: &crate::parent_context::ParentContext,
    collect_steps: bool,
) -> (ExprId, Vec<Step>) {
    let zero = ctx.num(0);
    let mut rewrite =
        crate::rule::Rewrite::with_local(zero, "Equivalent Residual Cancellation", before, zero);
    if let Some(event) = crate::rules::arithmetic::common_scale_abs_like_positive_assumption_event(
        ctx, before, parent_ctx,
    ) {
        rewrite = rewrite.assume(event);
    }

    finish_root_shortcut_with_rewrite_meta(
        ctx,
        before,
        rewrite,
        "Collapse Common-Scale Equivalent Difference",
        collect_steps,
    )
}

pub(super) fn matches_direct_sophie_germain_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    for (index, (term_expr, term_sign)) in view.terms.iter().copied().enumerate() {
        if term_sign != Sign::Neg {
            continue;
        }

        let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(other_index, term)| (other_index != index).then_some(term))
            .collect();
        if remaining_terms.len() != 2 || !remaining_terms.iter().all(|(_, sign)| *sign == Sign::Pos)
        {
            continue;
        }

        let compact_expr = AddView {
            root: expr,
            terms: remaining_terms,
        }
        .rebuild(ctx);
        if matches_direct_sophie_germain_pair_root(ctx, compact_expr, term_expr) {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_sophie_germain_zero_hot_candidate_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    if cas_ast::count_nodes(ctx, expr) > 48 {
        return false;
    }

    let view = AddView::from_expr(ctx, expr).terms;
    if view.len() != 3 {
        return false;
    }

    let positive_terms: smallvec::SmallVec<[ExprId; 2]> = view
        .iter()
        .filter_map(|(term, sign)| (*sign == Sign::Pos).then_some(*term))
        .collect();
    if positive_terms.len() != 2 {
        return false;
    }

    let negative_term = view
        .iter()
        .find_map(|(term, sign)| (*sign == Sign::Neg).then_some(*term));
    let Some(negative_term) = negative_term else {
        return false;
    };

    let compact_expr = build_signed_sum_expr_root(
        ctx,
        &[
            (positive_terms[0], Sign::Pos),
            (positive_terms[1], Sign::Pos),
        ],
    );
    if extract_sophie_germain_bases_root(ctx, compact_expr).is_none() {
        return false;
    }

    let product_factors = flatten_mul_chain(ctx, negative_term);
    product_factors.len() == 2
        && product_factors.iter().all(|factor| {
            matches!(ctx.get(*factor), Expr::Add(_, _) | Expr::Sub(_, _))
                && AddView::from_expr(ctx, *factor).terms.len() == 3
        })
}

pub(super) fn matches_direct_small_polynomial_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        || cas_ast::count_nodes(ctx, expr) > 24
        || expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
        || expr_contains_log_builtin_local(ctx, expr)
    {
        return false;
    }

    let policy = crate::polynomial_identity_support::PolynomialIdentityPolicy {
        max_nodes: 24,
        max_vars: 4,
        max_atoms: 0,
        var_limit: 4,
        max_scan_depth: 12,
        max_pow_exp_scan: 8,
        poly_budget: cas_math::multipoly::PolyBudget {
            max_terms: 24,
            max_total_degree: 8,
            max_pow_exp: 8,
        },
    };

    crate::polynomial_identity_support::try_prove_polynomial_identity_zero_with_policy(
        ctx, expr, &policy,
    )
    .is_some()
}

#[cfg(test)]
pub(super) fn matches_direct_geometric_difference_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    matches_geometric_difference_terms_root(ctx, &terms)
}

pub(super) fn matches_direct_small_zero_or_known_pair_residual_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    if matches_direct_small_zero_or_known_pair_base_root(ctx, expr)
        || matches_partitioned_direct_small_zero_sum_root(ctx, expr)
        || extract_partitioned_phase_shift_zero_chunks_root(ctx, expr).is_some()
    {
        return true;
    }

    false
}

fn should_defer_guarded_small_zero_additive_shortcut(ctx: &mut Context, expr: ExprId) -> bool {
    let (lhs, rhs) = match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => (lhs, rhs),
        _ => return false,
    };

    for (trig_side, guarded_side) in [(lhs, rhs), (rhs, lhs)] {
        if !expr_contains_trig_or_hyperbolic_builtin_local(ctx, trig_side)
            || !expr_contains_guarded_small_zero_family_local(ctx, guarded_side)
            || !matches!(ctx.get(trig_side), Expr::Add(_, _) | Expr::Sub(_, _))
        {
            continue;
        }

        if !matches_direct_small_zero_or_known_pair_base_root(ctx, trig_side) {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_two_factor_product_pair_zero_difference_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    fn perfect_square_pairwise_matches(
        ctx: &mut Context,
        left_a: ExprId,
        left_b: ExprId,
        right_a: ExprId,
        right_b: ExprId,
    ) -> bool {
        (matches_direct_perfect_square_trinomial_pair_root(ctx, left_a, right_a)
            && factors_match_by_equality_or_direct_pair_root(ctx, left_b, right_b))
            || (matches_direct_perfect_square_trinomial_pair_root(ctx, left_b, right_b)
                && factors_match_by_equality_or_direct_pair_root(ctx, left_a, right_a))
            || (matches_direct_perfect_square_trinomial_pair_root(ctx, left_a, right_b)
                && factors_match_by_equality_or_direct_pair_root(ctx, left_b, right_a))
            || (matches_direct_perfect_square_trinomial_pair_root(ctx, left_b, right_a)
                && factors_match_by_equality_or_direct_pair_root(ctx, left_a, right_b))
    }

    fn perfect_square_with_grouped_sum_diff_cubes_matches(
        ctx: &mut Context,
        two_factor_side: &[ExprId],
        three_factor_side: &[ExprId],
    ) -> bool {
        if two_factor_side.len() != 2 || three_factor_side.len() != 3 {
            return false;
        }

        for grouped_anchor_index in 0..three_factor_side.len() {
            let grouped_anchor = three_factor_side[grouped_anchor_index];
            let grouped_partner_factors: Vec<_> = three_factor_side
                .iter()
                .enumerate()
                .filter_map(|(index, factor)| (index != grouped_anchor_index).then_some(*factor))
                .collect();
            let grouped_partner = build_mul_expr_from_factors_root(ctx, &grouped_partner_factors);

            if (matches_direct_perfect_square_trinomial_pair_root(
                ctx,
                two_factor_side[0],
                grouped_anchor,
            ) && matches_direct_sum_diff_cubes_product_pair_root(
                ctx,
                two_factor_side[1],
                grouped_partner,
            )) || (matches_direct_perfect_square_trinomial_pair_root(
                ctx,
                two_factor_side[1],
                grouped_anchor,
            ) && matches_direct_sum_diff_cubes_product_pair_root(
                ctx,
                two_factor_side[0],
                grouped_partner,
            )) {
                return true;
            }
        }

        false
    }

    fn perfect_square_with_grouped_sophie_germain_matches(
        ctx: &mut Context,
        two_factor_side: &[ExprId],
        three_factor_side: &[ExprId],
    ) -> bool {
        if two_factor_side.len() != 2 || three_factor_side.len() != 3 {
            return false;
        }

        for grouped_anchor_index in 0..three_factor_side.len() {
            let grouped_anchor = three_factor_side[grouped_anchor_index];
            let grouped_partner_factors: Vec<_> = three_factor_side
                .iter()
                .enumerate()
                .filter_map(|(index, factor)| (index != grouped_anchor_index).then_some(*factor))
                .collect();
            let grouped_partner = build_mul_expr_from_factors_root(ctx, &grouped_partner_factors);

            if (matches_direct_perfect_square_trinomial_pair_root(
                ctx,
                two_factor_side[0],
                grouped_anchor,
            ) && matches_direct_sophie_germain_pair_root(
                ctx,
                two_factor_side[1],
                grouped_partner,
            )) || (matches_direct_perfect_square_trinomial_pair_root(
                ctx,
                two_factor_side[1],
                grouped_anchor,
            ) && matches_direct_sophie_germain_pair_root(
                ctx,
                two_factor_side[0],
                grouped_partner,
            )) {
                return true;
            }
        }

        false
    }

    fn perfect_square_with_grouped_higher_degree_difference_matches(
        ctx: &mut Context,
        two_factor_side: &[ExprId],
        five_factor_side: &[ExprId],
    ) -> bool {
        if two_factor_side.len() != 2 || five_factor_side.len() != 5 {
            return false;
        }

        for grouped_anchor_index in 0..five_factor_side.len() {
            let grouped_anchor = five_factor_side[grouped_anchor_index];
            let grouped_partner_factors: Vec<_> = five_factor_side
                .iter()
                .enumerate()
                .filter_map(|(index, factor)| (index != grouped_anchor_index).then_some(*factor))
                .collect();
            let grouped_partner = build_mul_expr_from_factors_root(ctx, &grouped_partner_factors);

            if (matches_direct_perfect_square_trinomial_pair_root(
                ctx,
                two_factor_side[0],
                grouped_anchor,
            ) && matches_direct_higher_degree_difference_pair_root(
                ctx,
                two_factor_side[1],
                grouped_partner,
            )) || (matches_direct_perfect_square_trinomial_pair_root(
                ctx,
                two_factor_side[1],
                grouped_anchor,
            ) && matches_direct_higher_degree_difference_pair_root(
                ctx,
                two_factor_side[0],
                grouped_partner,
            )) {
                return true;
            }
        }

        false
    }

    fn pairwise_matches(
        ctx: &mut Context,
        left_a: ExprId,
        left_b: ExprId,
        right_a: ExprId,
        right_b: ExprId,
    ) -> bool {
        (factors_match_by_equality_or_direct_pair_root(ctx, left_a, right_a)
            && factors_match_by_equality_or_direct_pair_root(ctx, left_b, right_b))
            || (factors_match_by_equality_or_direct_pair_root(ctx, left_a, right_b)
                && factors_match_by_equality_or_direct_pair_root(ctx, left_b, right_a))
    }

    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let (lhs_term, rhs_term) = match (view.terms[0].1, view.terms[1].1) {
        (Sign::Pos, Sign::Neg) => (view.terms[0].0, view.terms[1].0),
        (Sign::Neg, Sign::Pos) => (view.terms[1].0, view.terms[0].0),
        _ => return false,
    };

    let lhs_factors = flatten_mul_chain(ctx, lhs_term);
    let rhs_factors = flatten_mul_chain(ctx, rhs_term);

    if lhs_factors.len() == 2
        && rhs_factors.len() == 2
        && perfect_square_pairwise_matches(
            ctx,
            lhs_factors[0],
            lhs_factors[1],
            rhs_factors[0],
            rhs_factors[1],
        )
    {
        return true;
    }

    if (lhs_factors.len() == 2
        && rhs_factors.len() == 3
        && perfect_square_with_grouped_sum_diff_cubes_matches(ctx, &lhs_factors, &rhs_factors))
        || (lhs_factors.len() == 3
            && rhs_factors.len() == 2
            && perfect_square_with_grouped_sum_diff_cubes_matches(ctx, &rhs_factors, &lhs_factors))
    {
        return true;
    }

    if (lhs_factors.len() == 2
        && rhs_factors.len() == 3
        && perfect_square_with_grouped_sophie_germain_matches(ctx, &lhs_factors, &rhs_factors))
        || (lhs_factors.len() == 3
            && rhs_factors.len() == 2
            && perfect_square_with_grouped_sophie_germain_matches(ctx, &rhs_factors, &lhs_factors))
    {
        return true;
    }

    if (lhs_factors.len() == 2
        && rhs_factors.len() == 5
        && perfect_square_with_grouped_higher_degree_difference_matches(
            ctx,
            &lhs_factors,
            &rhs_factors,
        ))
        || (lhs_factors.len() == 5
            && rhs_factors.len() == 2
            && perfect_square_with_grouped_higher_degree_difference_matches(
                ctx,
                &rhs_factors,
                &lhs_factors,
            ))
    {
        return true;
    }

    if matches_direct_product_to_sum_sin_cos_factor_pair_zero_difference_root(ctx, expr) {
        return true;
    }

    let lhs_groupings = build_two_group_factorizations_root(ctx, &lhs_factors);
    let rhs_groupings = build_two_group_factorizations_root(ctx, &rhs_factors);
    if lhs_groupings.is_empty() || rhs_groupings.is_empty() {
        return false;
    }

    for (lhs_a, lhs_b) in lhs_groupings {
        for (rhs_a, rhs_b) in rhs_groupings.iter().copied() {
            if pairwise_matches(ctx, lhs_a, lhs_b, rhs_a, rhs_b) {
                return true;
            }
        }
    }

    false
}

pub(super) fn try_standard_two_factor_product_pair_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let has_common_scale = extract_common_multiplicative_residual_sum_root(ctx, expr).is_some();
    if let Some((_common_factor, residual_expr)) =
        extract_common_multiplicative_residual_sum_root(ctx, expr)
    {
        if matches_direct_small_zero_or_known_pair_residual_root(ctx, residual_expr)
            || has_structural_numeric_pythagorean_pair(ctx, residual_expr)
        {
            return None;
        }
    }

    if !matches_direct_two_factor_product_pair_zero_difference_root(ctx, expr) {
        return None;
    }

    let zero = ctx.num(0);
    let mut rewrite =
        crate::rule::Rewrite::with_local(zero, "Equivalent Product Pair Cancellation", expr, zero);
    if has_common_scale {
        let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
        if let Some(event) =
            crate::rules::arithmetic::common_scale_abs_like_positive_assumption_event(
                ctx,
                expr,
                &parent_ctx,
            )
        {
            rewrite = rewrite.assume(event);
        }
    }
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        if has_common_scale {
            "Collapse Common-Scale Equivalent Difference"
        } else {
            "Collapse Product of Equivalent Factors Difference"
        },
        collect_steps,
    ))
}

pub(super) fn try_standard_common_scale_exact_zero_shortcut_fallback(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (_common_factor, residual_expr) =
        extract_common_multiplicative_residual_sum_root(ctx, expr)?;
    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);

    if matches_direct_small_zero_or_known_pair_residual_root(ctx, residual_expr) {
        return Some(finish_common_scale_zero_shortcut_with_domain_meta(
            ctx,
            expr,
            &parent_ctx,
            collect_steps,
        ));
    }

    if try_standard_exact_zero_equivalence_shortcut(options, ctx, residual_expr, false).is_some() {
        return Some(finish_common_scale_zero_shortcut_with_domain_meta(
            ctx,
            expr,
            &parent_ctx,
            collect_steps,
        ));
    }

    if matches_direct_two_factor_product_pair_zero_difference_root(ctx, expr)
        || matches_direct_quotient_pair_zero_difference_root(ctx, expr)
    {
        return Some(finish_common_scale_zero_shortcut_with_domain_meta(
            ctx,
            expr,
            &parent_ctx,
            collect_steps,
        ));
    }

    let mut residual_simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut residual_simplifier.context, ctx);
    let mut residual_orchestrator = Orchestrator::new();
    residual_orchestrator.options = SimplifyOptions {
        collect_steps: false,
        suppress_depth_overflow_warnings: true,
        ..options.clone()
    };
    let (residual_result, _residual_steps, _stats) =
        residual_orchestrator.simplify_pipeline(residual_expr, &mut residual_simplifier);
    std::mem::swap(&mut residual_simplifier.context, ctx);

    let zero = ctx.num(0);
    if compare_expr(ctx, residual_result, zero) != Ordering::Equal {
        return None;
    }

    Some(finish_common_scale_zero_shortcut_with_domain_meta(
        ctx,
        expr,
        &parent_ctx,
        collect_steps,
    ))
}

pub(super) fn try_standard_sub_self_cancel_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    let odd_half_power_rule = crate::rules::arithmetic::ExpandOddHalfPowerToEnableCancellationRule;
    if let Some(rewrite) = crate::rule::Rule::apply(&odd_half_power_rule, ctx, expr, &parent_ctx) {
        let cancel_rule = crate::rules::arithmetic::SubSelfToZeroRule;
        let cancel_rewrite =
            crate::rule::Rule::apply(&cancel_rule, ctx, rewrite.new_expr, &parent_ctx)?;

        let result = cancel_rewrite.new_expr;
        let mut shortcut_steps = Vec::new();
        if collect_steps {
            let mut first_step = Step::with_snapshots(
                &rewrite.description,
                "Expand Odd Half Power",
                expr,
                rewrite.new_expr,
                smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
                Some(ctx),
                expr,
                rewrite.new_expr,
            );
            first_step.importance = crate::step::ImportanceLevel::High;
            {
                let meta = first_step.meta_mut();
                meta.before_local = rewrite.before_local;
                meta.after_local = rewrite.after_local;
                meta.assumption_events = rewrite.assumption_events.clone();
                meta.required_conditions = rewrite.required_conditions.clone();
                meta.poly_proof = rewrite.poly_proof.clone();
                meta.substeps = rewrite.substeps.clone();
            }
            shortcut_steps.push(first_step);

            let mut second_step = Step::with_snapshots(
                &cancel_rewrite.description,
                "Subtraction Self-Cancel",
                rewrite.new_expr,
                result,
                smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
                Some(ctx),
                rewrite.new_expr,
                result,
            );
            second_step.importance = crate::step::ImportanceLevel::High;
            {
                let meta = second_step.meta_mut();
                meta.before_local = cancel_rewrite.before_local;
                meta.after_local = cancel_rewrite.after_local;
                meta.assumption_events = cancel_rewrite.assumption_events.clone();
                meta.required_conditions = cancel_rewrite.required_conditions.clone();
                meta.poly_proof = cancel_rewrite.poly_proof.clone();
                meta.substeps = cancel_rewrite.substeps.clone();
            }
            shortcut_steps.push(second_step);
        }

        return Some((result, shortcut_steps));
    }

    let log_abs_rule = crate::rules::arithmetic::ExpandLogAbsMulDivToEnableCancellationRule;
    if let Some(rewrite) = crate::rule::Rule::apply(&log_abs_rule, ctx, expr, &parent_ctx) {
        let mut simplifier = crate::Simplifier::with_default_rules();
        std::mem::swap(&mut simplifier.context, ctx);
        let (result, inner_steps, _stats) = simplifier.simplify_with_stats(
            rewrite.new_expr,
            crate::SimplifyOptions {
                suppress_depth_overflow_warnings: true,
                ..options.clone()
            },
        );
        std::mem::swap(&mut simplifier.context, ctx);

        let zero = ctx.num(0);
        if compare_expr(ctx, result, zero) == Ordering::Equal && !inner_steps.is_empty() {
            let mut shortcut_steps = Vec::new();
            if collect_steps {
                let mut first_step = Step::with_snapshots(
                    &rewrite.description,
                    RULE_EXPAND_LOG_ABS_MUL_DIV,
                    expr,
                    rewrite.new_expr,
                    smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
                    Some(ctx),
                    expr,
                    rewrite.new_expr,
                );
                first_step.importance = crate::step::ImportanceLevel::High;
                {
                    let meta = first_step.meta_mut();
                    meta.before_local = rewrite.before_local;
                    meta.after_local = rewrite.after_local;
                    meta.assumption_events = rewrite.assumption_events.clone();
                    meta.required_conditions = rewrite.required_conditions.clone();
                    meta.poly_proof = rewrite.poly_proof.clone();
                    meta.substeps = rewrite.substeps.clone();
                }
                shortcut_steps.push(first_step);
                shortcut_steps.extend(inner_steps);
            }

            return Some((result, shortcut_steps));
        }

        if compare_expr(ctx, result, zero) == Ordering::Equal {
            let mut shortcut_steps = Vec::new();
            if collect_steps {
                let mut first_step = Step::with_snapshots(
                    &rewrite.description,
                    RULE_EXPAND_LOG_ABS_MUL_DIV,
                    expr,
                    rewrite.new_expr,
                    smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
                    Some(ctx),
                    expr,
                    rewrite.new_expr,
                );
                first_step.importance = crate::step::ImportanceLevel::High;
                {
                    let meta = first_step.meta_mut();
                    meta.before_local = rewrite.before_local;
                    meta.after_local = rewrite.after_local;
                    meta.assumption_events = rewrite.assumption_events.clone();
                    meta.required_conditions = rewrite.required_conditions.clone();
                    meta.poly_proof = rewrite.poly_proof.clone();
                    meta.substeps = rewrite.substeps.clone();
                }
                shortcut_steps.push(first_step);

                let mut second_step = Step::with_snapshots(
                    "Exact identity cancellation",
                    "Polynomial Identity",
                    rewrite.new_expr,
                    result,
                    smallvec::SmallVec::<[crate::step::PathStep; 8]>::new(),
                    Some(ctx),
                    rewrite.new_expr,
                    result,
                );
                second_step.importance = crate::step::ImportanceLevel::High;
                shortcut_steps.push(second_step);
            }

            return Some((result, shortcut_steps));
        }
    }

    let rule = crate::rules::arithmetic::SubSelfToZeroRule;
    let rewrite = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx)?;
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        rewrite,
        "Subtraction Self-Cancel",
        collect_steps,
    ))
}

pub(super) fn try_standard_abs_domain_add_sub_cancellation_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
    sticky_domain: Option<crate::ImplicitDomain>,
) -> Option<(ExprId, Vec<Step>)> {
    let mut parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    if let Some(domain) = sticky_domain {
        parent_ctx = parent_ctx.with_implicit_domain(Some(domain));
    }
    let rule = crate::rules::functions::AbsDomainAddSubCancellationRule;
    let rewrite = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx)?;
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Abs Domain Add/Sub Cancellation",
        collect_steps,
    ))
}

pub(super) fn child_isolated_exact_zero(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    child: ExprId,
) -> bool {
    if !matches!(ctx.get(child), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }

    if expr_contains_trig_or_hyperbolic_builtin_local(ctx, child) {
        let term_count = AddView::from_expr(ctx, child).terms.len();
        let allow_small_mixed_trig_additive_child = term_count <= 6
            && expr_contains_trig_builtin_local(ctx, child)
            && !expr_contains_hyperbolic_builtin_local(ctx, child)
            && !expr_contains_log_builtin_local(ctx, child);
        let allow_small_mixed_trig_hyperbolic_child = term_count <= 8
            && expr_contains_trig_builtin_local(ctx, child)
            && expr_contains_hyperbolic_builtin_local(ctx, child)
            && !expr_contains_log_builtin_local(ctx, child);
        if term_count > 4
            && !allow_small_mixed_trig_additive_child
            && !allow_small_mixed_trig_hyperbolic_child
        {
            return false;
        }

        let mut isolated_ctx = Context::new();
        let isolated_child = transplant_expr_subtree(ctx, child, &mut isolated_ctx);
        if isolated_simplify_rewrites_to_zero(options, &mut isolated_ctx, isolated_child) {
            return true;
        }

        if let Some(rewritten) =
            isolated_simplify_expr_if_changed(options, &mut isolated_ctx, isolated_child)
        {
            return isolated_simplify_rewrites_to_zero(options, &mut isolated_ctx, rewritten);
        }

        return false;
    }

    try_standard_exact_zero_equivalence_shortcut(options, ctx, child, false).is_some()
}

pub(super) fn child_matches_exact_zero_three_term_subset_rule(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    child: ExprId,
) -> bool {
    if !matches!(ctx.get(child), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }

    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, child);
    let rule = crate::rules::arithmetic::CollapseExactZeroThreeTermSubsetRule;
    let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, child, &parent_ctx) else {
        return false;
    };

    let zero = ctx.num(0);
    compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal
}

pub(super) fn is_small_exact_zero_base_family_root(ctx: &mut Context, expr: ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        && AddView::from_expr(ctx, expr).terms.len() <= 4
        && cas_ast::count_nodes(ctx, expr) <= 48
        && (matches_direct_small_zero_or_known_pair_base_root(ctx, expr)
            || extract_small_quotient_cancel_zero_candidate_root(ctx, expr).is_some())
}

fn is_potential_nested_zero_direct_pair_family_side_root(ctx: &mut Context, expr: ExprId) -> bool {
    if expr_contains_division_node_local(ctx, expr) {
        return matches!(
            ctx.get(expr),
            Expr::Div(_, _) | Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Mul(_, _)
        );
    }

    if extract_plain_sin_or_cos_arg_root(ctx, expr).is_some()
        || extract_plain_sinh_or_cosh_arg_root(ctx, expr).is_some()
        || is_plain_two_term_sin_cos_sum_or_diff_root(ctx, expr)
        || is_trig_sum_product_candidate_root(ctx, expr)
    {
        return true;
    }

    match ctx.get(expr) {
        Expr::Add(_, _) | Expr::Sub(_, _) => {
            let terms = AddView::from_expr(ctx, expr).terms;
            terms.len() <= 4 && expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
        }
        Expr::Mul(_, _) => {
            flatten_mul_chain(ctx, expr).len() <= 4
                && expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
        }
        Expr::Pow(base, exponent) => {
            extract_i64_integer(ctx, *exponent) == Some(2)
                && matches!(ctx.get(*base), Expr::Add(_, _) | Expr::Sub(_, _))
                && expr_contains_trig_builtin_local(ctx, *base)
        }
        _ => false,
    }
}

pub(super) fn is_potential_nested_zero_direct_pair_family_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let lhs_has_division = expr_contains_division_node_local(ctx, lhs_core);
    let rhs_has_division = expr_contains_division_node_local(ctx, rhs_core);
    if lhs_has_division || rhs_has_division {
        return lhs_has_division && rhs_has_division;
    }

    is_potential_nested_zero_direct_pair_family_side_root(ctx, lhs_core)
        && is_potential_nested_zero_direct_pair_family_side_root(ctx, rhs_core)
}

pub(super) fn child_matches_exact_zero_common_scale_known_residual(
    ctx: &mut Context,
    child: ExprId,
) -> bool {
    if let Some((_common_factor, residual_expr)) =
        extract_common_multiplicative_residual_sum_root(ctx, child)
    {
        if matches_direct_small_zero_or_known_pair_residual_root(ctx, residual_expr) {
            return true;
        }
    }

    false
}

pub(super) fn child_matches_exact_zero_common_scale_rule(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    child: ExprId,
) -> bool {
    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, child);
    let rule = crate::rules::arithmetic::CollapseExactZeroCommonScaledDifferenceRule;
    let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, child, &parent_ctx) else {
        return false;
    };

    let zero = ctx.num(0);
    compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal
}

pub(super) fn child_matches_direct_or_isolated_exact_zero(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    child: ExprId,
) -> bool {
    matches_direct_small_zero_identity_root(ctx, child)
        || matches_direct_hyperbolic_cosh_cubic_zero_identity_root(ctx, child)
        || try_standard_small_trig_zero_pair_shortcut(options, ctx, child, false).is_some()
        || child_isolated_exact_zero(options, ctx, child)
}

fn is_plausible_nonlog_additive_zero_partner(ctx: &Context, expr: ExprId) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() < 2 {
        return false;
    }

    let has_positive = terms.iter().any(|(_, sign)| *sign == Sign::Pos);
    let has_negative = terms.iter().any(|(_, sign)| *sign == Sign::Neg);
    if has_positive && has_negative {
        return true;
    }

    terms.iter().any(|(term, _)| {
        matches!(
            ctx.get(*term),
            Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_)
        ) || expr_contains_explicit_negative_marker_local(ctx, *term)
    })
}

pub(super) fn is_supported_nonlog_additive_nested_zero_child_partner(
    ctx: &Context,
    expr: ExprId,
) -> bool {
    matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        && !expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
        && !expr_contains_log_builtin_local(ctx, expr)
        && is_plausible_nonlog_additive_zero_partner(ctx, expr)
}

fn is_supported_nested_exact_zero_child_anchor(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
        || expr_contains_log_builtin_local(ctx, expr)
}

pub(super) fn supported_nested_zero_child_partner_profile_family(
    ctx: &Context,
    expr: ExprId,
) -> &'static str {
    if expr_contains_log_builtin_local(ctx, expr) {
        "log"
    } else if is_supported_nonlog_additive_nested_zero_child_partner(ctx, expr) {
        "nonlog_additive"
    } else {
        "other"
    }
}

pub(super) fn supported_nested_zero_partner_try_profile_label(
    family: &'static str,
    stage: &'static str,
) -> Option<&'static str> {
    match (family, stage) {
        ("log", "direct_small_zero") => Some("root.nested_zero.partner.log.try.direct_small_zero"),
        ("log", "symbolic_root_denesting") => {
            Some("root.nested_zero.partner.log.try.symbolic_root_denesting")
        }
        ("log", "atanh_square_ratio_log") => {
            Some("root.nested_zero.partner.log.try.atanh_square_ratio_log")
        }
        ("log", "exact_zero_equivalence") => {
            Some("root.nested_zero.partner.log.try.exact_zero_equivalence")
        }
        ("log", "isolated_simplify") => Some("root.nested_zero.partner.log.try.isolated_simplify"),
        ("nonlog_additive", "direct_small_zero") => {
            Some("root.nested_zero.partner.nonlog_additive.try.direct_small_zero")
        }
        ("nonlog_additive", "symbolic_root_denesting") => {
            Some("root.nested_zero.partner.nonlog_additive.try.symbolic_root_denesting")
        }
        ("nonlog_additive", "atanh_square_ratio_log") => {
            Some("root.nested_zero.partner.nonlog_additive.try.atanh_square_ratio_log")
        }
        ("nonlog_additive", "exact_zero_equivalence") => {
            Some("root.nested_zero.partner.nonlog_additive.try.exact_zero_equivalence")
        }
        ("nonlog_additive", "isolated_simplify") => {
            Some("root.nested_zero.partner.nonlog_additive.try.isolated_simplify")
        }
        _ => None,
    }
}

pub(super) fn should_try_supported_nested_zero_partner_isolated_simplify(
    ctx: &Context,
    expr: ExprId,
) -> bool {
    let node_count = cas_ast::count_nodes(ctx, expr);
    if is_plain_division_difference_root(ctx, expr) {
        return false;
    }

    if node_count <= 48 && !expr_contains_sqrt_or_half_power_local(ctx, expr) {
        return true;
    }

    node_count <= 96
        && (is_supported_nonlog_additive_nested_zero_child_partner(ctx, expr)
            || (expr_contains_log_builtin_local(ctx, expr)
                && expr_contains_sqrt_or_half_power_local(ctx, expr)))
}

pub(super) fn is_potential_direct_small_zero_identity_root(ctx: &Context, expr: ExprId) -> bool {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }

    let terms = AddView::from_expr(ctx, expr).terms;
    if !(2..=4).contains(&terms.len()) {
        return false;
    }

    let has_positive = terms.iter().any(|(_, sign)| *sign == Sign::Pos);
    let has_negative = terms.iter().any(|(_, sign)| *sign == Sign::Neg);
    has_positive
        && has_negative
        && terms.iter().any(|(term, _)| {
            matches!(
                ctx.get(*term),
                Expr::Mul(_, _) | Expr::Div(_, _) | Expr::Pow(_, _) | Expr::Function(_, _)
            )
        })
}

pub(super) fn try_standard_guarded_small_zero_pair_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !is_guarded_small_zero_composition_candidate_root(ctx, expr) {
        return None;
    }
    if should_defer_guarded_small_zero_additive_shortcut(ctx, expr) {
        return None;
    }

    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    match ctx.get(expr) {
        Expr::Mul(_, _) => {
            let rule = crate::rules::arithmetic::CollapseExactZeroProductFactorRule;
            let rewrite = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx)?;
            Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Zero Product via Exact Residual",
                collect_steps,
            ))
        }
        Expr::Add(_, _) | Expr::Sub(_, _) => {
            let rule = crate::rules::arithmetic::CollapseExactZeroThreeTermSubsetRule;
            let rewrite = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx)?;
            Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ))
        }
        _ => None,
    }
}

pub(super) fn try_standard_direct_small_zero_pair_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !is_direct_small_zero_composition_candidate_root(ctx, expr) {
        return None;
    }

    match ctx.get(expr) {
        Expr::Mul(_, _) | Expr::Add(_, _) | Expr::Sub(_, _) => {
            let zero = ctx.num(0);
            let rule_name = exact_zero_leaf_rule_name_root(ctx, expr);
            let rewrite = crate::rule::Rewrite::with_local(zero, rule_name, expr, zero);
            Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                rule_name,
                collect_steps,
            ))
        }
        _ => None,
    }
}

pub(super) fn narrow_known_pair_subset_remaining_rewrites_to_zero(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    remaining_expr: ExprId,
) -> bool {
    matches_direct_small_zero_identity_root(ctx, remaining_expr)
        || try_standard_sqrt_perfect_square_abs_subset_zero_shortcut(
            options,
            ctx,
            remaining_expr,
            false,
        )
        .is_some()
        || try_standard_atanh_square_ratio_log_subset_zero_shortcut(
            options,
            ctx,
            remaining_expr,
            false,
        )
        .is_some()
        || (cas_ast::count_nodes(ctx, remaining_expr) <= 48
            && isolated_simplify_rewrites_to_zero(options, ctx, remaining_expr))
}

fn try_extract_symbolic_root_denesting_subset_zero_chunks_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let view = AddView::from_expr(ctx, expr);
    if !(4..=17).contains(&view.terms.len()) {
        return None;
    }
    if !expr_contains_sqrt_or_half_power_local(ctx, expr) {
        return None;
    }
    let mut sqrt_like_terms = 0usize;
    let mut saw_positive_sqrt_like = false;
    let mut saw_negative_sqrt_like = false;
    for (term_expr, term_sign) in view.terms.iter().copied() {
        if expr_contains_sqrt_or_half_power_local(ctx, term_expr) {
            sqrt_like_terms += 1;
            match term_sign {
                Sign::Pos => saw_positive_sqrt_like = true,
                Sign::Neg => saw_negative_sqrt_like = true,
            }
        }
    }
    if sqrt_like_terms != 2 || !saw_positive_sqrt_like || !saw_negative_sqrt_like {
        return None;
    }
    let normalized_terms: smallvec::SmallVec<[(ExprId, Sign); 16]> = view
        .terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| normalize_signed_add_term_root(ctx, term_expr, term_sign))
        .collect();

    for first_index in 0..normalized_terms.len().saturating_sub(1) {
        for second_index in (first_index + 1)..normalized_terms.len() {
            let subset_terms = [
                normalized_terms[first_index],
                normalized_terms[second_index],
            ];
            let subset_expr = build_signed_sum_expr_root(ctx, &subset_terms);
            if !crate::rules::arithmetic::matches_direct_symbolic_root_denesting_zero_identity(
                ctx,
                subset_expr,
            ) {
                continue;
            }

            let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index != first_index && index != second_index)
                        .then_some(normalize_signed_add_term_root(ctx, term.0, term.1))
                })
                .collect();
            if !(2..=15).contains(&remaining_terms.len()) {
                continue;
            }

            let remaining_expr = AddView {
                root: expr,
                terms: remaining_terms,
            }
            .rebuild(ctx);
            if expr_contains_sqrt_or_half_power_local(ctx, remaining_expr) {
                continue;
            }

            let remaining_rewrites_to_zero =
                supported_nested_zero_partner_rewrites_to_zero(options, ctx, remaining_expr)
                    || try_standard_multiterm_trig_numeric_subset_zero_shortcut(
                        options,
                        ctx,
                        remaining_expr,
                        false,
                    )
                    .is_some()
                    || try_standard_exact_zero_equivalence_shortcut(
                        options,
                        ctx,
                        remaining_expr,
                        false,
                    )
                    .is_some()
                    || (cas_ast::count_nodes(ctx, remaining_expr) <= 48
                        && isolated_simplify_rewrites_to_zero(options, ctx, remaining_expr));
            if !remaining_rewrites_to_zero {
                continue;
            }

            return Some((subset_expr, remaining_expr));
        }
    }

    None
}

pub(super) fn try_standard_symbolic_root_denesting_subset_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (subset_expr, remaining_expr) =
        try_extract_symbolic_root_denesting_subset_zero_chunks_root(options, ctx, expr)?;
    let zero = ctx.num(0);
    if collect_steps {
        if let Some(steps) = try_build_chunk_pair_zero_shortcut_steps_root(
            options,
            ctx,
            expr,
            subset_expr,
            remaining_expr,
        ) {
            return Some((zero, steps));
        }
    }
    Some(run_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        zero,
        collect_steps,
    ))
}

pub(super) fn try_standard_nested_exact_zero_child_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (zero_chunk, rewritten) =
        try_extract_nested_exact_zero_child_chunks_root(options, ctx, expr)?;

    if collect_steps {
        let zero = ctx.num(0);
        if let Some(steps) =
            try_build_chunk_pair_zero_shortcut_steps_root(options, ctx, expr, zero_chunk, rewritten)
        {
            return Some((zero, steps));
        }
    }

    Some(run_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        rewritten,
        collect_steps,
    ))
}

fn try_extract_nested_exact_zero_child_chunks_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    if matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        let add_term_count = AddView::from_expr(ctx, expr).terms.len();
        if add_term_count >= 8 && expr_contains_log_builtin_local(ctx, expr) {
            return None;
        }
    }

    let is_supported_zero_partner = |ctx: &mut Context, partner: ExprId| {
        is_supported_nested_zero_child_partner(ctx, partner)
            || (!expr_contains_trig_or_hyperbolic_builtin_local(ctx, partner)
                && child_matches_direct_or_isolated_exact_zero(options, ctx, partner))
    };

    let rewritten = match ctx.get(expr) {
        Expr::Add(lhs, rhs) => {
            let lhs = *lhs;
            let rhs = *rhs;
            if is_supported_nested_exact_zero_child_anchor(ctx, lhs)
                && is_supported_zero_partner(ctx, rhs)
                && child_matches_direct_or_isolated_exact_zero(options, ctx, lhs)
            {
                Some((lhs, rhs))
            } else if is_supported_nested_exact_zero_child_anchor(ctx, rhs)
                && is_supported_zero_partner(ctx, lhs)
                && child_matches_direct_or_isolated_exact_zero(options, ctx, rhs)
            {
                Some((rhs, lhs))
            } else {
                None
            }
        }
        Expr::Sub(lhs, rhs) => {
            let lhs = *lhs;
            let rhs = *rhs;
            if is_supported_nested_exact_zero_child_anchor(ctx, rhs)
                && is_supported_zero_partner(ctx, lhs)
                && child_matches_direct_or_isolated_exact_zero(options, ctx, rhs)
            {
                Some((rhs, lhs))
            } else if is_supported_nested_exact_zero_child_anchor(ctx, lhs)
                && is_supported_zero_partner(ctx, rhs)
                && child_matches_direct_or_isolated_exact_zero(options, ctx, lhs)
            {
                Some((lhs, ctx.add(Expr::Neg(rhs))))
            } else {
                None
            }
        }
        _ => None,
    };

    rewritten
}

pub(super) fn try_standard_zero_product_with_exact_zero_child_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let Expr::Mul(lhs, rhs) = ctx.get(expr) else {
        return None;
    };
    let lhs = *lhs;
    let rhs = *rhs;

    let child_matches_direct_base_zero = |ctx: &mut Context, child: ExprId| {
        matches!(ctx.get(child), Expr::Add(_, _) | Expr::Sub(_, _))
            && matches_direct_small_zero_or_known_pair_base_root(ctx, child)
    };

    let child_matches_small_exact_zero_leaf = |ctx: &mut Context, child: ExprId| {
        matches!(ctx.get(child), Expr::Add(_, _) | Expr::Sub(_, _))
            && child_is_small_exact_zero_leaf_root(options, ctx, child)
    };

    let allow_small_zero_leaf_pair_shortcut =
        matches!(ctx.get(lhs), Expr::Add(_, _) | Expr::Sub(_, _))
            && matches!(ctx.get(rhs), Expr::Add(_, _) | Expr::Sub(_, _))
            && (expr_contains_trig_or_hyperbolic_builtin_local(ctx, lhs)
                || expr_contains_trig_or_hyperbolic_builtin_local(ctx, rhs)
                || expr_contains_log_builtin_local(ctx, lhs)
                || expr_contains_log_builtin_local(ctx, rhs));

    let lhs_small_exact_zero_leaf =
        allow_small_zero_leaf_pair_shortcut && child_matches_small_exact_zero_leaf(ctx, lhs);
    let rhs_small_exact_zero_leaf =
        allow_small_zero_leaf_pair_shortcut && child_matches_small_exact_zero_leaf(ctx, rhs);
    let lhs_zero_candidate = lhs_small_exact_zero_leaf
        || ((expr_contains_trig_or_hyperbolic_builtin_local(ctx, lhs)
            && child_matches_direct_or_isolated_exact_zero(options, ctx, lhs))
            || child_matches_direct_base_zero(ctx, lhs));
    let rhs_zero_candidate = rhs_small_exact_zero_leaf
        || ((expr_contains_trig_or_hyperbolic_builtin_local(ctx, rhs)
            && child_matches_direct_or_isolated_exact_zero(options, ctx, rhs))
            || child_matches_direct_base_zero(ctx, rhs));
    let lhs_partner_supported =
        lhs_small_exact_zero_leaf || is_supported_nested_zero_child_partner(ctx, lhs);
    let rhs_partner_supported =
        rhs_small_exact_zero_leaf || is_supported_nested_zero_child_partner(ctx, rhs);

    let zero_factor = if rhs_partner_supported && lhs_zero_candidate {
        Some(lhs)
    } else if lhs_partner_supported && rhs_zero_candidate {
        Some(rhs)
    } else {
        None
    }?;

    let zero = ctx.num(0);
    let mut shortcut_steps = Vec::new();
    if collect_steps {
        let mut step = Step::new_compact(
            "Collapse Exact Zero Additive Subexpression",
            "Collapse Exact Zero Additive Subexpression",
            expr,
            zero,
        );
        step.global_before = Some(expr);
        step.global_after = Some(zero);
        step.importance = crate::step::ImportanceLevel::High;
        shortcut_steps.push(step);
        let sibling = if compare_expr(ctx, zero_factor, lhs) == Ordering::Equal {
            rhs
        } else {
            lhs
        };
        shortcut_steps.push(build_root_shortcut_compact_step(
            ctx.add(Expr::Mul(zero, sibling)),
            zero,
            "Cualquier producto con un factor 0 vale 0",
            "Producto por cero",
        ));
    }

    Some((zero, shortcut_steps))
}

pub(super) fn try_extract_partitioned_direct_small_zero_sum_chunks_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    if matches_direct_pythagorean_extended_zero_identity_root(ctx, expr) {
        return None;
    }

    let terms = AddView::from_expr(ctx, expr).terms;
    if !(3..=8).contains(&terms.len()) {
        return None;
    }

    build_small_two_chunk_additive_partitions_root(ctx, &terms)
        .into_iter()
        .find(|(lhs_chunk, rhs_chunk)| {
            matches_direct_small_zero_or_known_pair_base_root(ctx, *lhs_chunk)
                && matches_direct_small_zero_or_known_pair_base_root(ctx, *rhs_chunk)
        })
}

pub(super) fn try_extract_partitioned_exact_zero_leaf_chunks_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    if matches_direct_pythagorean_extended_zero_identity_root(ctx, expr) {
        return None;
    }

    let terms = AddView::from_expr(ctx, expr).terms;
    if !(3..=8).contains(&terms.len()) {
        return None;
    }

    build_small_two_chunk_additive_partitions_root(ctx, &terms)
        .into_iter()
        .find(|(lhs_chunk, rhs_chunk)| {
            exact_zero_leaf_rewrites_to_zero_root(options, ctx, *lhs_chunk)
                && exact_zero_leaf_rewrites_to_zero_root(options, ctx, *rhs_chunk)
        })
}

fn matches_partitioned_direct_small_zero_sum_root(ctx: &mut Context, expr: ExprId) -> bool {
    try_extract_partitioned_direct_small_zero_sum_chunks_root(ctx, expr).is_some()
}

pub(super) fn try_standard_partitioned_direct_small_zero_sum_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        || extract_shared_additive_passthrough_sub_cores_root(ctx, expr).is_some()
    {
        return None;
    }

    let (lhs_chunk, rhs_chunk) = if collect_steps {
        try_extract_partitioned_exact_zero_leaf_chunks_root(options, ctx, expr)?
    } else {
        try_extract_partitioned_direct_small_zero_sum_chunks_root(ctx, expr)?
    };

    if collect_steps {
        let zero = ctx.num(0);
        let mut first_step = build_root_shortcut_compact_step(
            lhs_chunk,
            zero,
            "Collapse Exact Zero Additive Subexpression",
            "Collapse Exact Zero Additive Subexpression",
        );
        first_step.global_before = Some(expr);
        first_step.global_after = Some(rhs_chunk);

        let mut second_step = build_root_shortcut_compact_step(
            rhs_chunk,
            zero,
            "Collapse Exact Zero Additive Subexpression",
            "Collapse Exact Zero Additive Subexpression",
        );
        second_step.global_before = Some(rhs_chunk);
        second_step.global_after = Some(zero);
        return Some((zero, vec![first_step, second_step]));
    }

    let zero = ctx.num(0);
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero),
        "Collapse Exact Zero Additive Subexpression",
        collect_steps,
    ))
}

pub(super) fn merge_additive_zero_chunk_residual_root(
    ctx: &mut Context,
    first_residual: ExprId,
    other_chunk: ExprId,
) -> ExprId {
    let mut terms = smallvec::SmallVec::<[(ExprId, Sign); 8]>::new();
    extend_additive_terms_from_expr_root(ctx, first_residual, &mut terms);
    extend_additive_terms_from_expr_root(ctx, other_chunk, &mut terms);
    if terms.is_empty() {
        ctx.num(0)
    } else {
        build_signed_sum_expr_root(ctx, &terms)
    }
}

pub(super) fn exact_zero_leaf_rewrites_to_zero_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let contains_hyperbolic = expr_contains_hyperbolic_builtin_local(ctx, expr);
    let allow_small_isolated_fallback = matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        && !contains_hyperbolic
        && AddView::from_expr(ctx, expr).terms.len() <= 4
        && cas_ast::count_nodes(ctx, expr) <= 48;
    matches_direct_small_zero_or_known_pair_base_root(ctx, expr)
        || crate::rules::hyperbolic::try_build_atanh_square_ratio_log_zero_rewrite(ctx, expr)
            .is_some()
        || crate::rules::arithmetic::matches_direct_symbolic_root_denesting_zero_identity(ctx, expr)
        || matches_small_quotient_cancel_zero_identity_root(options, ctx, expr)
        || try_standard_direct_small_zero_identity_shortcut(options, ctx, expr, false).is_some()
        || (!contains_hyperbolic
            && try_standard_exact_zero_equivalence_shortcut(options, ctx, expr, false).is_some())
        || (allow_small_isolated_fallback && child_isolated_exact_zero(options, ctx, expr))
}

fn try_extract_recursive_additive_zero_chunks_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    try_extract_partitioned_exact_zero_leaf_chunks_root(options, ctx, expr)
        .or_else(|| try_extract_multiterm_trig_numeric_subset_zero_chunks_root(options, ctx, expr))
        .or_else(|| try_extract_symbolic_root_denesting_subset_zero_chunks_root(options, ctx, expr))
        .or_else(|| try_extract_atanh_square_ratio_log_subset_zero_chunks_root(options, ctx, expr))
        .or_else(|| try_extract_nested_exact_zero_child_chunks_root(options, ctx, expr))
}

fn peel_first_recursive_additive_zero_leaf_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    if let Some((lhs_chunk, rhs_chunk)) =
        try_extract_recursive_additive_zero_chunks_root(options, ctx, expr)
    {
        let lhs_term_count = AddView::from_expr(ctx, lhs_chunk).terms.len();
        let rhs_term_count = AddView::from_expr(ctx, rhs_chunk).terms.len();
        let preferred_first = if rhs_term_count < lhs_term_count {
            [(rhs_chunk, lhs_chunk), (lhs_chunk, rhs_chunk)]
        } else {
            [(lhs_chunk, rhs_chunk), (rhs_chunk, lhs_chunk)]
        };

        for (first_chunk, other_chunk) in preferred_first {
            if let Some((leaf_chunk, first_residual)) =
                peel_first_recursive_additive_zero_leaf_root(options, ctx, first_chunk)
            {
                let residual_expr =
                    merge_additive_zero_chunk_residual_root(ctx, first_residual, other_chunk);
                let zero = ctx.num(0);
                if compare_expr(ctx, residual_expr, zero) == Ordering::Equal
                    || exact_zero_leaf_rewrites_to_zero_root(options, ctx, residual_expr)
                    || try_extract_recursive_additive_zero_chunks_root(options, ctx, residual_expr)
                        .is_some()
                {
                    return Some((leaf_chunk, residual_expr));
                }
            }
        }
    }

    exact_zero_leaf_rewrites_to_zero_root(options, ctx, expr).then(|| (expr, ctx.num(0)))
}

pub(super) fn try_build_recursive_additive_zero_shortcut_steps(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<Step>> {
    let zero = ctx.num(0);
    let mut steps = Vec::new();
    let mut current = expr;

    loop {
        let (leaf_chunk, residual_expr) =
            peel_first_recursive_additive_zero_leaf_root(options, ctx, current)?;
        let mut step = build_root_shortcut_compact_step(
            leaf_chunk,
            zero,
            "Collapse Exact Zero Additive Subexpression",
            "Collapse Exact Zero Additive Subexpression",
        );
        step.global_before = Some(current);
        step.global_after = Some(residual_expr);
        steps.push(step);

        if compare_expr(ctx, residual_expr, zero) == Ordering::Equal {
            return Some(steps);
        }
        current = residual_expr;
    }
}

fn exact_zero_leaf_rule_name_root(ctx: &mut Context, expr: ExprId) -> &'static str {
    if matches_direct_symbolic_trig_sum_to_product_zero_identity_root(ctx, expr) {
        return "Aplicar suma a producto";
    }

    if matches_direct_half_angle_square_zero_identity_root(ctx, expr) {
        return "Aplicar identidad de ángulo mitad";
    }

    if matches_direct_general_phase_shift_zero_identity_root(ctx, expr) {
        return "Aplicar identidad de desfase";
    }

    if matches_small_quotient_cancel_zero_identity_root(
        &crate::phase::SimplifyOptions::default(),
        ctx,
        expr,
    ) {
        return "Restar fracciones y cancelar términos iguales";
    }

    if !expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
        && contains_direct_log_cancellation_zero_group_root(ctx, expr)
    {
        return "Expandir logaritmos y cancelar términos iguales";
    }

    "Collapse Exact Zero Additive Subexpression"
}

fn build_exact_zero_leaf_shortcut_steps_root(ctx: &mut Context, expr: ExprId) -> Vec<Step> {
    let zero = ctx.num(0);
    let rule_name = exact_zero_leaf_rule_name_root(ctx, expr);
    let mut step = build_root_shortcut_compact_step(expr, zero, rule_name, rule_name);
    step.global_before = Some(expr);
    step.global_after = Some(zero);
    vec![step]
}

pub(super) fn build_recursive_or_leaf_zero_chunk_steps_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> Option<Vec<Step>> {
    try_build_recursive_additive_zero_shortcut_steps(options, ctx, expr).or_else(|| {
        exact_zero_leaf_rewrites_to_zero_root(options, ctx, expr)
            .then(|| build_exact_zero_leaf_shortcut_steps_root(ctx, expr))
    })
}

fn try_extract_exact_zero_subset_passthrough_residual_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let view = AddView::from_expr(ctx, expr);
    if !(5..=8).contains(&view.terms.len()) {
        return None;
    }
    if !expr_contains_trig_builtin_local(ctx, expr) || !expr_contains_log_builtin_local(ctx, expr) {
        return None;
    }

    let mut log_like_terms = smallvec::SmallVec::<[(ExprId, Sign); 4]>::new();
    let mut trig_numeric_terms = smallvec::SmallVec::<[(ExprId, Sign); 4]>::new();
    let mut remaining_terms = smallvec::SmallVec::<[(ExprId, Sign); 8]>::new();
    for term in view.terms.iter().copied() {
        let is_numeric_term = matches!(ctx.get(term.0), Expr::Number(_));
        let contains_log_like = expr_contains_any_builtin_local(
            ctx,
            term.0,
            &[
                BuiltinFn::Ln,
                BuiltinFn::Log,
                BuiltinFn::Log10,
                BuiltinFn::Atanh,
            ],
        );
        if contains_log_like {
            log_like_terms.push(term);
        } else if expr_contains_trig_builtin_local(ctx, term.0) || is_numeric_term {
            trig_numeric_terms.push(term);
        } else {
            remaining_terms.push(term);
        }
    }

    if log_like_terms.len() != 2 || !(1..=2).contains(&remaining_terms.len()) {
        return None;
    }

    let trig_term_count = trig_numeric_terms
        .iter()
        .filter(|(term_expr, _)| expr_contains_trig_builtin_local(ctx, *term_expr))
        .count();
    let numeric_term_count = trig_numeric_terms
        .iter()
        .filter(|(term_expr, _)| matches!(ctx.get(*term_expr), Expr::Number(_)))
        .count();
    if trig_term_count != 2 || numeric_term_count != 1 || trig_numeric_terms.len() != 3 {
        return None;
    }

    let remaining_expr = AddView {
        root: expr,
        terms: remaining_terms,
    }
    .rebuild(ctx);
    if !is_function_free_arithmetic_expr_root(ctx, remaining_expr) {
        return None;
    }

    let log_like_expr = build_signed_sum_expr_root(ctx, &log_like_terms);
    if !expr_contains_any_builtin_local(ctx, log_like_expr, &[BuiltinFn::Atanh])
        || !expr_contains_any_builtin_local(
            ctx,
            log_like_expr,
            &[BuiltinFn::Ln, BuiltinFn::Log, BuiltinFn::Log10],
        )
    {
        return None;
    }

    let trig_numeric_expr = build_signed_sum_expr_root(ctx, &trig_numeric_terms);
    if !expr_contains_trig_builtin_local(ctx, trig_numeric_expr)
        || expr_contains_any_builtin_local(
            ctx,
            trig_numeric_expr,
            &[
                BuiltinFn::Ln,
                BuiltinFn::Log,
                BuiltinFn::Log10,
                BuiltinFn::Atanh,
            ],
        )
    {
        return None;
    }

    let log_like_rewrites_to_zero =
        matches_direct_small_zero_or_known_pair_base_root(ctx, log_like_expr)
            || try_standard_exact_zero_equivalence_shortcut(options, ctx, log_like_expr, false)
                .is_some()
            || (cas_ast::count_nodes(ctx, log_like_expr) <= 32
                && isolated_simplify_rewrites_to_zero(options, ctx, log_like_expr));
    if !log_like_rewrites_to_zero {
        return None;
    }

    let trig_numeric_rewrites_to_zero =
        matches_direct_small_zero_or_known_pair_base_root(ctx, trig_numeric_expr)
            || multiterm_trig_numeric_subset_rewrites_to_zero_runtime_safe(
                options,
                ctx,
                trig_numeric_expr,
            )
            || (cas_ast::count_nodes(ctx, trig_numeric_expr) <= 24
                && isolated_simplify_rewrites_to_zero(options, ctx, trig_numeric_expr));
    if trig_numeric_rewrites_to_zero {
        return Some(remaining_expr);
    }

    None
}

pub(super) fn try_standard_exact_zero_subset_passthrough_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let residual_expr =
        try_extract_exact_zero_subset_passthrough_residual_root(options, ctx, expr)?;
    Some(run_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        residual_expr,
        collect_steps,
    ))
}

pub(super) fn try_standard_binary_exact_zero_subset_passthrough_pair_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let binary_terms = match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) => [
            normalize_signed_add_term_root(ctx, lhs, Sign::Pos),
            normalize_signed_add_term_root(ctx, rhs, Sign::Pos),
        ],
        Expr::Sub(lhs, rhs) => [
            normalize_signed_add_term_root(ctx, lhs, Sign::Pos),
            normalize_signed_add_term_root(ctx, rhs, Sign::Neg),
        ],
        _ => return None,
    };

    for (candidate_index, candidate_term) in binary_terms.iter().copied().enumerate() {
        let partner_term = binary_terms[1 - candidate_index];
        let partner_expr = build_signed_sum_expr_root(ctx, &[partner_term]);
        if expr_contains_trig_or_hyperbolic_builtin_local(ctx, partner_expr)
            || expr_contains_log_builtin_local(ctx, partner_expr)
            || !is_function_free_arithmetic_expr_root(ctx, partner_expr)
        {
            continue;
        }

        let Some(residual_expr) =
            try_extract_exact_zero_subset_passthrough_residual_root(options, ctx, candidate_term.0)
        else {
            continue;
        };

        let combined_terms = [partner_term, (residual_expr, candidate_term.1)];
        let combined_expr = build_signed_sum_expr_root(ctx, &combined_terms);
        let combined_rewrites_to_zero =
            matches_direct_small_zero_or_known_pair_base_root(ctx, combined_expr)
                || try_standard_exact_zero_equivalence_shortcut(options, ctx, combined_expr, false)
                    .is_some()
                || (cas_ast::count_nodes(ctx, combined_expr) <= 96
                    && isolated_simplify_rewrites_to_zero(options, ctx, combined_expr));
        if !combined_rewrites_to_zero {
            continue;
        }

        let zero = ctx.num(0);
        return Some(run_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            zero,
            collect_steps,
        ));
    }

    None
}

pub(super) fn try_standard_direct_small_zero_identity_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    if AddView::from_expr(ctx, expr).terms.len() > 6 {
        return None;
    }

    if let Some((first_chunk, second_chunk)) =
        extract_partitioned_phase_shift_zero_chunks_root(ctx, expr)
    {
        let zero = ctx.num(0);
        let mut shortcut_steps = Vec::new();
        if collect_steps {
            let mut first_step = Step::new_compact(
                "Aplicar identidad de desfase",
                "Aplicar identidad de desfase",
                first_chunk,
                zero,
            );
            first_step.global_before = Some(expr);
            first_step.global_after = Some(second_chunk);
            first_step.importance = crate::step::ImportanceLevel::High;
            shortcut_steps.push(first_step);

            let mut second_step = Step::new_compact(
                "Aplicar identidad de desfase",
                "Aplicar identidad de desfase",
                second_chunk,
                zero,
            );
            second_step.global_before = Some(second_chunk);
            second_step.global_after = Some(zero);
            second_step.importance = crate::step::ImportanceLevel::High;
            shortcut_steps.push(second_step);
        }

        return Some((zero, shortcut_steps));
    }

    if let Some((_common_factor, residual_expr)) =
        extract_common_multiplicative_residual_sum_root(ctx, expr)
    {
        if extract_partitioned_phase_shift_zero_chunks_root(ctx, residual_expr).is_some() {
            let zero = ctx.num(0);
            let rewrite = crate::rule::Rewrite::with_local(
                zero,
                "Equivalent Residual Cancellation",
                expr,
                zero,
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Common-Scale Equivalent Difference",
                collect_steps,
            ));
        }
    }

    if let Some((lhs_core, rhs_core)) =
        crate::rules::arithmetic::extract_two_term_core_difference(ctx, expr)
    {
        if let Some(rewrite) =
            crate::rules::hyperbolic::try_build_atanh_square_ratio_log_zero_rewrite(ctx, expr)
        {
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ));
        }

        if collect_steps
            && expr_contains_division_node_local(ctx, lhs_core)
            && expr_contains_division_node_local(ctx, rhs_core)
        {
            return None;
        }

        if matches_direct_sum_diff_cubes_quotient_pair_root(ctx, lhs_core, rhs_core) {
            return None;
        }

        if matches_direct_half_angle_square_pair_root(ctx, lhs_core, rhs_core) {
            let zero = ctx.num(0);
            let rewrite = crate::rule::Rewrite::with_local(
                zero,
                "Aplicar identidad de ángulo mitad",
                expr,
                zero,
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Aplicar identidad de ángulo mitad",
                collect_steps,
            ));
        }

        if matches_direct_pythagorean_identity_pair_root(ctx, lhs_core, rhs_core)
            || matches_direct_pythagorean_extended_pair_root(ctx, lhs_core, rhs_core)
        {
            let zero = ctx.num(0);
            let rewrite = crate::rule::Rewrite::with_local(
                zero,
                "Collapse Exact Zero Additive Subexpression",
                expr,
                zero,
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ));
        }

        if matches_direct_angle_sum_diff_pair_root(ctx, lhs_core, rhs_core) {
            let zero = ctx.num(0);
            let rewrite =
                crate::rule::Rewrite::with_local(zero, "Angle Sum/Diff Identity", expr, zero);
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Angle Sum/Diff Identity",
                collect_steps,
            ));
        }
    }

    if collect_steps
        && (matches_direct_nested_fraction_simplified_zero_identity_root(ctx, expr)
            || matches_direct_consecutive_telescoping_fraction_zero_identity_root(ctx, expr))
    {
        return None;
    }

    if collect_steps {
        let rule = crate::rules::algebra::fractions::SubFractionsRule;
        if crate::rule::Rule::apply(&rule, ctx, expr, &crate::ParentContext::root()).is_some() {
            return None;
        }
    }

    if let Some((lhs_core, rhs_core)) = extract_direct_trig_sum_product_zero_cores_root(ctx, expr) {
        if is_plain_two_term_sin_cos_sum_or_diff_root(ctx, lhs_core)
            && is_trig_sum_product_candidate_root(ctx, rhs_core)
        {
            let zero = ctx.num(0);
            let rewrite =
                crate::rule::Rewrite::with_local(zero, "Aplicar suma a producto", expr, zero);
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Aplicar suma a producto",
                collect_steps,
            ));
        }

        if is_trig_sum_product_candidate_root(ctx, lhs_core)
            && is_plain_two_term_sin_cos_sum_or_diff_root(ctx, rhs_core)
        {
            let zero = ctx.num(0);
            let rewrite =
                crate::rule::Rewrite::with_local(zero, "Aplicar producto a suma", expr, zero);
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Aplicar producto a suma",
                collect_steps,
            ));
        }
    }

    if AddView::from_expr(ctx, expr).terms.len() == 2
        && (matches_direct_odd_half_power_zero_scope_root(ctx, expr)
            || matches_direct_odd_half_power_zero_identity_root(ctx, expr))
    {
        return None;
    }

    if let Some((lhs_core, rhs_core)) =
        extract_shared_additive_passthrough_sub_cores_root(ctx, expr)
    {
        if matches_direct_half_angle_square_pair_root(ctx, lhs_core, rhs_core) {
            let zero = ctx.num(0);
            let rewrite = crate::rule::Rewrite::with_local(
                zero,
                "Aplicar identidad de ángulo mitad",
                expr,
                zero,
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Aplicar identidad de ángulo mitad",
                collect_steps,
            ));
        }

        if matches_direct_pythagorean_identity_pair_root(ctx, lhs_core, rhs_core)
            || matches_direct_pythagorean_factor_form_pair_root(ctx, lhs_core, rhs_core)
        {
            let zero = ctx.num(0);
            let rewrite =
                crate::rule::Rewrite::with_local(zero, "Pythagorean Identity", expr, zero);
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Pythagorean Identity",
                collect_steps,
            ));
        }

        if matches_direct_cos_square_diff_pair_root(ctx, lhs_core, rhs_core) {
            let zero = ctx.num(0);
            let rewrite = crate::rule::Rewrite::with_local(
                zero,
                "Collapse Exact Zero Additive Subexpression",
                expr,
                zero,
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ));
        }

        if matches_direct_angle_sum_diff_pair_root(ctx, lhs_core, rhs_core) {
            let zero = ctx.num(0);
            let rewrite =
                crate::rule::Rewrite::with_local(zero, "Angle Sum/Diff Identity", expr, zero);
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Angle Sum/Diff Identity",
                collect_steps,
            ));
        }

        if matches_direct_trig_cubic_cosine_pair_root(ctx, lhs_core, rhs_core) {
            let zero = ctx.num(0);
            let rewrite = crate::rule::Rewrite::with_local(
                zero,
                "Collapse Exact Zero Additive Subexpression",
                expr,
                zero,
            );
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ));
        }
    }

    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 || !view.terms.iter().any(|(_, sign)| *sign == Sign::Neg) {
        return None;
    }

    let has_supported_shape = expr_contains_any_builtin_local(
        ctx,
        expr,
        &[
            BuiltinFn::Sin,
            BuiltinFn::Cos,
            BuiltinFn::Tan,
            BuiltinFn::Cot,
            BuiltinFn::Sec,
            BuiltinFn::Csc,
            BuiltinFn::Sinh,
            BuiltinFn::Cosh,
            BuiltinFn::Tanh,
        ],
    ) || expr_contains_division_node_local(ctx, expr)
        || expr_contains_sqrt_or_half_power_local(ctx, expr)
        || expr_contains_factorial_call_local(ctx, expr);
    if !has_supported_shape {
        return None;
    }

    let direct_child_zero_composition = match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => {
            matches_direct_small_zero_or_known_pair_base_root(ctx, lhs)
                && matches_direct_small_zero_or_known_pair_base_root(ctx, rhs)
        }
        _ => false,
    };
    let direct_trig_odd_half_partition =
        matches_direct_trig_product_to_sum_and_odd_half_partition_root(ctx, expr);
    let direct_trig_geometric_partition =
        matches_direct_trig_product_to_sum_and_geometric_difference_partition_root(ctx, expr);
    let direct_pair_match = match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => {
            matches_direct_small_zero_pair_root(ctx, *lhs, *rhs)
        }
        _ => false,
    };
    if !direct_child_zero_composition
        && !direct_trig_odd_half_partition
        && !direct_trig_geometric_partition
        && !matches_direct_small_zero_identity_root(ctx, expr)
        && !direct_pair_match
        && !matches_partitioned_direct_small_zero_sum_root(ctx, expr)
    {
        return None;
    }

    let zero = ctx.num(0);
    if collect_steps && view.terms.len() > 2 {
        if let Some(steps) = try_build_recursive_additive_zero_shortcut_steps(options, ctx, expr) {
            return Some((zero, steps));
        }
    }
    let rewrite = crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero);
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Collapse Exact Zero Additive Subexpression",
        collect_steps,
    ))
}

pub(super) fn try_standard_direct_known_pair_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if let Some((_common_factor, residual_expr)) =
        extract_common_multiplicative_residual_sum_root(ctx, expr)
    {
        if matches_direct_small_zero_or_known_pair_residual_root(ctx, residual_expr) {
            return None;
        }
    }

    if matches_direct_trig_product_to_sum_and_odd_half_partition_root(ctx, expr)
        || matches_direct_trig_product_to_sum_and_geometric_difference_partition_root(ctx, expr)
        || matches_partitioned_direct_small_zero_sum_root(ctx, expr)
    {
        return None;
    }

    if let Some((lhs_core, rhs_core)) =
        crate::rules::arithmetic::extract_two_term_core_difference(ctx, expr)
    {
        if matches_direct_pythagorean_identity_pair_root(ctx, lhs_core, rhs_core)
            || matches_direct_pythagorean_extended_pair_root(ctx, lhs_core, rhs_core)
            || matches_direct_angle_sum_diff_pair_root(ctx, lhs_core, rhs_core)
        {
            return None;
        }
    }

    if matches_direct_two_factor_product_pair_zero_difference_root(ctx, expr)
        || matches_direct_or_isolated_quotient_pair_zero_difference_root(options, ctx, expr)
    {
        let zero = ctx.num(0);
        let rewrite =
            crate::rule::Rewrite::with_local(zero, "Equivalent Residual Cancellation", expr, zero);
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Common-Scale Equivalent Difference",
            collect_steps,
        ));
    }

    None
}

pub(super) fn is_potential_small_exact_zero_leaf_root(ctx: &Context, expr: ExprId) -> bool {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }

    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() > 4 || cas_ast::count_nodes(ctx, expr) > 48 {
        return false;
    }

    let has_cancellation_marker = terms.iter().any(|(_, sign)| *sign == Sign::Neg)
        || terms
            .iter()
            .any(|(term, _)| expr_contains_explicit_negative_marker_local(ctx, *term));
    let has_structural_zero_family_marker =
        expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
            || expr_contains_log_builtin_local(ctx, expr)
            || expr_contains_division_node_local(ctx, expr)
            || expr_contains_sqrt_or_half_power_local(ctx, expr)
            || expr_contains_factorial_call_local(ctx, expr);
    if !has_cancellation_marker && !has_structural_zero_family_marker {
        return false;
    }

    true
}

pub(super) fn child_is_small_exact_zero_leaf_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    if !is_potential_small_exact_zero_leaf_root(ctx, expr) {
        return false;
    }

    exact_zero_leaf_rewrites_to_zero_root(options, ctx, expr)
        || (!expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
            && isolated_simplify_rewrites_to_zero(options, ctx, expr))
}

pub(super) fn is_targeted_early_small_zero_additive_combination_candidate_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let (lhs, rhs) = match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => (lhs, rhs),
        _ => return false,
    };

    let child_shape_ok = |ctx: &Context, child: ExprId| {
        if !matches!(ctx.get(child), Expr::Add(_, _) | Expr::Sub(_, _)) {
            return false;
        }
        let terms = AddView::from_expr(ctx, child).terms;
        ((2..=3).contains(&terms.len())
            || (terms.len() == 4
                && expr_contains_division_node_local(ctx, child)
                && expr_contains_any_builtin_local(ctx, child, &[BuiltinFn::Sin, BuiltinFn::Cos])))
            && terms.iter().any(|(_, sign)| *sign == Sign::Neg)
    };

    if !child_shape_ok(ctx, lhs) || !child_shape_ok(ctx, rhs) {
        return false;
    }

    let child_hits_target_family = |ctx: &mut Context, child: ExprId| {
        matches_direct_nested_fraction_simplified_zero_identity_root(ctx, child)
            || matches_direct_log_product_contract_zero_identity_root(ctx, child)
            || matches_direct_log_square_product_split_zero_identity_root(ctx, child)
            || matches_direct_ln_abs_product_split_zero_identity_root(ctx, child)
            || matches_direct_sophie_germain_zero_identity_root(ctx, child)
            || is_potential_direct_three_term_phase_shift_zero_subset_root(ctx, child)
            || crate::rules::arithmetic::try_build_small_direct_zero_core_rewrite(ctx, child)
                .is_some()
    };

    child_hits_target_family(ctx, lhs) || child_hits_target_family(ctx, rhs)
}

pub(super) fn try_standard_targeted_direct_small_zero_additive_combination_shortcut(
    simplifier: &mut Simplifier,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (result, shortcut_steps, required_conditions) = {
        let ctx = &mut simplifier.context;
        if !is_targeted_early_small_zero_additive_combination_candidate_root(ctx, expr) {
            return None;
        }

        let rewrite =
            crate::rules::arithmetic::try_build_direct_small_zero_additive_combination_rewrite(
                ctx, expr,
            )?;
        let required_conditions = rewrite.required_conditions.clone();
        let (result, shortcut_steps) = finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Exact Zero Additive Subexpression",
            collect_steps,
        );
        (result, shortcut_steps, required_conditions)
    };

    simplifier.extend_required_conditions(required_conditions);
    Some((result, shortcut_steps))
}

fn direct_small_zero_additive_combination_max_terms_root(ctx: &Context, expr: ExprId) -> usize {
    if expr_contains_division_node_local(ctx, expr)
        && expr_contains_any_builtin_local(ctx, expr, &[BuiltinFn::Sin, BuiltinFn::Cos])
    {
        7
    } else {
        6
    }
}

pub(super) fn try_standard_direct_small_zero_additive_combination_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let view = AddView::from_expr(ctx, expr);
    let term_count = view.terms.len();
    if term_count < 4
        || term_count > direct_small_zero_additive_combination_max_terms_root(ctx, expr)
    {
        return None;
    }

    // Rational-constant sums are Core constant folding's job (instant); every
    // probe below hunts STRUCTURED cancellations (trig products, surd pairs,
    // chunked partitions), and none of them can beat the fold. The solver's
    // rational-root screening pipes candidate evaluations like
    // `1^3 - 6·1^2 + 11·1 - 6` through here as root expressions — measured
    // ~2-3 ms of probe cost per candidate. Surd/transcendental terms make
    // `as_rational_const` bail, so the sums the probes exist for still enter.
    let all_rational_const_terms = view
        .terms
        .iter()
        .all(|(term, _)| cas_ast::views::as_rational_const(ctx, *term, 8).is_some());
    if all_rational_const_terms {
        return None;
    }

    if term_count > 6 {
        let rewrite =
            crate::rules::arithmetic::try_build_direct_small_zero_additive_combination_rewrite(
                ctx, expr,
            )?;
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Exact Zero Additive Subexpression",
            collect_steps,
        ));
    }

    if matches_direct_trig_product_to_sum_and_odd_half_partition_root(ctx, expr) {
        let zero = ctx.num(0);
        let rewrite =
            crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero);
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Exact Zero Additive Subexpression",
            collect_steps,
        ));
    }

    if matches_direct_trig_product_to_sum_and_geometric_difference_partition_root(ctx, expr) {
        let zero = ctx.num(0);
        let rewrite =
            crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero);
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Exact Zero Additive Subexpression",
            collect_steps,
        ));
    }

    if let Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) = ctx.get(expr).clone() {
        let child_exact_zero_composition =
            matches_direct_small_zero_or_known_pair_base_root(ctx, lhs)
                && matches_direct_small_zero_or_known_pair_base_root(ctx, rhs);
        let leaf_exact_zero_composition = child_is_small_exact_zero_leaf_root(options, ctx, lhs)
            && child_is_small_exact_zero_leaf_root(options, ctx, rhs);
        if child_exact_zero_composition || leaf_exact_zero_composition {
            let zero = ctx.num(0);
            if collect_steps && matches!(ctx.get(expr), Expr::Add(_, _)) {
                if let Some(steps) =
                    try_build_chunk_pair_zero_shortcut_steps_root(options, ctx, expr, lhs, rhs)
                {
                    return Some((zero, steps));
                }
            }
            let rewrite =
                crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero);
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Exact Zero Additive Subexpression",
                collect_steps,
            ));
        }
    }

    if matches_partitioned_direct_small_zero_sum_root(ctx, expr) {
        let zero = ctx.num(0);
        let rewrite =
            crate::rule::Rewrite::with_local(zero, "Exact Zero Core Composition", expr, zero);
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Exact Zero Additive Subexpression",
            collect_steps,
        ));
    }

    if let Some((first_chunk, second_chunk)) =
        extract_partitioned_phase_shift_zero_chunks_root(ctx, expr)
    {
        let zero = ctx.num(0);
        let mut shortcut_steps = Vec::new();
        if collect_steps {
            let mut first_step = Step::new_compact(
                "Aplicar identidad de desfase",
                "Aplicar identidad de desfase",
                first_chunk,
                zero,
            );
            first_step.global_before = Some(expr);
            first_step.global_after = Some(second_chunk);
            first_step.importance = crate::step::ImportanceLevel::High;
            shortcut_steps.push(first_step);

            let mut second_step = Step::new_compact(
                "Aplicar identidad de desfase",
                "Aplicar identidad de desfase",
                second_chunk,
                zero,
            );
            second_step.global_before = Some(second_chunk);
            second_step.global_after = Some(zero);
            second_step.importance = crate::step::ImportanceLevel::High;
            shortcut_steps.push(second_step);
        }

        return Some((zero, shortcut_steps));
    }

    if !crate::rules::arithmetic::maybe_direct_small_zero_additive_combination_candidate(ctx, expr)
    {
        return None;
    }

    let rewrite =
        crate::rules::arithmetic::try_build_direct_small_zero_additive_combination_rewrite(
            ctx, expr,
        )?;
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Collapse Exact Zero Additive Subexpression",
        collect_steps,
    ))
}

pub(super) fn passthrough_residual_zero_rule_name_root(
    ctx: &mut Context,
    residual_expr: ExprId,
) -> Option<&'static str> {
    if matches_direct_symbolic_trig_sum_to_product_zero_identity_root(ctx, residual_expr) {
        return Some("Aplicar suma a producto");
    }

    if matches_direct_half_angle_square_zero_identity_root(ctx, residual_expr) {
        return Some("Aplicar identidad de ángulo mitad");
    }

    if matches_direct_general_phase_shift_zero_identity_root(ctx, residual_expr) {
        return Some("Aplicar identidad de desfase");
    }

    if matches_direct_nested_zero_trig_ratio_alias_residual_pair_root(ctx, residual_expr) {
        return Some("Collapse Exact Zero Additive Subexpression");
    }

    if matches_direct_log_square_product_split_zero_identity_root(ctx, residual_expr)
        || matches_direct_ln_abs_product_split_zero_identity_root(ctx, residual_expr)
    {
        return Some("Expandir logaritmos y cancelar términos iguales");
    }

    if matches_direct_small_zero_identity_root(ctx, residual_expr) {
        return Some("Collapse Exact Zero Additive Subexpression");
    }

    None
}
