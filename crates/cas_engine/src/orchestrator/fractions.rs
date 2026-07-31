//! Orquestador: familia `fractions` (troceo P1).
//!
//! Ver la cabecera de `orchestrator.rs` para el contexto.

use super::*;

pub(super) fn is_same_denominator_difference_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let mut denominator = None;
    for (term_expr, _) in view.terms {
        let Expr::Div(_, den) = ctx.get(term_expr) else {
            return false;
        };

        if let Some(existing_den) = denominator {
            if compare_expr(ctx, *den, existing_den) != Ordering::Equal {
                return false;
            }
        } else {
            denominator = Some(*den);
        }
    }

    denominator.is_some()
}

fn has_shared_additive_denominator_root(ctx: &mut Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return false;
    }

    let mut denominator = None;
    for (term_expr, _) in view.terms {
        let Expr::Div(_, den) = ctx.get(term_expr) else {
            return false;
        };

        if let Some(existing_den) = denominator {
            if compare_expr(ctx, *den, existing_den) != Ordering::Equal {
                return false;
            }
        } else {
            denominator = Some(*den);
        }
    }

    denominator.is_some()
}

pub(super) fn matches_structural_same_denominator_distribution_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if !(3..=5).contains(&view.terms.len()) || !has_shared_additive_denominator_root(ctx, expr) {
        return false;
    }

    for candidate_index in 0..view.terms.len() {
        let (combined_term, combined_sign) = view.terms[candidate_index];
        let Expr::Div(combined_numerator, combined_denominator) = ctx.get(combined_term).clone()
        else {
            continue;
        };

        let combined_terms = AddView::from_expr(ctx, combined_numerator).terms;
        if combined_terms.len() != view.terms.len() - 1
            || combined_terms.iter().any(|(_, sign)| *sign != Sign::Pos)
        {
            continue;
        }

        let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != candidate_index).then_some(term))
            .collect();
        if remaining_terms
            .iter()
            .any(|(term_expr, term_sign)| match ctx.get(*term_expr) {
                Expr::Div(_, den) => {
                    *term_sign == combined_sign
                        || compare_expr(ctx, *den, combined_denominator) != Ordering::Equal
                }
                _ => true,
            })
        {
            continue;
        }

        let mut used = vec![false; remaining_terms.len()];
        let mut matched_all = true;
        for (combined_child, _) in combined_terms {
            let mut matched = false;
            for (index, (remaining_expr, _remaining_sign)) in remaining_terms.iter().enumerate() {
                if used[index] {
                    continue;
                }
                let Expr::Div(remaining_numerator, _) = ctx.get(*remaining_expr).clone() else {
                    continue;
                };
                if compare_expr(ctx, combined_child, remaining_numerator) == Ordering::Equal {
                    used[index] = true;
                    matched = true;
                    break;
                }
            }
            if !matched {
                matched_all = false;
                break;
            }
        }

        if matched_all {
            return true;
        }
    }

    false
}

pub(super) fn extract_same_denominator_direct_pair_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let first = view.terms[0];
    let second = view.terms[1];
    let (first_num, first_den) = match ctx.get(first.0) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    let (second_num, second_den) = match ctx.get(second.0) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    if compare_expr(ctx, first_den, second_den) != Ordering::Equal {
        return None;
    }

    match (first.1, second.1) {
        (Sign::Pos, Sign::Neg) => Some((first_den, first_num, second_num)),
        (Sign::Neg, Sign::Pos) => Some((first_den, second_num, first_num)),
        _ => None,
    }
}

pub(super) fn has_sum_diff_cubes_quotient_term_root(ctx: &mut Context, expr: ExprId) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    matches!(terms.len(), 2 | 4)
        && terms.iter().any(|(term_expr, _)| {
            extract_sum_diff_cubes_quotient_bases_root(ctx, *term_expr).is_some()
        })
}

pub(super) fn matches_direct_small_zero_nested_fraction_division_hot_pair_root(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    for (nested_fraction_side, other_division_side) in [(lhs, rhs), (rhs, lhs)] {
        if !expr_contains_division_node_local(ctx, nested_fraction_side)
            || !expr_contains_division_node_local(ctx, other_division_side)
        {
            continue;
        }

        if matches_direct_depth_three_unit_continued_fraction_zero_identity_root(
            ctx,
            nested_fraction_side,
        ) && (matches_direct_consecutive_telescoping_fraction_zero_identity_root(
            ctx,
            other_division_side,
        ) || extract_small_quotient_cancel_zero_candidate_root(ctx, other_division_side)
            .is_some())
        {
            return true;
        }
    }

    false
}

fn is_guarded_small_zero_shifted_quotient_candidate_root(ctx: &mut Context, expr: ExprId) -> bool {
    let (numerator, denominator) = match ctx.get(expr).clone() {
        Expr::Div(numerator, denominator) => (numerator, denominator),
        _ => return false,
    };
    let Some(numerator_core) = strip_positive_one_passthrough_root(ctx, numerator) else {
        return false;
    };
    let Some(denominator_core) = strip_positive_one_passthrough_root(ctx, denominator) else {
        return false;
    };

    matches_guarded_small_zero_pair_root(ctx, numerator_core, denominator_core)
}

fn is_direct_small_zero_shifted_quotient_candidate_root(ctx: &mut Context, expr: ExprId) -> bool {
    let (numerator, denominator) = match ctx.get(expr).clone() {
        Expr::Div(numerator, denominator) => (numerator, denominator),
        _ => return false,
    };
    let Some(numerator_core) = strip_positive_one_passthrough_root(ctx, numerator) else {
        return false;
    };
    let Some(denominator_core) = strip_positive_one_passthrough_root(ctx, denominator) else {
        return false;
    };

    matches_direct_small_zero_pair_root(ctx, numerator_core, denominator_core)
}

pub(super) fn extract_div_by_two_numerator_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    matches!(ctx.get(*den), Expr::Number(n) if n.is_integer() && n.to_integer() == 2.into())
        .then_some(*num)
}

pub(super) fn build_collapsed_successive_unit_fractions_expr_root(
    ctx: &mut Context,
    base: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let doubled_base = smart_mul(ctx, two, base);
    let numerator = ctx.add(Expr::Add(doubled_base, one));
    let plus_one = ctx.add(Expr::Add(base, one));
    let denominator = build_mul_expr_from_factors_root(ctx, &[base, plus_one]);
    ctx.add(Expr::Div(numerator, denominator))
}

pub(super) fn build_consecutive_telescoping_fraction_difference_expr_root(
    ctx: &mut Context,
    base: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let plus_one = ctx.add(Expr::Add(base, one));
    let lhs = ctx.add(Expr::Div(one, base));
    let rhs = ctx.add(Expr::Div(one, plus_one));
    ctx.add(Expr::Sub(lhs, rhs))
}

pub(super) fn matches_direct_nested_fraction_simplified_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (continued_fraction_expr, rational_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if let Some(arg) = extract_depth_two_unit_reciprocal_continued_fraction_arg_root(
            ctx,
            continued_fraction_expr,
        ) {
            if matches_depth_two_unit_continued_fraction_target_root(ctx, rational_expr, arg) {
                return true;
            }
        }

        if let Some(arg) = extract_depth_three_unit_reciprocal_continued_fraction_arg_root(
            ctx,
            continued_fraction_expr,
        ) {
            if matches_depth_three_unit_reciprocal_continued_fraction_target_root(
                ctx,
                rational_expr,
                arg,
            ) {
                return true;
            }
        }

        let Some(arg) =
            extract_depth_three_unit_continued_fraction_arg_root(ctx, continued_fraction_expr)
        else {
            continue;
        };
        if matches_depth_three_unit_continued_fraction_target_root(ctx, rational_expr, arg) {
            return true;
        }
    }

    false
}

fn extract_depth_two_unit_reciprocal_continued_fraction_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let denominator = extract_unit_fraction_denominator_root(ctx, expr)?;
    let first_tail = extract_positive_one_plus_other_term_root(ctx, denominator)?;
    let inner_sum = extract_unit_fraction_denominator_root(ctx, first_tail)?;
    extract_positive_one_plus_other_term_root(ctx, inner_sum)
}

fn extract_depth_three_unit_reciprocal_continued_fraction_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let denominator = extract_unit_fraction_denominator_root(ctx, expr)?;
    let first_tail = extract_positive_one_plus_other_term_root(ctx, denominator)?;
    let inner_sum = extract_unit_fraction_denominator_root(ctx, first_tail)?;
    let second_tail = extract_positive_one_plus_other_term_root(ctx, inner_sum)?;
    let second_inner_sum = extract_unit_fraction_denominator_root(ctx, second_tail)?;
    extract_positive_one_plus_other_term_root(ctx, second_inner_sum)
}

fn matches_depth_two_unit_continued_fraction_target_root(
    ctx: &mut Context,
    expr: ExprId,
    arg: ExprId,
) -> bool {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return false,
    };

    let one = ctx.num(1);
    let two = ctx.num(2);
    let expected_numerator = ctx.add(Expr::Add(arg, one));
    let expected_denominator = ctx.add(Expr::Add(arg, two));

    compare_expr(ctx, numerator, expected_numerator) == Ordering::Equal
        && compare_expr(ctx, denominator, expected_denominator) == Ordering::Equal
}

fn matches_depth_three_unit_reciprocal_continued_fraction_target_root(
    ctx: &mut Context,
    expr: ExprId,
    arg: ExprId,
) -> bool {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return false,
    };

    let two = ctx.num(2);
    let three = ctx.num(3);
    let expected_numerator = ctx.add(Expr::Add(arg, two));
    let two_times_arg = smart_mul(ctx, two, arg);
    let expected_denominator = ctx.add(Expr::Add(two_times_arg, three));

    compare_expr(ctx, numerator, expected_numerator) == Ordering::Equal
        && compare_expr(ctx, denominator, expected_denominator) == Ordering::Equal
}

pub(super) fn matches_direct_depth_three_unit_continued_fraction_zero_identity_terms_root(
    ctx: &mut Context,
    terms: &[(ExprId, Sign)],
) -> bool {
    if terms.len() != 3 {
        return false;
    }

    let mut continued_fraction_tail = None;
    let mut rational_expr = None;
    let mut saw_positive_one = false;

    for (term_expr, term_sign) in terms.iter().copied() {
        match (term_sign, ctx.get(term_expr)) {
            (Sign::Pos, Expr::Number(n)) if n.is_one() => {
                if saw_positive_one {
                    return false;
                }
                saw_positive_one = true;
            }
            (Sign::Pos, _) => {
                if continued_fraction_tail.is_some() {
                    return false;
                }
                continued_fraction_tail = Some(term_expr);
            }
            (Sign::Neg, _) => {
                if rational_expr.is_some() {
                    return false;
                }
                rational_expr = Some(term_expr);
            }
        }
    }

    let Some(continued_fraction_tail) = continued_fraction_tail else {
        return false;
    };
    let Some(rational_expr) = rational_expr else {
        return false;
    };
    if !saw_positive_one {
        return false;
    }

    let Some(depth_two_expr) = extract_unit_fraction_denominator_root(ctx, continued_fraction_tail)
    else {
        return false;
    };
    let Some(arg) = extract_depth_two_unit_continued_fraction_arg_root(ctx, depth_two_expr) else {
        return false;
    };
    matches_depth_three_unit_continued_fraction_target_root(ctx, rational_expr, arg)
}

pub(super) fn matches_direct_depth_three_unit_continued_fraction_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    matches_direct_depth_three_unit_continued_fraction_zero_identity_terms_root(ctx, &view.terms)
}

fn extract_depth_two_unit_continued_fraction_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let first_tail = extract_positive_one_plus_other_term_root(ctx, expr)?;
    let inner_sum = extract_unit_fraction_denominator_root(ctx, first_tail)?;
    let second_tail = extract_positive_one_plus_other_term_root(ctx, inner_sum)?;
    extract_unit_fraction_denominator_root(ctx, second_tail)
}

pub(super) fn matches_direct_nested_fraction_simplified_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return false;
    }

    if matches_direct_depth_three_unit_continued_fraction_zero_identity_terms_root(ctx, &view.terms)
    {
        return true;
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
        if remaining_terms.is_empty() {
            continue;
        }

        let remaining_expr = AddView {
            root: expr,
            terms: remaining_terms,
        }
        .rebuild(ctx);
        if matches_direct_nested_fraction_simplified_pair_root(ctx, remaining_expr, term_expr) {
            return true;
        }
    }

    false
}

fn extract_depth_three_unit_continued_fraction_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let first_tail = extract_positive_one_plus_other_term_root(ctx, expr)?;
    let inner_sum = extract_unit_fraction_denominator_root(ctx, first_tail)?;
    extract_depth_two_unit_continued_fraction_arg_root(ctx, inner_sum)
}

fn matches_depth_three_unit_continued_fraction_target_root(
    ctx: &mut Context,
    expr: ExprId,
    arg: ExprId,
) -> bool {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return false,
    };

    let one = ctx.num(1);
    let two = ctx.num(2);
    let three = ctx.num(3);
    let three_times_arg = smart_mul(ctx, three, arg);
    let expected_numerator = ctx.add(Expr::Add(three_times_arg, two));
    let two_times_arg = smart_mul(ctx, two, arg);
    let expected_denominator = ctx.add(Expr::Add(two_times_arg, one));

    compare_expr(ctx, numerator, expected_numerator) == Ordering::Equal
        && compare_expr(ctx, denominator, expected_denominator) == Ordering::Equal
}

fn build_sum_diff_cubes_quotient_expansion_root(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> ExprId {
    let build_square = |ctx: &mut Context, expr: ExprId| -> ExprId {
        match ctx.get(expr) {
            Expr::Number(n) => ctx.add(Expr::Number(n.clone() * n.clone())),
            _ => {
                let two = ctx.num(2);
                ctx.add(Expr::Pow(expr, two))
            }
        }
    };
    let lhs_sq = build_square(ctx, lhs);
    let rhs_sq = build_square(ctx, rhs);
    let mixed = build_mul_expr_from_factors_root(ctx, &[lhs, rhs]);
    build_balanced_add(ctx, &[lhs_sq, mixed, rhs_sq])
}

pub(super) fn matches_direct_sum_diff_cubes_quotient_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (quotient_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((lhs, rhs)) = extract_sum_diff_cubes_quotient_bases_root(ctx, quotient_expr)
        else {
            continue;
        };
        let expanded = build_sum_diff_cubes_quotient_expansion_root(ctx, lhs, rhs);
        if compare_expr(ctx, expanded, target_expr) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, expanded) <= 24
            && cas_ast::count_nodes(ctx, target_expr) <= 24
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                expanded,
                target_expr,
            )
        {
            return true;
        }
    }

    false
}

pub(super) fn extract_literal_rational_root(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    match ctx.get(expr) {
        Expr::Number(n) => Some(n.clone()),
        Expr::Neg(inner) => extract_literal_rational_root(ctx, *inner).map(|n| -n),
        Expr::Div(numerator, denominator) => {
            let numerator = extract_literal_rational_root(ctx, *numerator)?;
            let denominator = extract_literal_rational_root(ctx, *denominator)?;
            Some(numerator / denominator)
        }
        _ => None,
    }
}

pub(super) fn matches_direct_cube_root_rationalization_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) =
            cas_math::root_den_rationalize_support::try_rewrite_rationalize_cube_root_den_expr(
                ctx,
                source_expr,
            )
        else {
            continue;
        };
        let normalized_rewritten = strip_multiplicative_one_root(ctx, rewrite.rewritten);
        let normalized_target = strip_multiplicative_one_root(ctx, target_expr);
        if compare_expr(ctx, normalized_rewritten, normalized_target) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, normalized_rewritten) <= 20
            && cas_ast::count_nodes(ctx, normalized_target) <= 20
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                normalized_rewritten,
                normalized_target,
            )
        {
            return true;
        }
        let difference = ctx.add(Expr::Sub(normalized_rewritten, normalized_target));
        if cas_ast::count_nodes(ctx, difference) <= 32
            && isolated_simplify_rewrites_to_zero(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                difference,
            )
        {
            return true;
        }
    }

    false
}

fn extract_plus_or_minus_one_denominator_arg_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, bool)> {
    match ctx.get(expr) {
        Expr::Sub(lhs, rhs) if matches!(ctx.get(*rhs), Expr::Number(n) if n.is_one()) => {
            Some((*lhs, false))
        }
        Expr::Add(lhs, rhs) if matches!(ctx.get(*rhs), Expr::Number(n) if n.is_one()) => {
            Some((*lhs, true))
        }
        Expr::Add(lhs, rhs) if matches!(ctx.get(*lhs), Expr::Number(n) if n.is_one()) => {
            Some((*rhs, true))
        }
        _ => None,
    }
}

fn extract_rational_plus_minus_one_sum_arg_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 || !terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
        return None;
    }

    let mut minus_arg = None;
    let mut plus_arg = None;
    for (term_expr, _) in terms {
        let Expr::Div(num, den) = ctx.get(term_expr) else {
            return None;
        };
        if !matches!(ctx.get(*num), Expr::Number(n) if n.is_one()) {
            return None;
        }
        let (arg, is_plus) = extract_plus_or_minus_one_denominator_arg_root(ctx, *den)?;
        if is_plus {
            plus_arg = Some(arg);
        } else {
            minus_arg = Some(arg);
        }
    }

    let minus_arg = minus_arg?;
    let plus_arg = plus_arg?;
    (compare_expr(ctx, minus_arg, plus_arg) == Ordering::Equal).then_some(minus_arg)
}

fn matches_rational_plus_minus_one_target_root(
    ctx: &mut Context,
    expr: ExprId,
    arg: ExprId,
) -> bool {
    let two = ctx.num(2);
    let one = ctx.num(1);
    let numerator = smart_mul(ctx, two, arg);
    let squared = ctx.add(Expr::Pow(arg, two));
    let den_poly = ctx.add(Expr::Sub(squared, one));
    let minus_one_den = ctx.add(Expr::Sub(arg, one));
    let plus_one_den = ctx.add(Expr::Add(arg, one));
    let den_factored = ctx.add(Expr::Mul(minus_one_den, plus_one_den));
    let poly_target = ctx.add(Expr::Div(numerator, den_poly));
    let factored_target = ctx.add(Expr::Div(numerator, den_factored));

    compare_expr(ctx, expr, poly_target) == Ordering::Equal
        || compare_expr(ctx, expr, factored_target) == Ordering::Equal
}

pub(super) fn matches_direct_rational_plus_minus_one_sum_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (sum_expr, rational_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(arg) = extract_rational_plus_minus_one_sum_arg_root(ctx, sum_expr) else {
            continue;
        };
        if matches_rational_plus_minus_one_target_root(ctx, rational_expr, arg) {
            return true;
        }
    }

    false
}

pub(super) fn extract_addition_of_successive_unit_fractions_arg_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 || !view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
        return None;
    }

    let mut denominators = Vec::with_capacity(2);
    for (term_expr, _) in view.terms {
        let Expr::Div(num, den) = ctx.get(term_expr) else {
            return None;
        };
        if !matches!(ctx.get(*num), Expr::Number(n) if n.is_one()) {
            return None;
        }
        denominators.push(*den);
    }

    let first = denominators[0];
    let second = denominators[1];
    if let Some(base) = extract_plus_one_expr_target_root(ctx, first) {
        if compare_expr(ctx, base, second) == Ordering::Equal {
            return Some(second);
        }
    }
    if let Some(base) = extract_plus_one_expr_target_root(ctx, second) {
        if compare_expr(ctx, base, first) == Ordering::Equal {
            return Some(first);
        }
    }

    None
}

pub(super) fn extract_collapsed_successive_unit_fractions_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };

    let one = ctx.num(1);
    let two = ctx.num(2);
    let view = AddView::from_expr(ctx, num);
    if view.terms.len() != 2 || !view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
        return None;
    }

    let mut base = None;
    let mut saw_one = false;
    for (term_expr, _) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if saw_one {
                return None;
            }
            saw_one = true;
            continue;
        }

        let Expr::Mul(lhs, rhs) = ctx.get(term_expr) else {
            return None;
        };
        let doubled_base = if extract_i64_integer(ctx, *lhs) == Some(2) {
            *rhs
        } else if extract_i64_integer(ctx, *rhs) == Some(2) {
            *lhs
        } else {
            return None;
        };
        if base.replace(doubled_base).is_some() {
            return None;
        }
    }

    let base = base?;
    if !saw_one {
        return None;
    }

    let plus_one = ctx.add(Expr::Add(base, one));
    let doubled_base = smart_mul(ctx, two, base);
    let expected_num = ctx.add(Expr::Add(doubled_base, one));
    let expected_den = ctx.add(Expr::Mul(base, plus_one));
    let squared_base = ctx.add(Expr::Pow(base, two));
    let expanded_den = ctx.add(Expr::Add(squared_base, base));
    if compare_expr(ctx, num, expected_num) == Ordering::Equal
        && (compare_expr(ctx, den, expected_den) == Ordering::Equal
            || compare_expr(ctx, den, expanded_den) == Ordering::Equal
            || matches_direct_linear_factoring_pair_root(ctx, den, expected_den))
    {
        return Some(base);
    }

    None
}

pub(super) fn extract_consecutive_telescoping_fraction_difference_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut positive_den = None;
    let mut negative_den = None;
    for (term_expr, sign) in view.terms {
        let denominator = extract_unit_fraction_denominator_root(ctx, term_expr)?;
        match sign {
            Sign::Pos => {
                if positive_den.replace(denominator).is_some() {
                    return None;
                }
            }
            Sign::Neg => {
                if negative_den.replace(denominator).is_some() {
                    return None;
                }
            }
        }
    }

    let positive_den = positive_den?;
    let negative_den = negative_den?;
    let one = ctx.num(1);
    let positive_plus_one = ctx.add(Expr::Add(positive_den, one));
    (compare_expr(ctx, positive_plus_one, negative_den) == Ordering::Equal).then_some(positive_den)
}

pub(super) fn matches_direct_addition_of_successive_unit_fractions_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (sum_expr, collapsed_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(base) = extract_addition_of_successive_unit_fractions_arg_root(ctx, sum_expr)
        else {
            continue;
        };
        let Some(collapsed_base) =
            extract_collapsed_successive_unit_fractions_arg_root(ctx, collapsed_expr)
        else {
            continue;
        };
        if compare_expr(ctx, base, collapsed_base) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn build_tangent_addition_fraction_root(
    ctx: &mut Context,
    lhs_arg: ExprId,
    rhs_arg: ExprId,
) -> ExprId {
    let sum_arg = ctx.add(Expr::Add(lhs_arg, rhs_arg));
    let numerator = ctx.call_builtin(BuiltinFn::Sin, vec![sum_arg]);
    let lhs_cos = ctx.call_builtin(BuiltinFn::Cos, vec![lhs_arg]);
    let rhs_cos = ctx.call_builtin(BuiltinFn::Cos, vec![rhs_arg]);
    let denominator = build_mul_expr_from_factors_root(ctx, &[lhs_cos, rhs_cos]);
    ctx.add(Expr::Div(numerator, denominator))
}

pub(super) fn extract_direct_tangent_addition_fraction_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    let Some((BuiltinFn::Sin, sum_arg)) = extract_plain_sin_or_cos_arg_root(ctx, numerator) else {
        return None;
    };
    let Expr::Add(sum_lhs, sum_rhs) = ctx.get(sum_arg) else {
        return None;
    };
    let sum_lhs = *sum_lhs;
    let sum_rhs = *sum_rhs;
    let (den_lhs, den_rhs) =
        extract_plain_trig_product_pair_args_root(ctx, denominator, BuiltinFn::Cos)?;
    matches_unordered_expr_pair_root(ctx, sum_lhs, sum_rhs, den_lhs, den_rhs)
        .then_some((sum_lhs, sum_rhs))
}

pub(super) fn extract_unit_fraction_denominator_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    match ctx.get(numerator) {
        Expr::Number(n) if n.is_one() => Some(denominator),
        _ => {
            let one = ctx.num(1);
            (compare_expr(ctx, numerator, one) == Ordering::Equal).then_some(denominator)
        }
    }
}

pub(super) fn matches_direct_consecutive_telescoping_fraction_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut product_core = None;
    let mut product_sign = None;
    let mut single_terms: smallvec::SmallVec<[(ExprId, Sign); 2]> = smallvec::SmallVec::new();

    for (term_expr, term_sign) in view.terms {
        let Some(denominator) = extract_unit_fraction_denominator_root(ctx, term_expr) else {
            return false;
        };

        if let Some(candidate_u) = extract_consecutive_product_core_root(ctx, denominator) {
            if product_core.is_some() {
                return false;
            }
            product_core = Some(candidate_u);
            product_sign = Some(term_sign);
        } else {
            single_terms.push((denominator, term_sign));
        }
    }

    let (u, product_sign) = match (product_core, product_sign) {
        (Some(u), Some(sign)) => (u, sign),
        _ => return false,
    };
    if single_terms.len() != 2 {
        return false;
    }

    let one = ctx.num(1);
    let u_plus_one = ctx.add(Expr::Add(u, one));
    let mut saw_u = false;
    let mut saw_u_plus_one = false;

    for (denominator, sign) in single_terms {
        if compare_expr(ctx, denominator, u) == Ordering::Equal {
            if sign == product_sign {
                return false;
            }
            saw_u = true;
            continue;
        }
        if compare_expr(ctx, denominator, u_plus_one) == Ordering::Equal {
            if sign != product_sign {
                return false;
            }
            saw_u_plus_one = true;
            continue;
        }
        return false;
    }

    saw_u && saw_u_plus_one
}

fn numerator_matches_two_times_shift_root(
    ctx: &mut Context,
    numerator: ExprId,
    shift: ExprId,
) -> bool {
    if extract_i64_integer(ctx, numerator) == Some(2) && extract_i64_integer(ctx, shift) == Some(1)
    {
        return true;
    }

    let factors = flatten_mul_chain(ctx, numerator);
    if factors.len() != 2 {
        return false;
    }

    let two = ctx.num(2);
    (compare_expr(ctx, factors[0], two) == Ordering::Equal
        && compare_expr(ctx, factors[1], shift) == Ordering::Equal)
        || (compare_expr(ctx, factors[1], two) == Ordering::Equal
            && compare_expr(ctx, factors[0], shift) == Ordering::Equal)
}

fn matches_direct_symmetric_partial_fraction_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 || !expr_contains_division_node_local(ctx, expr) {
        return false;
    }

    let positive_terms: smallvec::SmallVec<[(ExprId, Sign); 1]> = view
        .terms
        .iter()
        .copied()
        .filter(|(_, sign)| *sign == Sign::Pos)
        .collect();
    let negative_terms: smallvec::SmallVec<[(ExprId, Sign); 2]> = view
        .terms
        .iter()
        .copied()
        .filter(|(_, sign)| *sign == Sign::Neg)
        .collect();
    if positive_terms.len() != 1 || negative_terms.len() != 2 {
        return false;
    }

    let Some(positive_denominator) =
        extract_unit_fraction_denominator_root(ctx, positive_terms[0].0)
    else {
        return false;
    };
    let Expr::Sub(base, shift) = ctx.get(positive_denominator).clone() else {
        return false;
    };

    for (negative_unit, target_fraction) in [
        (negative_terms[0].0, negative_terms[1].0),
        (negative_terms[1].0, negative_terms[0].0),
    ] {
        let Some(negative_denominator) = extract_unit_fraction_denominator_root(ctx, negative_unit)
        else {
            continue;
        };
        if !positive_two_term_sum_matches_terms_root(ctx, negative_denominator, base, shift) {
            continue;
        }

        let Expr::Div(target_numerator, target_denominator) = ctx.get(target_fraction).clone()
        else {
            continue;
        };
        if !numerator_matches_two_times_shift_root(ctx, target_numerator, shift) {
            continue;
        }

        let two = ctx.num(2);
        let base_squared = ctx.add(Expr::Pow(base, two));
        let shift_squared = build_square_preserving_one_root(ctx, shift);
        let expected_denominator = ctx.add(Expr::Sub(base_squared, shift_squared));
        if compare_expr(ctx, target_denominator, expected_denominator) != Ordering::Equal {
            continue;
        }

        return true;
    }

    false
}

pub(super) fn matches_direct_same_denominator_common_scaled_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if !(3..=5).contains(&view.terms.len())
        || !has_shared_additive_denominator_root(ctx, expr)
        || expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
        || cas_ast::count_nodes(ctx, expr) > 32
    {
        return false;
    }

    if matches_structural_same_denominator_distribution_zero_identity_root(ctx, expr) {
        return true;
    }

    let parent_ctx = crate::ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let rule = crate::rules::arithmetic::CollapseExactZeroCommonScaledDifferenceRule;
    let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) else {
        return false;
    };
    let zero = ctx.num(0);
    compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal
}

pub(super) fn matches_direct_affine_common_denominator_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3
        || !expr_contains_division_node_local(ctx, expr)
        || expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
        || cas_ast::count_nodes(ctx, expr) > 32
    {
        return false;
    }

    let mut plain_term = None;
    let mut div_terms = smallvec::SmallVec::<[(ExprId, ExprId, Sign); 2]>::new();
    for (term_expr, term_sign) in view.terms {
        match ctx.get(term_expr).clone() {
            Expr::Div(numerator, denominator) => {
                div_terms.push((numerator, denominator, term_sign));
            }
            _ => {
                if plain_term.is_some() {
                    return false;
                }
                plain_term = Some((term_expr, term_sign));
            }
        }
    }

    let Some((plain_expr, plain_sign)) = plain_term else {
        return false;
    };
    if div_terms.len() != 2 || compare_expr(ctx, div_terms[0].1, div_terms[1].1) != Ordering::Equal
    {
        return false;
    }

    let denominator = div_terms[0].1;
    let scaled_plain = smart_mul(ctx, plain_expr, denominator);
    for (combined_index, other_index) in [(0usize, 1usize), (1usize, 0usize)] {
        let actual_combined_terms = extract_normalized_signed_terms_with_outer_sign_root(
            ctx,
            div_terms[combined_index].0,
            div_terms[combined_index].2,
        );
        let mut expected_terms = smallvec::SmallVec::<[(ExprId, Sign); 8]>::new();
        expected_terms.push((scaled_plain, plain_sign));
        expected_terms.extend(extract_normalized_signed_terms_with_outer_sign_root(
            ctx,
            div_terms[other_index].0,
            div_terms[other_index].2,
        ));

        if signed_term_multiset_matches_root(
            ctx,
            &actual_combined_terms,
            &flipped_signed_terms_root(&expected_terms),
        ) {
            return true;
        }
    }

    false
}

fn extract_reciprocal_sum_difference_nested_fraction_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId, ExprId)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
        return None;
    };

    let numerator_view = AddView::from_expr(ctx, numerator);
    let denominator_view = AddView::from_expr(ctx, denominator);
    if numerator_view.terms.len() != 2 || denominator_view.terms.len() != 2 {
        return None;
    }

    let mut numerator_denominators = smallvec::SmallVec::<[ExprId; 2]>::new();
    for (term_expr, term_sign) in numerator_view.terms {
        if term_sign != Sign::Pos {
            return None;
        }
        numerator_denominators.push(extract_unit_fraction_denominator_root(ctx, term_expr)?);
    }

    let mut positive_denominator = None;
    let mut negative_denominator = None;
    for (term_expr, term_sign) in denominator_view.terms {
        let unit_denominator = extract_unit_fraction_denominator_root(ctx, term_expr)?;
        match term_sign {
            Sign::Pos if positive_denominator.is_none() => {
                positive_denominator = Some(unit_denominator);
            }
            Sign::Neg if negative_denominator.is_none() => {
                negative_denominator = Some(unit_denominator);
            }
            _ => return None,
        }
    }

    let positive_denominator = positive_denominator?;
    let negative_denominator = negative_denominator?;
    if !numerator_denominators
        .iter()
        .any(|den| compare_expr(ctx, *den, positive_denominator) == Ordering::Equal)
        || !numerator_denominators
            .iter()
            .any(|den| compare_expr(ctx, *den, negative_denominator) == Ordering::Equal)
    {
        return None;
    }

    let target_numerator = ctx.add(Expr::Add(positive_denominator, negative_denominator));
    let target_denominator = ctx.add(Expr::Sub(negative_denominator, positive_denominator));
    let target_expr = ctx.add(Expr::Div(target_numerator, target_denominator));
    Some((
        target_expr,
        positive_denominator,
        negative_denominator,
        denominator,
    ))
}

pub(super) fn matches_direct_reciprocal_sum_difference_nested_fraction_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (fraction_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((target_candidate, _, _, _)) =
            extract_reciprocal_sum_difference_nested_fraction_target_root(ctx, fraction_expr)
        else {
            continue;
        };
        if compare_expr(ctx, target_candidate, target_expr) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_small_rational_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    if matches_direct_symmetric_partial_fraction_zero_identity_root(ctx, expr) {
        return true;
    }
    if matches_direct_same_denominator_common_scaled_zero_identity_root(ctx, expr) {
        return true;
    }

    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3
        || !expr_contains_division_node_local(ctx, expr)
        || expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
    {
        return false;
    }

    let parent_ctx = crate::ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let rule = crate::rules::arithmetic::CollapseExactZeroThreeTermSubsetRule;
    let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) else {
        return false;
    };
    let zero = ctx.num(0);
    compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal
}

pub(super) fn matches_direct_sum_diff_cubes_quotient_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() == 4 {
        for candidate_index in 0..view.terms.len() {
            let (quotient_term, quotient_sign) = view.terms[candidate_index];
            let Some((a, b)) = extract_sum_diff_cubes_quotient_bases_root(ctx, quotient_term)
            else {
                continue;
            };

            let remaining_terms = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| (index != candidate_index).then_some(term))
                .collect::<Vec<_>>();

            let normalized_remaining_terms = if quotient_sign == Sign::Pos {
                remaining_terms
                    .into_iter()
                    .map(|(term_expr, term_sign)| {
                        let flipped_sign = match term_sign {
                            Sign::Pos => Sign::Neg,
                            Sign::Neg => Sign::Pos,
                        };
                        (term_expr, flipped_sign)
                    })
                    .collect::<Vec<_>>()
            } else {
                remaining_terms
            };

            let expected_poly = {
                let two = ctx.num(2);
                let a_sq = ctx.add(Expr::Pow(a, two));
                let ab = smart_mul(ctx, a, b);
                let b_sq = ctx.add(Expr::Pow(b, two));
                let ab_plus_b_sq = ctx.add(Expr::Add(ab, b_sq));
                ctx.add(Expr::Add(a_sq, ab_plus_b_sq))
            };
            let remaining_expr = build_signed_sum_expr_root(ctx, &normalized_remaining_terms);
            if compare_expr(ctx, remaining_expr, expected_poly) == Ordering::Equal {
                return true;
            }
        }
    }

    let parent_ctx = build_root_shortcut_parent_ctx(&SimplifyOptions::default(), ctx, expr);
    let rule = crate::rules::arithmetic::SubtractExpandedSumDiffCubesQuotientRule;
    let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) else {
        return false;
    };
    let zero = ctx.num(0);
    compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal
}

pub(super) fn extract_small_quotient_cancel_zero_candidate_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let view = AddView::from_expr(ctx, expr);
    if !(2..=4).contains(&view.terms.len())
        || !expr_contains_division_node_local(ctx, expr)
        || cas_ast::count_nodes(ctx, expr) > 24
        || cas_ast::collect_variables(ctx, expr).len() != 1
    {
        return None;
    }

    let quotient_index = view
        .terms
        .iter()
        .position(|(term, _)| matches!(ctx.get(*term), Expr::Div(_, _)))?;
    if view
        .terms
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != quotient_index)
        .any(|(_, (term, _))| expr_contains_division_node_local(ctx, *term))
    {
        return None;
    }

    let (quotient_term, quotient_sign) = view.terms[quotient_index];
    let quotient_expr = match quotient_sign {
        Sign::Pos => quotient_term,
        Sign::Neg => ctx.add(Expr::Neg(quotient_term)),
    };
    let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 4]> = view
        .terms
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, term)| (index != quotient_index).then_some(term))
        .collect();
    if remaining_terms.is_empty() {
        return None;
    }
    let negated_remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 4]> = remaining_terms
        .iter()
        .map(|(term, sign)| {
            (
                *term,
                match sign {
                    Sign::Pos => Sign::Neg,
                    Sign::Neg => Sign::Pos,
                },
            )
        })
        .collect();
    let target_expr = build_signed_sum_expr_root(ctx, &negated_remaining_terms);
    (!expr_contains_division_node_local(ctx, target_expr)).then_some((quotient_expr, target_expr))
}

pub(super) fn matches_small_quotient_cancel_zero_hot_candidate_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if !(2..=4).contains(&view.terms.len())
        || !expr_contains_division_node_local(ctx, expr)
        || cas_ast::count_nodes(ctx, expr) > 24
    {
        return false;
    }

    let Some(quotient_index) = view
        .terms
        .iter()
        .position(|(term, _)| matches!(ctx.get(*term), Expr::Div(_, _)))
    else {
        return false;
    };

    !view
        .terms
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != quotient_index)
        .any(|(_, (term, _))| expr_contains_division_node_local(ctx, *term))
}

pub(super) fn matches_small_quotient_cancel_zero_identity_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    extract_small_quotient_cancel_zero_candidate_root(ctx, expr).is_some_and(
        |(quotient_side, target_side)| {
            isolated_simplify_rewrites_to_target(options, ctx, quotient_side, target_side)
        },
    )
}

fn extract_sum_diff_cubes_quotient_bases_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator_view = AddView::from_expr(ctx, *numerator);
    let denominator_view = AddView::from_expr(ctx, *denominator);
    if numerator_view.terms.len() != 2 || denominator_view.terms.len() != 2 {
        return None;
    }

    let mut denominator_pos = None;
    let mut denominator_neg = None;
    for (term_expr, term_sign) in denominator_view.terms {
        match term_sign {
            Sign::Pos => denominator_pos = Some(term_expr),
            Sign::Neg => denominator_neg = Some(term_expr),
        }
    }

    let mut numerator_pos = None;
    let mut numerator_neg = None;
    for (term_expr, term_sign) in numerator_view.terms {
        let base = extract_plain_cube_base_root(ctx, term_expr)?;
        match term_sign {
            Sign::Pos => numerator_pos = Some(base),
            Sign::Neg => numerator_neg = Some(base),
        }
    }

    let (Some(denominator_pos), Some(denominator_neg), Some(numerator_pos), Some(numerator_neg)) = (
        denominator_pos,
        denominator_neg,
        numerator_pos,
        numerator_neg,
    ) else {
        return None;
    };

    (compare_expr(ctx, denominator_pos, numerator_pos) == Ordering::Equal
        && compare_expr(ctx, denominator_neg, numerator_neg) == Ordering::Equal)
        .then_some((denominator_pos, denominator_neg))
}

pub(super) fn matches_direct_quotient_pair_zero_difference_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let (lhs_term, rhs_term) = match (view.terms[0].1, view.terms[1].1) {
        (Sign::Pos, Sign::Neg) => (view.terms[0].0, view.terms[1].0),
        (Sign::Neg, Sign::Pos) => (view.terms[1].0, view.terms[0].0),
        _ => return false,
    };

    let (lhs_num, lhs_den) = match ctx.get(lhs_term) {
        Expr::Div(num, den) => (*num, *den),
        _ => return false,
    };
    let (rhs_num, rhs_den) = match ctx.get(rhs_term) {
        Expr::Div(num, den) => (*num, *den),
        _ => return false,
    };

    factors_match_by_equality_or_direct_pair_root(ctx, lhs_num, rhs_num)
        && factors_match_by_equality_or_direct_pair_root(ctx, lhs_den, rhs_den)
}

pub(super) fn matches_direct_or_isolated_quotient_pair_zero_difference_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let (lhs_term, rhs_term) = match (view.terms[0].1, view.terms[1].1) {
        (Sign::Pos, Sign::Neg) => (view.terms[0].0, view.terms[1].0),
        (Sign::Neg, Sign::Pos) => (view.terms[1].0, view.terms[0].0),
        _ => return false,
    };

    let (lhs_num, lhs_den) = match ctx.get(lhs_term) {
        Expr::Div(num, den) => (*num, *den),
        _ => return false,
    };
    let (rhs_num, rhs_den) = match ctx.get(rhs_term) {
        Expr::Div(num, den) => (*num, *den),
        _ => return false,
    };

    if !factors_match_by_equality_or_direct_pair_root(ctx, lhs_num, rhs_num) {
        return false;
    }
    if factors_match_by_equality_or_direct_pair_root(ctx, lhs_den, rhs_den) {
        return true;
    }

    let denominator_difference = ctx.add(Expr::Sub(lhs_den, rhs_den));
    if matches_direct_small_zero_or_known_pair_residual_root(ctx, denominator_difference) {
        return true;
    }

    isolated_simplify_rewrites_to_zero(options, ctx, denominator_difference)
}

pub(super) fn try_standard_sum_diff_cubes_fraction_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let Expr::Div(_, _) = ctx.get(expr) else {
        return None;
    };

    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    let rule = crate::rules::algebra::CancelSumDiffCubesFractionRule;
    let rewrite = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx)?;
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Cancel Sum/Difference of Cubes Fraction",
        collect_steps,
    ))
}

fn run_shifted_quotient_rebuilt_root_shortcut_simplify(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    before: ExprId,
    rewritten: ExprId,
    collect_steps: bool,
) -> (ExprId, Vec<Step>) {
    run_named_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        before,
        rewritten,
        "Collapse Shifted Quotient of Equivalent Expressions",
        "Collapse Shifted Quotient of Equivalent Expressions",
        collect_steps,
    )
}

fn child_matches_exact_zero_same_denominator_direct_or_passthrough_pair(
    ctx: &mut Context,
    child: ExprId,
) -> bool {
    let Some((_den, lhs_core, rhs_core)) = extract_same_denominator_direct_pair_root(ctx, child)
    else {
        return false;
    };

    if matches_known_direct_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_half_angle_binomial_square_pair_root(ctx, lhs_core, rhs_core)
    {
        return true;
    }

    extract_shared_additive_passthrough_pair_cores_root(ctx, lhs_core, rhs_core).is_some_and(
        |(lhs_residual, rhs_residual)| {
            matches_known_direct_pair_root(ctx, lhs_residual, rhs_residual)
                || matches_direct_half_angle_binomial_square_pair_root(
                    ctx,
                    lhs_residual,
                    rhs_residual,
                )
        },
    )
}

fn matches_shifted_quotient_exact_one_direct_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    compare_expr(ctx, lhs_core, rhs_core) == Ordering::Equal
        || matches_known_direct_pair_root(ctx, lhs_core, rhs_core)
        || crate::rules::arithmetic::try_build_direct_trig_double_angle_cos_variant_equivalence_rewrite(
            ctx,
            lhs_core,
            rhs_core,
        )
        .is_some()
        || matches_direct_half_angle_binomial_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_pythagorean_factor_form_pair_root(ctx, lhs_core, rhs_core)
}

pub(super) fn matches_shifted_quotient_exact_one_direct_or_passthrough_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    if matches_shifted_quotient_exact_one_direct_pair_root(ctx, lhs_core, rhs_core) {
        return true;
    }

    extract_shared_additive_passthrough_pair_cores_root(ctx, lhs_core, rhs_core).is_some_and(
        |(lhs_residual, rhs_residual)| {
            matches_shifted_quotient_exact_one_passthrough_residual_pair_root(
                ctx,
                lhs_residual,
                rhs_residual,
            )
        },
    )
}

fn matches_shifted_quotient_exact_one_passthrough_residual_pair_root(
    ctx: &mut Context,
    lhs_residual: ExprId,
    rhs_residual: ExprId,
) -> bool {
    matches_shifted_quotient_exact_one_direct_pair_root(ctx, lhs_residual, rhs_residual)
        || crate::rules::arithmetic::try_build_direct_sum_diff_cubes_quotient_equivalence_rewrite(
            ctx,
            lhs_residual,
            rhs_residual,
        )
        .is_some()
}

pub(super) fn matches_shifted_quotient_exact_one_root_gate_candidate(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    if matches_direct_small_zero_pair_root(ctx, lhs_core, rhs_core) {
        return true;
    }

    if matches_shifted_quotient_exact_one_direct_pair_root(ctx, lhs_core, rhs_core) {
        return true;
    }

    if crate::rules::arithmetic::try_build_direct_sub_fraction_combination_equivalence_rewrite(
        ctx, lhs_core, rhs_core,
    )
    .is_some()
    {
        return true;
    }

    if crate::rules::arithmetic::matches_shifted_quotient_fraction_residual_narrow_pair_candidate(
        ctx, lhs_core, rhs_core,
    ) {
        return true;
    }

    is_potential_shifted_quotient_exact_one_direct_pair_side_root(ctx, lhs_core)
        && is_potential_shifted_quotient_exact_one_direct_pair_side_root(ctx, rhs_core)
        && matches_shifted_quotient_exact_one_direct_or_passthrough_pair_root(
            ctx, lhs_core, rhs_core,
        )
}

pub(super) fn matches_shifted_quotient_direct_small_zero_hot_gate_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let side_meta = |ctx: &mut Context, expr: ExprId| {
        if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
            return None;
        };

        let add_view = AddView::from_expr(ctx, expr);
        Some((
            add_view.terms.len(),
            scan_hot_direct_small_zero_family_flags_root(ctx, expr),
        ))
    };

    let is_fast_exact_zero_side =
        |ctx: &mut Context,
         expr: ExprId,
         side_meta: Option<(usize, HotDirectSmallZeroFamilyFlags)>| {
            let Some((term_len, flags)) = side_meta else {
                return false;
            };
            if term_len != 3 {
                return false;
            }

            let has_log = flags.has_log;
            let has_division = flags.has_division;
            let has_trig = flags.has_trig;
            let has_hyperbolic = flags.has_hyperbolic;

            (has_log
                && !has_division
                && !has_trig
                && !has_hyperbolic
                && matches_direct_log_product_contract_zero_identity_root(ctx, expr))
                || (has_division
                    && !has_log
                    && !has_trig
                    && !has_hyperbolic
                    && matches_direct_depth_three_unit_continued_fraction_zero_identity_terms_root(
                        ctx,
                        &AddView::from_expr(ctx, expr).terms,
                    ))
        };

    let is_fast_hot_partner_side =
        |ctx: &mut Context,
         expr: ExprId,
         side_meta: Option<(usize, HotDirectSmallZeroFamilyFlags)>| {
            let Some((term_len, flags)) = side_meta else {
                return false;
            };

            if term_len == 2
                && flags.has_division
                && !flags.has_log
                && !flags.has_trig
                && !flags.has_hyperbolic
                && matches_small_quotient_cancel_zero_hot_candidate_root(ctx, expr)
            {
                return true;
            }

            if term_len == 3
                && !flags.has_log
                && !flags.has_trig
                && !flags.has_hyperbolic
                && !flags.has_division
                && matches_direct_sophie_germain_zero_hot_candidate_root(ctx, expr)
            {
                return true;
            }

            matches_hot_direct_small_zero_family_with_flags_root(ctx, expr, flags)
        };

    let lhs_meta = run_profiled_orchestrator_section("root.div.00a.meta.scan.lhs", None, || {
        side_meta(ctx, lhs_core)
    });
    let rhs_meta = run_profiled_orchestrator_section("root.div.00a.meta.scan.rhs", None, || {
        side_meta(ctx, rhs_core)
    });
    let lhs_is_fast_exact_zero_side =
        run_profiled_orchestrator_bool_section("root.div.00a.fast_exact_zero_side.lhs", || {
            is_fast_exact_zero_side(ctx, lhs_core, lhs_meta)
        });
    let rhs_is_fast_exact_zero_side =
        run_profiled_orchestrator_bool_section("root.div.00a.fast_exact_zero_side.rhs", || {
            is_fast_exact_zero_side(ctx, rhs_core, rhs_meta)
        });

    (lhs_is_fast_exact_zero_side
        && run_profiled_orchestrator_bool_section("root.div.00a.hot_partner_side.rhs", || {
            is_fast_hot_partner_side(ctx, rhs_core, rhs_meta)
        }))
        || (rhs_is_fast_exact_zero_side
            && run_profiled_orchestrator_bool_section("root.div.00a.hot_partner_side.lhs", || {
                is_fast_hot_partner_side(ctx, lhs_core, lhs_meta)
            }))
}

pub(super) fn extract_shifted_quotient_positive_one_passthrough_cores_root(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<(ExprId, ExprId)> {
    strip_positive_one_passthrough_root(ctx, numerator)
        .zip(strip_positive_one_passthrough_root(ctx, denominator))
}

pub(super) fn matches_shifted_quotient_nested_zero_fast_gate_candidate_from_cores_root(
    ctx: &mut Context,
    numerator_core: ExprId,
    denominator_core: ExprId,
) -> bool {
    let is_same_denominator_zero = |ctx: &mut Context, expr: ExprId| {
        matches_structural_same_denominator_distribution_zero_identity_root(ctx, expr)
    };

    (is_same_denominator_zero(ctx, numerator_core)
        && is_small_exact_zero_base_family_root(ctx, denominator_core))
        || (is_same_denominator_zero(ctx, denominator_core)
            && is_small_exact_zero_base_family_root(ctx, numerator_core))
        || (is_small_exact_zero_base_family_root(ctx, numerator_core)
            && is_small_exact_zero_base_family_root(ctx, denominator_core))
}

#[cfg(test)]
pub(super) fn matches_shifted_quotient_nested_zero_fast_gate_candidate_root(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
) -> bool {
    extract_shifted_quotient_positive_one_passthrough_cores_root(ctx, numerator, denominator)
        .is_some_and(|(numerator_core, denominator_core)| {
            matches_shifted_quotient_nested_zero_fast_gate_candidate_from_cores_root(
                ctx,
                numerator_core,
                denominator_core,
            )
        })
}

pub(super) fn is_potential_shifted_quotient_exact_one_direct_pair_side_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            is_potential_shifted_quotient_exact_one_direct_pair_side_root(ctx, inner)
        }
        Expr::Number(_)
        | Expr::Constant(_)
        | Expr::Function(_, _)
        | Expr::Div(_, _)
        | Expr::Pow(_, _) => true,
        Expr::Mul(_, _) => {
            expr_contains_builtin_function_local(ctx, expr)
                || expr_contains_sqrt_or_half_power_local(ctx, expr)
                || extract_direct_factored_linear_shift_pair_root(ctx, expr).is_some()
                || extract_direct_two_linear_shift_product_root(ctx, expr).is_some()
                || extract_direct_three_linear_shift_product_root(ctx, expr).is_some()
                || (cas_ast::collect_variables(ctx, expr).is_empty()
                    && cas_ast::count_nodes(ctx, expr) <= 16)
        }
        Expr::Add(_, _) | Expr::Sub(_, _) => {
            let terms = AddView::from_expr(ctx, expr).terms;
            terms.len() <= 4
                && (expr_contains_builtin_function_local(ctx, expr)
                    || has_numeric_pythagorean_complement_pair(ctx, expr)
                    || extract_direct_expanded_linear_shift_pair_root(ctx, expr).is_some()
                    || terms.iter().any(|(term, _)| {
                        matches!(
                            ctx.get(*term),
                            Expr::Div(_, _) | Expr::Pow(_, _) | Expr::Function(_, _)
                        )
                    })
                    || (cas_ast::collect_variables(ctx, expr).is_empty()
                        && cas_ast::count_nodes(ctx, expr) <= 16))
        }
        Expr::Variable(_) | Expr::Matrix { .. } | Expr::SessionRef(_) | Expr::Hold(_) => false,
    }
}

fn child_matches_exact_zero_two_factor_or_quotient_pair(ctx: &mut Context, child: ExprId) -> bool {
    matches!(ctx.get(child), Expr::Add(_, _) | Expr::Sub(_, _))
        && (matches_direct_two_factor_product_pair_zero_difference_root(ctx, child)
            || matches_direct_quotient_pair_zero_difference_root(ctx, child))
}

pub(super) fn try_standard_collapsed_fraction_direct_pair_factor_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if !(2..=4).contains(&factors.len()) {
        return None;
    }
    if !factors
        .iter()
        .copied()
        .any(|factor| is_potential_collapsed_fraction_direct_pair_source_root(ctx, factor))
    {
        return None;
    }

    for partner_index in 0..factors.len() {
        if !is_potential_collapsed_fraction_direct_pair_source_root(ctx, factors[partner_index]) {
            continue;
        }
        let Some(partner_canonical) =
            canonicalize_direct_pair_factor_root(ctx, factors[partner_index])
        else {
            continue;
        };
        if extract_collapsed_successive_unit_fractions_arg_root(ctx, partner_canonical).is_none() {
            continue;
        }
        let remaining_factors: Vec<_> = factors
            .iter()
            .enumerate()
            .filter_map(|(index, factor)| (index != partner_index).then_some(*factor))
            .collect();
        let combined_factor = build_mul_expr_from_factors_root(ctx, &remaining_factors);
        let Some(factor_canonical) = canonicalize_direct_pair_factor_root(ctx, combined_factor)
        else {
            continue;
        };

        let partner_changed =
            compare_expr(ctx, partner_canonical, factors[partner_index]) != Ordering::Equal;
        let factor_changed =
            compare_expr(ctx, factor_canonical, combined_factor) != Ordering::Equal;
        if !partner_changed && !factor_changed {
            continue;
        }

        let rewritten =
            build_mul_expr_from_factors_root(ctx, &[partner_canonical, factor_canonical]);
        let rewrite = crate::rule::Rewrite::new(rewritten).desc("Canonical Direct Pair Product");
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Canonical Direct Pair Product",
            collect_steps,
        ));
    }

    None
}

pub(super) fn try_standard_collapsed_fraction_factored_numerator_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for fraction_index in 0..2 {
        let partner_index = 1 - fraction_index;
        let Some(denominator) =
            extract_unit_fraction_denominator_root(ctx, factors[fraction_index])
        else {
            continue;
        };
        if extract_consecutive_product_core_root(ctx, denominator).is_none() {
            continue;
        }

        let partner = factors[partner_index];
        let partner_factored = factor_sum_diff_cubes_partner_root(ctx, partner)
            .or_else(|| factor_small_linear_shift_product_partner_root(ctx, partner))
            .or_else(|| factor_known_small_polynomial_partner_root(ctx, partner));
        let Some(partner_factored) = partner_factored else {
            continue;
        };
        if compare_expr(ctx, partner_factored, partner) == Ordering::Equal {
            continue;
        }

        let rewritten = ctx.add(Expr::Div(partner_factored, denominator));
        let rewrite = crate::rule::Rewrite::new(rewritten)
            .desc("Canonizar numerador factorizable sobre fracción consecutiva colapsada");
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Collapsed Fraction Factored Numerator",
            collect_steps,
        ));
    }

    None
}

pub(super) fn try_standard_collapsed_fraction_partner_canonicalization_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for fraction_index in 0..2 {
        let partner_index = 1 - fraction_index;
        let Some(denominator) =
            extract_unit_fraction_denominator_root(ctx, factors[fraction_index])
        else {
            continue;
        };
        if extract_consecutive_product_core_root(ctx, denominator).is_none() {
            continue;
        }

        let partner = factors[partner_index];
        let partner_canonical = canonicalize_direct_pair_factor_root(ctx, partner)
            .or_else(|| factor_small_linear_shift_product_partner_root(ctx, partner))
            .or_else(|| factor_known_small_polynomial_partner_root(ctx, partner))
            .or_else(|| {
                let rewrite = try_rewrite_automatic_factor_expr(ctx, partner)?;
                let factored = strip_multiplicative_one_root(ctx, rewrite.rewritten);
                (compare_expr(ctx, factored, partner) != Ordering::Equal).then_some(factored)
            });

        let Some(partner_canonical) = partner_canonical else {
            continue;
        };
        if compare_expr(ctx, partner_canonical, partner) == Ordering::Equal {
            continue;
        }

        let rewritten = if fraction_index == 0 {
            ctx.add(Expr::Mul(factors[fraction_index], partner_canonical))
        } else {
            ctx.add(Expr::Mul(partner_canonical, factors[fraction_index]))
        };
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            rewritten,
            "Canonical Collapsed Fraction Partner",
            "Canonical Collapsed Fraction Partner",
            collect_steps,
        ));
    }

    None
}

pub(super) fn try_standard_tangent_addition_fraction_product_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (index, factor) in factors.iter().copied().enumerate() {
        if extract_direct_tangent_addition_fraction_target_root(ctx, factor).is_none() {
            continue;
        }
        let Expr::Div(numerator, denominator) = ctx.get(factor) else {
            continue;
        };
        let numerator = *numerator;
        let denominator = *denominator;

        let rewritten_numerator_factors = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(factor_index, other_factor)| {
                (factor_index != index).then_some(other_factor)
            })
            .chain(std::iter::once(numerator))
            .collect::<Vec<_>>();
        let rewritten_numerator =
            build_mul_expr_from_factors_root(ctx, &rewritten_numerator_factors);
        let rewritten = ctx.add(Expr::Div(rewritten_numerator, denominator));
        let rewrite = crate::rule::Rewrite::new(rewritten)
            .desc("Colapsar producto sobre fracción de suma de tangentes");
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Tangent Addition Fraction Product",
            collect_steps,
        ));
    }

    None
}

fn is_potential_collapsed_fraction_direct_pair_source_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    match ctx.get(expr) {
        Expr::Add(_, _) => {
            extract_addition_of_successive_unit_fractions_arg_root(ctx, expr).is_some()
        }
        Expr::Div(_, _) => {
            extract_collapsed_successive_unit_fractions_arg_root(ctx, expr).is_some()
        }
        Expr::Neg(inner) => is_potential_collapsed_fraction_direct_pair_source_root(ctx, *inner),
        _ => false,
    }
}

pub(super) fn try_standard_same_denominator_distribution_pair_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        || !expr_contains_division_node_local(ctx, expr)
    {
        return None;
    }

    let total_terms = AddView::from_expr(ctx, expr).terms.len();
    if !(5..=7).contains(&total_terms) {
        return None;
    }

    let (lhs, rhs) = match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => (lhs, rhs),
        _ => return None,
    };

    let child_shape_ok = |ctx: &mut Context, child: ExprId| {
        matches!(ctx.get(child), Expr::Add(_, _) | Expr::Sub(_, _))
            && (2..=4).contains(&AddView::from_expr(ctx, child).terms.len())
    };
    if !child_shape_ok(ctx, lhs) || !child_shape_ok(ctx, rhs) {
        return None;
    }

    let same_denominator_zero_child = |ctx: &mut Context, child: ExprId| {
        child_matches_exact_zero_common_scale_rule(options, ctx, child)
            || matches_direct_same_denominator_common_scaled_zero_identity_root(ctx, child)
    };

    let pair = if same_denominator_zero_child(ctx, lhs)
        && exact_zero_leaf_rewrites_to_zero_root(options, ctx, rhs)
    {
        Some((lhs, rhs))
    } else if same_denominator_zero_child(ctx, rhs)
        && exact_zero_leaf_rewrites_to_zero_root(options, ctx, lhs)
    {
        Some((rhs, lhs))
    } else {
        None
    }?;

    let zero = ctx.num(0);
    if collect_steps && matches!(ctx.get(expr), Expr::Add(_, _)) {
        if let Some(steps) =
            try_build_chunk_pair_zero_shortcut_steps_root(options, ctx, expr, pair.0, pair.1)
        {
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

pub(super) fn try_standard_shifted_quotient_exact_one_shortcut_with_direct_small_zero_hint(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
    direct_small_zero_prevalidated: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if direct_small_zero_prevalidated
        || is_direct_small_zero_shifted_quotient_candidate_root(ctx, expr)
    {
        let one = ctx.num(1);
        let rewrite =
            crate::rule::Rewrite::with_local(one, "Exact Zero Core Quotient Identity", expr, one);
        return Some(finish_root_shortcut_with_rewrite_meta(
            ctx,
            expr,
            rewrite,
            "Collapse Shifted Quotient of Equivalent Expressions",
            collect_steps,
        ));
    }

    if is_guarded_small_zero_shifted_quotient_candidate_root(ctx, expr) {
        let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
        let rule = crate::rules::arithmetic::CollapseExactOneShiftedQuotientRule;
        if let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) {
            return Some(finish_root_shortcut_with_rewrite_meta(
                ctx,
                expr,
                rewrite,
                "Collapse Shifted Quotient of Equivalent Expressions",
                collect_steps,
            ));
        }
    }

    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let pair_sample = profiling.then(|| {
        format!(
            "{}  ||  {}",
            render_expr_for_orchestrator_profile(ctx, numerator),
            render_expr_for_orchestrator_profile(ctx, denominator)
        )
    });
    let passthrough_cores = strip_positive_one_passthrough_root(ctx, numerator)
        .zip(strip_positive_one_passthrough_root(ctx, denominator));
    if profiling {
        if let Some(sample) = pair_sample.clone() {
            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                "root.div.02.shifted_quotient_exact_one.passthrough_gate",
                sample,
            );
        }
        run_profiled_orchestrator_bool_section(
            "root.div.02.shifted_quotient_exact_one.passthrough_gate",
            || passthrough_cores.is_some(),
        );
    }
    let (numerator_core, denominator_core) = passthrough_cores?;

    if matches_shifted_quotient_exact_one_direct_pair_root(ctx, numerator_core, denominator_core) {
        let one = ctx.num(1);
        return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            one,
            collect_steps,
        ));
    }
    if let Some((numerator_residual, denominator_residual)) =
        extract_shared_additive_passthrough_pair_cores_root(ctx, numerator_core, denominator_core)
    {
        if matches_shifted_quotient_exact_one_passthrough_residual_pair_root(
            ctx,
            numerator_residual,
            denominator_residual,
        ) {
            let one = ctx.num(1);
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }
    }
    if ((expr_contains_reciprocal_trig_builtin_local(ctx, numerator_core)
        && expr_contains_trig_or_hyperbolic_builtin_local(ctx, denominator_core))
        || (expr_contains_reciprocal_trig_builtin_local(ctx, denominator_core)
            && expr_contains_trig_or_hyperbolic_builtin_local(ctx, numerator_core)))
        && child_isolated_exact_zero(options, ctx, numerator_core)
        && child_isolated_exact_zero(options, ctx, denominator_core)
    {
        let one = ctx.num(1);
        return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            one,
            collect_steps,
        ));
    }
    if is_nested_additive_log_residual_pair_root(ctx, numerator_core, denominator_core) {
        return None;
    }

    // Under an active wall-clock budget, the generic shifted-quotient exact-one
    // probe is not worth paying on nested-fraction pairs after the cheap direct
    // checks above have already failed. The expensive proof path frequently
    // misses on these shapes and can blow past the interactive deadline.
    if options.deadline.is_some()
        && (expr_contains_division_node_local(ctx, numerator)
            || expr_contains_division_node_local(ctx, denominator))
    {
        return None;
    }

    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    let rule = crate::rules::arithmetic::CollapseExactOneShiftedQuotientRule;
    if let Some(sample) = pair_sample.clone() {
        crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
            "root.div.02.shifted_quotient_exact_one.rule_apply",
            sample,
        );
    }
    let rewrite = if let Some(rewrite) =
        run_profiled_root_shortcut("root.div.02.shifted_quotient_exact_one.rule_apply", || {
            crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx)
        }) {
        rewrite
    } else {
        if let Some(sample) = pair_sample.clone() {
            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                "root.div.02.shifted_quotient_exact_one.fallback_residual_zero",
                sample,
            );
        }
        run_profiled_root_shortcut(
            "root.div.02.shifted_quotient_exact_one.fallback_residual_zero",
            || {
                let residual_difference = ctx.add(Expr::Sub(numerator_core, denominator_core));

                let mut residual_simplifier = crate::Simplifier::with_default_rules();
                std::mem::swap(&mut residual_simplifier.context, ctx);
                let mut residual_orchestrator = Orchestrator::new();
                residual_orchestrator.options = SimplifyOptions {
                    collect_steps: false,
                    suppress_depth_overflow_warnings: true,
                    ..options.clone()
                };
                let (residual_result, _residual_steps, _stats) = residual_orchestrator
                    .simplify_pipeline(residual_difference, &mut residual_simplifier);
                std::mem::swap(&mut residual_simplifier.context, ctx);

                let zero = ctx.num(0);
                if compare_expr(ctx, residual_result, zero) != Ordering::Equal {
                    return None;
                }

                Some(crate::rule::Rewrite::with_local(
                    ctx.add(Expr::Div(denominator, denominator)),
                    "Equivalent Residual Cancellation",
                    numerator,
                    denominator,
                ))
            },
        )?
    };

    let one = ctx.num(1);
    if let Expr::Div(numerator, denominator) = ctx.get(rewrite.new_expr) {
        if compare_expr(ctx, *numerator, *denominator) == Ordering::Equal {
            let mut shortcut_steps = Vec::new();
            if collect_steps {
                shortcut_steps.push(build_root_shortcut_step_from_rewrite(
                    ctx,
                    expr,
                    &rewrite,
                    "Collapse Shifted Quotient of Equivalent Expressions",
                ));
                shortcut_steps.push(build_root_shortcut_compact_step(
                    rewrite.new_expr,
                    one,
                    "Cancelar numerador y denominador iguales",
                    "Simplificar fracción",
                ));
            }
            return Some((one, shortcut_steps));
        }
    }

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

    if compare_expr(ctx, result, one) != Ordering::Equal {
        return None;
    }

    let mut shortcut_steps = Vec::new();
    if collect_steps {
        shortcut_steps.push(build_root_shortcut_step_from_rewrite(
            ctx,
            expr,
            &rewrite,
            "Collapse Shifted Quotient of Equivalent Expressions",
        ));
        shortcut_steps.extend(inner_steps);
    }

    Some((result, shortcut_steps))
}

pub(super) fn try_standard_shifted_quotient_exact_one_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    try_standard_shifted_quotient_exact_one_shortcut_with_direct_small_zero_hint(
        options,
        ctx,
        expr,
        collect_steps,
        false,
    )
}

pub(super) fn try_standard_shifted_quotient_nested_zero_core_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let profile_nested_zero =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    macro_rules! profiled_nested_zero_bool {
        ($name:literal, $body:expr) => {{
            if profile_nested_zero {
                run_profiled_orchestrator_bool_section($name, || $body)
            } else {
                $body
            }
        }};
    }

    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    let one = ctx.num(1);

    let numerator_core = strip_positive_one_passthrough_root(ctx, numerator);
    let denominator_core = strip_positive_one_passthrough_root(ctx, denominator);

    if let (Some(numerator_core), Some(denominator_core)) = (numerator_core, denominator_core) {
        if is_small_positive_additive_trig_passthrough_core_root(ctx, numerator_core)
            && is_small_positive_additive_trig_passthrough_core_root(ctx, denominator_core)
        {
            return None;
        }

        let direct_pair_family = profiled_nested_zero_bool!(
            "root.div.03a.nested_zero.direct_pair_family",
            profiled_nested_zero_bool!(
                "root.div.03a0.nested_zero.direct_pair_family_candidate_gate",
                is_potential_nested_zero_direct_pair_family_pair_root(
                    ctx,
                    numerator_core,
                    denominator_core,
                )
            ) && {
                matches_direct_trig_product_to_sum_sin_sin_pair_root(
                    ctx,
                    numerator_core,
                    denominator_core,
                ) || matches_direct_trig_product_to_sum_sin_cos_pair_root(
                    ctx,
                    numerator_core,
                    denominator_core,
                ) || matches_direct_trig_product_to_sum_cos_cos_pair_root(
                    ctx,
                    numerator_core,
                    denominator_core,
                ) || matches_direct_nested_fraction_simplified_pair_root(
                    ctx,
                    numerator_core,
                    denominator_core,
                ) || matches_direct_hyperbolic_sinh_sum_to_product_pair_root(
                    ctx,
                    numerator_core,
                    denominator_core,
                ) || matches_direct_hyperbolic_cosh_sum_to_product_pair_root(
                    ctx,
                    numerator_core,
                    denominator_core,
                ) || matches_direct_hyperbolic_cosh_difference_to_product_pair_root(
                    ctx,
                    numerator_core,
                    denominator_core,
                ) || matches_direct_recursive_hyperbolic_sinh_sum_pair_root(
                    ctx,
                    numerator_core,
                    denominator_core,
                ) || matches_direct_recursive_hyperbolic_cosh_sum_pair_root(
                    ctx,
                    numerator_core,
                    denominator_core,
                ) || matches_direct_cos_square_diff_pair_root(ctx, numerator_core, denominator_core)
                    || matches_direct_trig_binomial_square_pair_root(
                        ctx,
                        numerator_core,
                        denominator_core,
                    )
                    || matches_direct_angle_sum_diff_pair_root(
                        ctx,
                        numerator_core,
                        denominator_core,
                    )
            }
        );
        if direct_pair_family {
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }

        let both_direct_small_zero_or_known_pair_base = profiled_nested_zero_bool!(
            "root.div.03a1.nested_zero.both_direct_small_zero_or_known_pair_base",
            is_small_exact_zero_base_family_root(ctx, numerator_core)
                && is_small_exact_zero_base_family_root(ctx, denominator_core)
        );
        if both_direct_small_zero_or_known_pair_base {
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }

        let both_small_exact_zero_leaf_possible =
            is_potential_small_exact_zero_leaf_root(ctx, numerator_core)
                && is_potential_small_exact_zero_leaf_root(ctx, denominator_core);
        let small_exact_zero_leaf_pair_sample =
            (profile_nested_zero && both_small_exact_zero_leaf_possible).then(|| {
                format!(
                    "{}  ||  {}",
                    render_expr_for_orchestrator_profile(ctx, numerator_core),
                    render_expr_for_orchestrator_profile(ctx, denominator_core)
                )
            });
        if let Some(sample) = small_exact_zero_leaf_pair_sample {
            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                "root.div.03b2.nested_zero.both_small_exact_zero_leaf",
                sample,
            );
        }
        let both_small_exact_zero_leaf = both_small_exact_zero_leaf_possible
            && profiled_nested_zero_bool!(
                "root.div.03b2.nested_zero.both_small_exact_zero_leaf",
                {
                    child_is_small_exact_zero_leaf_root(options, ctx, numerator_core)
                        && child_is_small_exact_zero_leaf_root(options, ctx, denominator_core)
                }
            );
        if both_small_exact_zero_leaf {
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }

        let both_direct_small_zero =
            profiled_nested_zero_bool!("root.div.03b.nested_zero.both_direct_small_zero", {
                is_potential_direct_small_zero_identity_root(ctx, numerator_core)
                    && is_potential_direct_small_zero_identity_root(ctx, denominator_core)
                    && matches_direct_small_zero_identity_root(ctx, numerator_core)
                    && matches_direct_small_zero_identity_root(ctx, denominator_core)
            });
        if both_direct_small_zero {
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }

        let reciprocal_trig_both_child_zero = profiled_nested_zero_bool!(
            "root.div.03c.nested_zero.reciprocal_trig_both_child_zero",
            {
                ((expr_contains_reciprocal_trig_builtin_local(ctx, numerator_core)
                    && expr_contains_trig_or_hyperbolic_builtin_local(ctx, denominator_core))
                    || (expr_contains_reciprocal_trig_builtin_local(ctx, denominator_core)
                        && expr_contains_trig_or_hyperbolic_builtin_local(ctx, numerator_core)))
                    && child_isolated_exact_zero(options, ctx, numerator_core)
                    && child_isolated_exact_zero(options, ctx, denominator_core)
            }
        );
        if reciprocal_trig_both_child_zero {
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }

        let zero_child_supported_partner = profiled_nested_zero_bool!(
            "root.div.03d.nested_zero.zero_child_supported_partner",
            {
                let mut matched = false;
                if expr_contains_trig_or_hyperbolic_builtin_local(ctx, numerator_core)
                    && is_supported_nested_zero_child_partner(ctx, denominator_core)
                {
                    let partner_family =
                        supported_nested_zero_child_partner_profile_family(ctx, denominator_core);
                    let label = match partner_family {
                        "log" => {
                            "root.div.03d1.nested_zero.zero_child_supported_partner.partner_family.log"
                        }
                        "nonlog_additive" => {
                            "root.div.03d2.nested_zero.zero_child_supported_partner.partner_family.nonlog_additive"
                        }
                        _ => {
                            "root.div.03d3.nested_zero.zero_child_supported_partner.partner_family.other"
                        }
                    };
                    if profile_nested_zero {
                        crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                            label,
                            format!(
                                "{}  ||  {}",
                                render_expr_for_orchestrator_profile(ctx, numerator_core),
                                render_expr_for_orchestrator_profile(ctx, denominator_core)
                            ),
                        );
                    }
                    matched |= if profile_nested_zero {
                        run_profiled_orchestrator_bool_section(label, || {
                            child_matches_direct_or_isolated_exact_zero(
                                options,
                                ctx,
                                numerator_core,
                            ) && supported_nested_zero_partner_rewrites_to_zero(
                                options,
                                ctx,
                                denominator_core,
                            )
                        })
                    } else {
                        child_matches_direct_or_isolated_exact_zero(options, ctx, numerator_core)
                            && supported_nested_zero_partner_rewrites_to_zero(
                                options,
                                ctx,
                                denominator_core,
                            )
                    };
                }
                if !matched
                    && expr_contains_trig_or_hyperbolic_builtin_local(ctx, denominator_core)
                    && is_supported_nested_zero_child_partner(ctx, numerator_core)
                {
                    let partner_family =
                        supported_nested_zero_child_partner_profile_family(ctx, numerator_core);
                    let label = match partner_family {
                        "log" => {
                            "root.div.03d1.nested_zero.zero_child_supported_partner.partner_family.log"
                        }
                        "nonlog_additive" => {
                            "root.div.03d2.nested_zero.zero_child_supported_partner.partner_family.nonlog_additive"
                        }
                        _ => {
                            "root.div.03d3.nested_zero.zero_child_supported_partner.partner_family.other"
                        }
                    };
                    if profile_nested_zero {
                        crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                            label,
                            format!(
                                "{}  ||  {}",
                                render_expr_for_orchestrator_profile(ctx, denominator_core),
                                render_expr_for_orchestrator_profile(ctx, numerator_core)
                            ),
                        );
                    }
                    matched |= if profile_nested_zero {
                        run_profiled_orchestrator_bool_section(label, || {
                            child_matches_direct_or_isolated_exact_zero(
                                options,
                                ctx,
                                denominator_core,
                            ) && supported_nested_zero_partner_rewrites_to_zero(
                                options,
                                ctx,
                                numerator_core,
                            )
                        })
                    } else {
                        child_matches_direct_or_isolated_exact_zero(options, ctx, denominator_core)
                            && supported_nested_zero_partner_rewrites_to_zero(
                                options,
                                ctx,
                                numerator_core,
                            )
                    };
                }
                matched
            }
        );
        if zero_child_supported_partner {
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }

        let narrow_mixed_double_angle =
            profiled_nested_zero_bool!("root.div.03e.nested_zero.narrow_mixed_double_angle", {
                (matches_narrow_trig_mixed_double_angle_zero_candidate_root(ctx, numerator_core)
                    && child_isolated_exact_zero(options, ctx, numerator_core)
                    && (matches_direct_small_zero_identity_root(ctx, denominator_core)
                        || (is_small_trig_or_hyperbolic_zero_child(
                            options,
                            ctx,
                            denominator_core,
                        ) && child_isolated_exact_zero(options, ctx, denominator_core))))
                    || (matches_narrow_trig_mixed_double_angle_zero_candidate_root(
                        ctx,
                        denominator_core,
                    ) && child_isolated_exact_zero(options, ctx, denominator_core)
                        && (matches_direct_small_zero_identity_root(ctx, numerator_core)
                            || (is_small_trig_or_hyperbolic_zero_child(
                                options,
                                ctx,
                                numerator_core,
                            ) && child_isolated_exact_zero(options, ctx, numerator_core))))
            });
        if narrow_mixed_double_angle {
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }

        let narrow_small_trig_zero_pair =
            (is_potential_small_trig_zero_identity_root(ctx, numerator_core)
                && matches_direct_small_zero_identity_root(ctx, numerator_core)
                && ((is_potential_small_trig_zero_identity_root(ctx, denominator_core)
                    && matches_direct_small_zero_identity_root(ctx, denominator_core))
                    || is_small_trig_or_hyperbolic_zero_child(options, ctx, denominator_core)))
                || (is_potential_small_trig_zero_identity_root(ctx, denominator_core)
                    && matches_direct_small_zero_identity_root(ctx, denominator_core)
                    && ((is_potential_small_trig_zero_identity_root(ctx, numerator_core)
                        && matches_direct_small_zero_identity_root(ctx, numerator_core))
                        || is_small_trig_or_hyperbolic_zero_child(options, ctx, numerator_core)));
        let narrow_small_trig_zero_pair = profiled_nested_zero_bool!(
            "root.div.03f.nested_zero.narrow_small_trig_pair",
            narrow_small_trig_zero_pair
        );
        if narrow_small_trig_zero_pair {
            return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                one,
                collect_steps,
            ));
        }

        if expr_contains_trig_or_hyperbolic_builtin_local(ctx, numerator_core)
            && expr_contains_trig_or_hyperbolic_builtin_local(ctx, denominator_core)
        {
            let residual_difference = ctx.add(Expr::Sub(numerator_core, denominator_core));
            let residual_sample = profile_nested_zero
                .then(|| render_expr_for_orchestrator_profile(ctx, residual_difference));
            if profile_nested_zero {
                crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                    "root.div.03g1.nested_zero.residual_difference_phase_shift_zero",
                    residual_sample.clone().unwrap_or_default(),
                );
                crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                    "root.div.03g2.nested_zero.residual_difference_isolated_zero_fallback",
                    residual_sample.clone().unwrap_or_default(),
                );
            }
            let residual_difference_phase_shift_zero = profiled_nested_zero_bool!(
                "root.div.03g1.nested_zero.residual_difference_phase_shift_zero",
                matches_direct_three_term_phase_shift_zero_subset_root(ctx, residual_difference)
            );
            if residual_difference_phase_shift_zero {
                return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                    options,
                    ctx,
                    expr,
                    one,
                    collect_steps,
                ));
            }

            let residual_difference_isolated_zero = profiled_nested_zero_bool!(
                "root.div.03g2.nested_zero.residual_difference_isolated_zero_fallback",
                {
                    if profile_nested_zero {
                        crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                            "root.div.03g2a.nested_zero.residual_difference_direct_small_zero",
                            residual_sample.clone().unwrap_or_default(),
                        );
                    }
                    let residual_difference_direct_small_zero = profiled_nested_zero_bool!(
                        "root.div.03g2a.nested_zero.residual_difference_direct_small_zero",
                        matches_direct_small_zero_identity_root(ctx, residual_difference)
                    );
                    if residual_difference_direct_small_zero {
                        true
                    } else {
                        if profile_nested_zero {
                            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                "root.div.03g2b.nested_zero.residual_difference_hyperbolic_cosh_cubic_zero",
                                residual_sample.clone().unwrap_or_default(),
                            );
                        }
                        let residual_difference_hyperbolic_cosh_cubic_zero =
                            profiled_nested_zero_bool!(
                                "root.div.03g2b.nested_zero.residual_difference_hyperbolic_cosh_cubic_zero",
                                matches_direct_hyperbolic_cosh_cubic_zero_identity_root(
                                    ctx,
                                    residual_difference,
                                )
                            );
                        if residual_difference_hyperbolic_cosh_cubic_zero {
                            true
                        } else {
                            if profile_nested_zero {
                                crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                    "root.div.03g2c0.nested_zero.residual_difference_hyperbolic_pair",
                                    residual_sample.clone().unwrap_or_default(),
                                );
                            }
                            let residual_difference_hyperbolic_pair = profiled_nested_zero_bool!(
                                "root.div.03g2c0.nested_zero.residual_difference_hyperbolic_pair",
                                matches_direct_nested_zero_hyperbolic_residual_pair_root(
                                    ctx,
                                    residual_difference,
                                )
                            );
                            if residual_difference_hyperbolic_pair {
                                true
                            } else {
                                if profile_nested_zero {
                                    crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                        "root.div.03g2c1.nested_zero.residual_difference_hyperbolic_angle_difference",
                                        residual_sample.clone().unwrap_or_default(),
                                    );
                                }
                                let residual_difference_hyperbolic_angle_difference =
                                    if profile_nested_zero {
                                        run_profiled_orchestrator_bool_section(
                                            "root.div.03g2c1.nested_zero.residual_difference_hyperbolic_angle_difference",
                                            || {
                                                matches_direct_nested_zero_hyperbolic_angle_difference_residual_pair_root(
                                                    ctx,
                                                    residual_difference,
                                                )
                                            },
                                        )
                                    } else {
                                        matches_direct_nested_zero_hyperbolic_angle_difference_residual_pair_root(
                                            ctx,
                                            residual_difference,
                                        )
                                    };
                                if !residual_difference_hyperbolic_angle_difference
                                    && profile_nested_zero
                                {
                                    crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                        "root.div.03g2c2.nested_zero.residual_difference_hyperbolic_triple_angle",
                                        residual_sample.clone().unwrap_or_default(),
                                    );
                                }
                                let residual_difference_hyperbolic_triple_angle =
                                    if !residual_difference_hyperbolic_angle_difference
                                        && profile_nested_zero
                                    {
                                        run_profiled_orchestrator_bool_section(
                                            "root.div.03g2c2.nested_zero.residual_difference_hyperbolic_triple_angle",
                                            || {
                                                matches_direct_nested_zero_hyperbolic_triple_angle_residual_pair_root(
                                                    ctx,
                                                    residual_difference,
                                                )
                                            },
                                        )
                                    } else if !residual_difference_hyperbolic_angle_difference {
                                        matches_direct_nested_zero_hyperbolic_triple_angle_residual_pair_root(
                                            ctx,
                                            residual_difference,
                                        )
                                    } else {
                                        false
                                    };
                                if !residual_difference_hyperbolic_angle_difference
                                    && !residual_difference_hyperbolic_triple_angle
                                    && profile_nested_zero
                                {
                                    crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                        "root.div.03g2c3.nested_zero.residual_difference_pure_double_angle",
                                        residual_sample.clone().unwrap_or_default(),
                                    );
                                }
                                let residual_difference_pure_double_angle =
                                    if !residual_difference_hyperbolic_angle_difference
                                        && !residual_difference_hyperbolic_triple_angle
                                        && profile_nested_zero
                                    {
                                        run_profiled_orchestrator_bool_section(
                                            "root.div.03g2c3.nested_zero.residual_difference_pure_double_angle",
                                            || {
                                                matches_direct_nested_zero_pure_double_angle_residual_pair_root(
                                                    ctx,
                                                    residual_difference,
                                                )
                                            },
                                        )
                                    } else if !residual_difference_hyperbolic_angle_difference
                                        && !residual_difference_hyperbolic_triple_angle
                                    {
                                        matches_direct_nested_zero_pure_double_angle_residual_pair_root(
                                            ctx,
                                            residual_difference,
                                        )
                                    } else {
                                        false
                                    };
                                let residual_difference_trig_ratio =
                                    if !residual_difference_hyperbolic_angle_difference
                                        && !residual_difference_hyperbolic_triple_angle
                                        && !residual_difference_pure_double_angle
                                        && profile_nested_zero
                                    {
                                        crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                            "root.div.03g2c5a.nested_zero.residual_difference_trig_ratio",
                                            residual_sample.clone().unwrap_or_default(),
                                        );
                                        run_profiled_orchestrator_bool_section(
                                            "root.div.03g2c5a.nested_zero.residual_difference_trig_ratio",
                                            || {
                                                matches_direct_nested_zero_trig_ratio_or_reciprocal_residual_pair_root(
                                                    ctx,
                                                    residual_difference,
                                                    true,
                                                )
                                            },
                                        )
                                    } else if !residual_difference_hyperbolic_angle_difference
                                        && !residual_difference_hyperbolic_triple_angle
                                        && !residual_difference_pure_double_angle
                                    {
                                        matches_direct_nested_zero_trig_ratio_or_reciprocal_residual_pair_root(
                                            ctx,
                                            residual_difference,
                                            true,
                                        )
                                    } else {
                                        false
                                    };
                                let residual_difference_trig_reciprocal =
                                    if !residual_difference_hyperbolic_angle_difference
                                        && !residual_difference_hyperbolic_triple_angle
                                        && !residual_difference_pure_double_angle
                                        && !residual_difference_trig_ratio
                                        && profile_nested_zero
                                    {
                                        crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                            "root.div.03g2c5b.nested_zero.residual_difference_trig_reciprocal",
                                            residual_sample.clone().unwrap_or_default(),
                                        );
                                        run_profiled_orchestrator_bool_section(
                                            "root.div.03g2c5b.nested_zero.residual_difference_trig_reciprocal",
                                            || {
                                                matches_direct_nested_zero_trig_ratio_or_reciprocal_residual_pair_root(
                                                    ctx,
                                                    residual_difference,
                                                    false,
                                                )
                                            },
                                        )
                                    } else if !residual_difference_hyperbolic_angle_difference
                                        && !residual_difference_hyperbolic_triple_angle
                                        && !residual_difference_pure_double_angle
                                        && !residual_difference_trig_ratio
                                    {
                                        matches_direct_nested_zero_trig_ratio_or_reciprocal_residual_pair_root(
                                            ctx,
                                            residual_difference,
                                            false,
                                        )
                                    } else {
                                        false
                                    };
                                let residual_difference_trig_ratio_alias =
                                    if !residual_difference_hyperbolic_angle_difference
                                        && !residual_difference_hyperbolic_triple_angle
                                        && !residual_difference_pure_double_angle
                                        && !residual_difference_trig_ratio
                                        && !residual_difference_trig_reciprocal
                                        && profile_nested_zero
                                    {
                                        crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                            "root.div.03g2c6a.nested_zero.residual_difference_trig_ratio_alias",
                                            residual_sample.clone().unwrap_or_default(),
                                        );
                                        run_profiled_orchestrator_bool_section(
                                            "root.div.03g2c6a.nested_zero.residual_difference_trig_ratio_alias",
                                            || {
                                                matches_direct_nested_zero_trig_ratio_alias_residual_pair_root(
                                                    ctx,
                                                    residual_difference,
                                                )
                                            },
                                        )
                                    } else if !residual_difference_hyperbolic_angle_difference
                                        && !residual_difference_hyperbolic_triple_angle
                                        && !residual_difference_pure_double_angle
                                        && !residual_difference_trig_ratio
                                        && !residual_difference_trig_reciprocal
                                    {
                                        matches_direct_nested_zero_trig_ratio_alias_residual_pair_root(
                                            ctx,
                                            residual_difference,
                                        )
                                    } else {
                                        false
                                    };
                                let residual_difference_trig_double_angle_cos_variant = false;
                                if residual_difference_hyperbolic_angle_difference
                                    || residual_difference_hyperbolic_triple_angle
                                    || residual_difference_pure_double_angle
                                    || residual_difference_trig_ratio
                                    || residual_difference_trig_reciprocal
                                    || residual_difference_trig_ratio_alias
                                    || residual_difference_trig_double_angle_cos_variant
                                {
                                    true
                                } else {
                                    let residual_difference_term_count =
                                        AddView::from_expr(ctx, residual_difference).terms.len();
                                    let skip_three_term_subset_rule_for_two_term_reciprocal_trig =
                                        residual_difference_term_count == 2
                                            && expr_contains_reciprocal_trig_builtin_local(
                                                ctx,
                                                residual_difference,
                                            );
                                    if profile_nested_zero {
                                        crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                            "root.div.03g2c4.nested_zero.residual_difference_three_term_subset_rule",
                                            residual_sample.clone().unwrap_or_default(),
                                        );
                                    }
                                    let residual_difference_three_term_subset_rule =
                                        if skip_three_term_subset_rule_for_two_term_reciprocal_trig
                                        {
                                            false
                                        } else {
                                            profiled_nested_zero_bool!(
                                            "root.div.03g2c4.nested_zero.residual_difference_three_term_subset_rule",
                                            child_matches_exact_zero_three_term_subset_rule(
                                                options,
                                                ctx,
                                                residual_difference,
                                            )
                                        )
                                        };
                                    if residual_difference_three_term_subset_rule {
                                        true
                                    } else {
                                        let residual_difference_trig_ratio = false;
                                        let residual_difference_trig_reciprocal = false;
                                        if residual_difference_trig_ratio
                                            || residual_difference_trig_reciprocal
                                        {
                                            true
                                        } else {
                                            let residual_difference_trig_ratio_alias = false;
                                            if !residual_difference_trig_ratio_alias
                                                && profile_nested_zero
                                            {
                                                crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                            "root.div.03g2c6b.nested_zero.residual_difference_signed_pure_double_angle",
                                            residual_sample.clone().unwrap_or_default(),
                                        );
                                            }
                                            let residual_difference_signed_pure_double_angle =
                                                if !residual_difference_trig_ratio_alias
                                                    && profile_nested_zero
                                                {
                                                    run_profiled_orchestrator_bool_section(
                                                "root.div.03g2c6b.nested_zero.residual_difference_signed_pure_double_angle",
                                                || {
                                                    matches_direct_nested_zero_signed_pure_double_angle_residual_pair_root(
                                                        ctx,
                                                        residual_difference,
                                                    )
                                                },
                                            )
                                                } else if !residual_difference_trig_ratio_alias {
                                                    matches_direct_nested_zero_signed_pure_double_angle_residual_pair_root(
                                                ctx,
                                                residual_difference,
                                            )
                                                } else {
                                                    false
                                                };
                                            if !residual_difference_trig_ratio_alias
                                                && !residual_difference_signed_pure_double_angle
                                                && profile_nested_zero
                                            {
                                                crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                            "root.div.03g2c6c.nested_zero.residual_difference_half_angle_tan",
                                            residual_sample.clone().unwrap_or_default(),
                                        );
                                            }
                                            let residual_difference_half_angle_tan =
                                                if !residual_difference_trig_ratio_alias
                                                    && !residual_difference_signed_pure_double_angle
                                                    && profile_nested_zero
                                                {
                                                    run_profiled_orchestrator_bool_section(
                                                "root.div.03g2c6c.nested_zero.residual_difference_half_angle_tan",
                                                || {
                                                    matches_direct_nested_zero_half_angle_tan_residual_pair_root(
                                                        ctx,
                                                        residual_difference,
                                                    )
                                                },
                                            )
                                                } else if !residual_difference_trig_ratio_alias
                                                    && !residual_difference_signed_pure_double_angle
                                                {
                                                    matches_direct_nested_zero_half_angle_tan_residual_pair_root(
                                                ctx,
                                                residual_difference,
                                            )
                                                } else {
                                                    false
                                                };
                                            if residual_difference_trig_ratio_alias
                                                || residual_difference_signed_pure_double_angle
                                                || residual_difference_half_angle_tan
                                            {
                                                true
                                            } else {
                                                if profile_nested_zero {
                                                    crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                "root.div.03g2c7a.nested_zero.residual_difference_exact_zero_same_denominator_pair",
                                                residual_sample.clone().unwrap_or_default(),
                                            );
                                                }
                                                let residual_difference_exact_zero_same_denominator_pair =
                                                    if profile_nested_zero {
                                                        run_profiled_orchestrator_bool_section(
                                                    "root.div.03g2c7a.nested_zero.residual_difference_exact_zero_same_denominator_pair",
                                                    || {
                                                        child_matches_exact_zero_same_denominator_direct_or_passthrough_pair(
                                                            ctx,
                                                            residual_difference,
                                                        )
                                                    },
                                                )
                                                    } else {
                                                        false
                                                    };
                                                if !residual_difference_exact_zero_same_denominator_pair
                                                && profile_nested_zero
                                            {
                                                crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                "root.div.03g2c7b.nested_zero.residual_difference_exact_zero_two_factor_or_quotient_pair",
                                                residual_sample.clone().unwrap_or_default(),
                                            );
                                            }
                                                let residual_difference_exact_zero_two_factor_or_quotient_pair =
                                            if !residual_difference_exact_zero_same_denominator_pair
                                                && profile_nested_zero
                                            {
                                                run_profiled_orchestrator_bool_section(
                                                    "root.div.03g2c7b.nested_zero.residual_difference_exact_zero_two_factor_or_quotient_pair",
                                                    || {
                                                        child_matches_exact_zero_two_factor_or_quotient_pair(
                                                            ctx,
                                                            residual_difference,
                                                        )
                                                    },
                                                )
                                            } else {
                                                false
                                            };
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && profile_nested_zero
                                        {
                                            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                "root.div.03g2c7c1.nested_zero.residual_difference_exact_zero_common_scale_known_residual",
                                                residual_sample.clone().unwrap_or_default(),
                                            );
                                        }
                                                let residual_difference_exact_zero_common_scale_known_residual =
                                            if !residual_difference_exact_zero_same_denominator_pair
                                                && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                                && profile_nested_zero
                                            {
                                                run_profiled_orchestrator_bool_section(
                                                    "root.div.03g2c7c1.nested_zero.residual_difference_exact_zero_common_scale_known_residual",
                                                    || {
                                                        child_matches_exact_zero_common_scale_known_residual(
                                                            ctx,
                                                            residual_difference,
                                                        )
                                                    },
                                                )
                                            } else {
                                                false
                                            };
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && !residual_difference_exact_zero_common_scale_known_residual
                                            && profile_nested_zero
                                        {
                                            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                "root.div.03g2c7c2.nested_zero.residual_difference_exact_zero_common_scale_rule",
                                                residual_sample.clone().unwrap_or_default(),
                                            );
                                        }
                                                let residual_difference_exact_zero_common_scale_rule =
                                            if !residual_difference_exact_zero_same_denominator_pair
                                                && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                                && !residual_difference_exact_zero_common_scale_known_residual
                                                && profile_nested_zero
                                            {
                                                run_profiled_orchestrator_bool_section(
                                                    "root.div.03g2c7c2.nested_zero.residual_difference_exact_zero_common_scale_rule",
                                                    || {
                                                        child_matches_exact_zero_common_scale_rule(
                                                            options,
                                                            ctx,
                                                            residual_difference,
                                                        )
                                                    },
                                                )
                                            } else {
                                                false
                                            };
                                                let residual_difference_exact_zero_common_scale_rule_family =
                                            if !residual_difference_exact_zero_same_denominator_pair
                                                && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                                && !residual_difference_exact_zero_common_scale_known_residual
                                                && profile_nested_zero
                                            {
                                                crate::rules::arithmetic::classify_exact_zero_common_scale_route_profile_family(
                                                    ctx,
                                                    residual_difference,
                                                )
                                            } else {
                                                None
                                            };
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && !residual_difference_exact_zero_common_scale_known_residual
                                            && profile_nested_zero
                                        {
                                            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                "root.div.03g2c7c2a.nested_zero.residual_difference_exact_zero_common_scale_same_denominator",
                                                residual_sample.clone().unwrap_or_default(),
                                            );
                                        }
                                                let residual_difference_exact_zero_common_scale_same_denominator =
                                            if !residual_difference_exact_zero_same_denominator_pair
                                                && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                                && !residual_difference_exact_zero_common_scale_known_residual
                                                && profile_nested_zero
                                            {
                                                run_profiled_orchestrator_bool_section(
                                                    "root.div.03g2c7c2a.nested_zero.residual_difference_exact_zero_common_scale_same_denominator",
                                                    || {
                                                        residual_difference_exact_zero_common_scale_rule_family
                                                            == Some(
                                                                crate::rules::arithmetic::ExactZeroCommonScaleRouteProfileFamily::SameDenominator,
                                                            )
                                                    },
                                                )
                                            } else {
                                                false
                                            };
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && !residual_difference_exact_zero_common_scale_known_residual
                                            && !residual_difference_exact_zero_common_scale_same_denominator
                                            && profile_nested_zero
                                        {
                                            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                "root.div.03g2c7c2b.nested_zero.residual_difference_exact_zero_common_scale_residual_direct",
                                                residual_sample.clone().unwrap_or_default(),
                                            );
                                        }
                                                let residual_difference_exact_zero_common_scale_residual_direct =
                                            if !residual_difference_exact_zero_same_denominator_pair
                                                && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                                && !residual_difference_exact_zero_common_scale_known_residual
                                                && !residual_difference_exact_zero_common_scale_same_denominator
                                                && profile_nested_zero
                                            {
                                                run_profiled_orchestrator_bool_section(
                                                    "root.div.03g2c7c2b.nested_zero.residual_difference_exact_zero_common_scale_residual_direct",
                                                    || {
                                                        residual_difference_exact_zero_common_scale_rule_family
                                                            == Some(
                                                                crate::rules::arithmetic::ExactZeroCommonScaleRouteProfileFamily::ResidualDirect,
                                                            )
                                                    },
                                                )
                                            } else {
                                                false
                                            };
                                                let residual_difference_exact_zero_common_scale_residual_direct_label =
                                            if !residual_difference_exact_zero_same_denominator_pair
                                                && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                                && !residual_difference_exact_zero_common_scale_known_residual
                                                && !residual_difference_exact_zero_common_scale_same_denominator
                                                && profile_nested_zero
                                            {
                                                crate::rules::arithmetic::classify_exact_zero_common_scale_residual_direct_profile_label(
                                                    ctx,
                                                    residual_difference,
                                                )
                                            } else {
                                                None
                                            };
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && !residual_difference_exact_zero_common_scale_known_residual
                                            && !residual_difference_exact_zero_common_scale_same_denominator
                                            && profile_nested_zero
                                        {
                                            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                "root.div.03g2c7c2b1.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_default_simplify_trig_ratio",
                                                residual_sample.clone().unwrap_or_default(),
                                            );
                                        }
                                                let residual_difference_exact_zero_common_scale_residual_direct_default_simplify_trig_ratio =
                                            if !residual_difference_exact_zero_same_denominator_pair
                                                && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                                && !residual_difference_exact_zero_common_scale_known_residual
                                                && !residual_difference_exact_zero_common_scale_same_denominator
                                                && profile_nested_zero
                                            {
                                                run_profiled_orchestrator_bool_section(
                                                    "root.div.03g2c7c2b1.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_default_simplify_trig_ratio",
                                                    || {
                                                        residual_difference_exact_zero_common_scale_residual_direct_label
                                                            == Some(
                                                                "rule.direct_core_equivalence.default_simplify.family.trig_ratio",
                                                            )
                                                    },
                                                )
                                            } else {
                                                false
                                            };
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && !residual_difference_exact_zero_common_scale_known_residual
                                            && !residual_difference_exact_zero_common_scale_same_denominator
                                            && !residual_difference_exact_zero_common_scale_residual_direct_default_simplify_trig_ratio
                                            && profile_nested_zero
                                        {
                                            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                "root.div.03g2c7c2b2.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_signed_double_angle",
                                                residual_sample.clone().unwrap_or_default(),
                                            );
                                        }
                                                let residual_difference_exact_zero_common_scale_residual_direct_signed_double_angle =
                                            if !residual_difference_exact_zero_same_denominator_pair
                                                && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                                && !residual_difference_exact_zero_common_scale_known_residual
                                                && !residual_difference_exact_zero_common_scale_same_denominator
                                                && !residual_difference_exact_zero_common_scale_residual_direct_default_simplify_trig_ratio
                                                && profile_nested_zero
                                            {
                                                run_profiled_orchestrator_bool_section(
                                                    "root.div.03g2c7c2b2.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_signed_double_angle",
                                                    || {
                                                        residual_difference_exact_zero_common_scale_residual_direct_label
                                                            == Some(
                                                                "rule.direct_core_equivalence.default_simplify.family.other.non_hyperbolic.signed_double_angle",
                                                            )
                                                            || residual_difference_exact_zero_common_scale_residual_direct_label
                                                                == Some(
                                                                    "rule.direct_core_equivalence.family.double_angle_contraction",
                                                                )
                                                    },
                                                )
                                            } else {
                                                false
                                            };
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && !residual_difference_exact_zero_common_scale_known_residual
                                            && !residual_difference_exact_zero_common_scale_same_denominator
                                            && !residual_difference_exact_zero_common_scale_residual_direct_default_simplify_trig_ratio
                                            && !residual_difference_exact_zero_common_scale_residual_direct_signed_double_angle
                                            && profile_nested_zero
                                        {
                                            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                "root.div.03g2c7c2b3.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other",
                                                residual_sample.clone().unwrap_or_default(),
                                            );
                                        }
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && !residual_difference_exact_zero_common_scale_known_residual
                                            && !residual_difference_exact_zero_common_scale_same_denominator
                                            && !residual_difference_exact_zero_common_scale_residual_direct_default_simplify_trig_ratio
                                            && !residual_difference_exact_zero_common_scale_residual_direct_signed_double_angle
                                            && profile_nested_zero
                                        {
                                            let residual_difference_exact_zero_common_scale_residual_direct_phase_shift_identity_label =
                                                crate::rules::arithmetic::classify_exact_zero_common_scale_residual_direct_phase_shift_identity_profile_label(
                                                    ctx,
                                                    residual_difference,
                                                );
                                            for label in [
                                                "root.div.03g2c7c2b3a1.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.forward_linear_to_shifted",
                                                "root.div.03g2c7c2b3a2.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.forward_shifted_to_linear",
                                                "root.div.03g2c7c2b3a3.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.forward_shifted_to_shifted",
                                                "root.div.03g2c7c2b3a4.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.reverse_linear_to_shifted",
                                                "root.div.03g2c7c2b3a5.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.reverse_shifted_to_linear",
                                                "root.div.03g2c7c2b3a6.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.reverse_shifted_to_shifted",
                                            ] {
                                                crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                    label,
                                                    residual_sample.clone().unwrap_or_default(),
                                                );
                                            }
                                            for label in [
                                                "root.div.03g2c7c2b3a1.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.forward_linear_to_shifted",
                                                "root.div.03g2c7c2b3a2.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.forward_shifted_to_linear",
                                                "root.div.03g2c7c2b3a3.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.forward_shifted_to_shifted",
                                                "root.div.03g2c7c2b3a4.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.reverse_linear_to_shifted",
                                                "root.div.03g2c7c2b3a5.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.reverse_shifted_to_linear",
                                                "root.div.03g2c7c2b3a6.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity.reverse_shifted_to_shifted",
                                            ] {
                                                let matches_label =
                                                    residual_difference_exact_zero_common_scale_residual_direct_phase_shift_identity_label
                                                        == Some(label);
                                                run_profiled_orchestrator_bool_section(label, || {
                                                    matches_label
                                                });
                                            }
                                            let residual_difference_exact_zero_common_scale_residual_direct_other_label =
                                                crate::rules::arithmetic::classify_exact_zero_common_scale_residual_direct_other_profile_label(
                                                    ctx,
                                                    residual_difference,
                                                );
                                            for label in [
                                                "root.div.03g2c7c2b3a.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity",
                                                "root.div.03g2c7c2b3b.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.default_simplify_non_hyperbolic_other",
                                                "root.div.03g2c7c2b3c.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.default_simplify_other",
                                                "root.div.03g2c7c2b3d.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.direct_core_other",
                                                "root.div.03g2c7c2b3e.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.remaining_family",
                                            ] {
                                                crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                    label,
                                                    residual_sample.clone().unwrap_or_default(),
                                                );
                                            }
                                            for label in [
                                                "root.div.03g2c7c2b3a.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.phase_shift_identity",
                                                "root.div.03g2c7c2b3b.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.default_simplify_non_hyperbolic_other",
                                                "root.div.03g2c7c2b3c.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.default_simplify_other",
                                                "root.div.03g2c7c2b3d.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.direct_core_other",
                                                "root.div.03g2c7c2b3e.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other.remaining_family",
                                            ] {
                                                let matches_label =
                                                    residual_difference_exact_zero_common_scale_residual_direct_other_label
                                                        == Some(label);
                                                run_profiled_orchestrator_bool_section(label, || {
                                                    matches_label
                                                });
                                            }
                                            run_profiled_orchestrator_bool_section(
                                                "root.div.03g2c7c2b3.nested_zero.residual_difference_exact_zero_common_scale_residual_direct_other",
                                                || {
                                                    residual_difference_exact_zero_common_scale_residual_direct_label
                                                        .is_some()
                                                },
                                            );
                                        }
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && !residual_difference_exact_zero_common_scale_known_residual
                                            && !residual_difference_exact_zero_common_scale_same_denominator
                                            && !residual_difference_exact_zero_common_scale_residual_direct
                                            && profile_nested_zero
                                        {
                                            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                "root.div.03g2c7c2c.nested_zero.residual_difference_exact_zero_common_scale_tail_fast_trig_raw",
                                                residual_sample.clone().unwrap_or_default(),
                                            );
                                        }
                                                let residual_difference_exact_zero_common_scale_tail_fast_trig_raw =
                                            if !residual_difference_exact_zero_same_denominator_pair
                                                && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                                && !residual_difference_exact_zero_common_scale_known_residual
                                                && !residual_difference_exact_zero_common_scale_same_denominator
                                                && !residual_difference_exact_zero_common_scale_residual_direct
                                                && profile_nested_zero
                                            {
                                                run_profiled_orchestrator_bool_section(
                                                    "root.div.03g2c7c2c.nested_zero.residual_difference_exact_zero_common_scale_tail_fast_trig_raw",
                                                    || {
                                                        residual_difference_exact_zero_common_scale_rule_family
                                                            == Some(
                                                                crate::rules::arithmetic::ExactZeroCommonScaleRouteProfileFamily::TailFastTrigRaw,
                                                            )
                                                    },
                                                )
                                            } else {
                                                false
                                            };
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && !residual_difference_exact_zero_common_scale_known_residual
                                            && !residual_difference_exact_zero_common_scale_same_denominator
                                            && !residual_difference_exact_zero_common_scale_residual_direct
                                            && !residual_difference_exact_zero_common_scale_tail_fast_trig_raw
                                            && profile_nested_zero
                                        {
                                            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                "root.div.03g2c7c2d.nested_zero.residual_difference_exact_zero_common_scale_tail_fast_trig_normalized",
                                                residual_sample.clone().unwrap_or_default(),
                                            );
                                        }
                                                let residual_difference_exact_zero_common_scale_tail_fast_trig_normalized =
                                            if !residual_difference_exact_zero_same_denominator_pair
                                                && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                                && !residual_difference_exact_zero_common_scale_known_residual
                                                && !residual_difference_exact_zero_common_scale_same_denominator
                                                && !residual_difference_exact_zero_common_scale_residual_direct
                                                && !residual_difference_exact_zero_common_scale_tail_fast_trig_raw
                                                && profile_nested_zero
                                            {
                                                run_profiled_orchestrator_bool_section(
                                                    "root.div.03g2c7c2d.nested_zero.residual_difference_exact_zero_common_scale_tail_fast_trig_normalized",
                                                    || {
                                                        residual_difference_exact_zero_common_scale_rule_family
                                                            == Some(
                                                                crate::rules::arithmetic::ExactZeroCommonScaleRouteProfileFamily::TailFastTrigNormalized,
                                                            )
                                                    },
                                                )
                                            } else {
                                                false
                                            };
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && !residual_difference_exact_zero_common_scale_known_residual
                                            && !residual_difference_exact_zero_common_scale_same_denominator
                                            && !residual_difference_exact_zero_common_scale_residual_direct
                                            && !residual_difference_exact_zero_common_scale_tail_fast_trig_raw
                                            && !residual_difference_exact_zero_common_scale_tail_fast_trig_normalized
                                            && profile_nested_zero
                                        {
                                            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                "root.div.03g2c7c2e.nested_zero.residual_difference_exact_zero_common_scale_tail_two_term_core_equivalence",
                                                residual_sample.clone().unwrap_or_default(),
                                            );
                                        }
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && !residual_difference_exact_zero_common_scale_known_residual
                                            && !residual_difference_exact_zero_common_scale_same_denominator
                                            && !residual_difference_exact_zero_common_scale_residual_direct
                                            && !residual_difference_exact_zero_common_scale_tail_fast_trig_raw
                                            && !residual_difference_exact_zero_common_scale_tail_fast_trig_normalized
                                            && profile_nested_zero
                                        {
                                            run_profiled_orchestrator_bool_section(
                                                "root.div.03g2c7c2e.nested_zero.residual_difference_exact_zero_common_scale_tail_two_term_core_equivalence",
                                                || {
                                                    residual_difference_exact_zero_common_scale_rule_family
                                                        == Some(
                                                            crate::rules::arithmetic::ExactZeroCommonScaleRouteProfileFamily::TailTwoTermCoreEquivalence,
                                                        )
                                                },
                                            );
                                        }
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && !residual_difference_exact_zero_common_scale_known_residual
                                            && !residual_difference_exact_zero_common_scale_same_denominator
                                            && !residual_difference_exact_zero_common_scale_residual_direct
                                            && !residual_difference_exact_zero_common_scale_tail_fast_trig_raw
                                            && !residual_difference_exact_zero_common_scale_tail_fast_trig_normalized
                                            && profile_nested_zero
                                        {
                                            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                "root.div.03g2c7c2f.nested_zero.residual_difference_exact_zero_common_scale_other",
                                                residual_sample.clone().unwrap_or_default(),
                                            );
                                        }
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && !residual_difference_exact_zero_common_scale_known_residual
                                            && !residual_difference_exact_zero_common_scale_same_denominator
                                            && !residual_difference_exact_zero_common_scale_residual_direct
                                            && !residual_difference_exact_zero_common_scale_tail_fast_trig_raw
                                            && !residual_difference_exact_zero_common_scale_tail_fast_trig_normalized
                                            && profile_nested_zero
                                        {
                                            run_profiled_orchestrator_bool_section(
                                                "root.div.03g2c7c2f.nested_zero.residual_difference_exact_zero_common_scale_other",
                                                || {
                                                    residual_difference_exact_zero_common_scale_rule_family
                                                        == Some(
                                                            crate::rules::arithmetic::ExactZeroCommonScaleRouteProfileFamily::Other,
                                                        )
                                                },
                                            );
                                        }
                                                let residual_difference_exact_zero_common_scale_shortcut =
                                            residual_difference_exact_zero_common_scale_known_residual
                                                || residual_difference_exact_zero_common_scale_rule;
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && !residual_difference_exact_zero_common_scale_shortcut
                                            && profile_nested_zero
                                        {
                                            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                "root.div.03g2c7d.nested_zero.residual_difference_exact_zero_shortcut_other",
                                                residual_sample.clone().unwrap_or_default(),
                                            );
                                        }
                                                if !residual_difference_exact_zero_same_denominator_pair
                                            && !residual_difference_exact_zero_two_factor_or_quotient_pair
                                            && !residual_difference_exact_zero_common_scale_shortcut
                                            && profile_nested_zero
                                        {
                                            run_profiled_orchestrator_bool_section(
                                                "root.div.03g2c7d.nested_zero.residual_difference_exact_zero_shortcut_other",
                                                || {
                                                    try_standard_exact_zero_equivalence_shortcut(
                                                        options,
                                                        ctx,
                                                        residual_difference,
                                                        false,
                                                    )
                                                    .is_some()
                                                },
                                            );
                                        }
                                                if profile_nested_zero {
                                                    crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                                "root.div.03g2c.nested_zero.residual_difference_isolated_zero",
                                                residual_sample.clone().unwrap_or_default(),
                                            );
                                                }
                                                profiled_nested_zero_bool!(
                                            "root.div.03g2c.nested_zero.residual_difference_isolated_zero",
                                            child_isolated_exact_zero(options, ctx, residual_difference)
                                        )
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            );
            if residual_difference_isolated_zero {
                return Some(run_shifted_quotient_rebuilt_root_shortcut_simplify(
                    options,
                    ctx,
                    expr,
                    one,
                    collect_steps,
                ));
            }
        }
    }

    let new_numerator = numerator_core.and_then(|core| {
        profiled_nested_zero_bool!("root.div.03h.nested_zero.new_numerator_rewrite_to_one", {
            if expr_contains_trig_or_hyperbolic_builtin_local(ctx, core) {
                if let Some(other) = denominator_core {
                    if is_supported_nested_zero_child_partner(ctx, other) {
                        let partner_family =
                            supported_nested_zero_child_partner_profile_family(ctx, other);
                        let label = match partner_family {
                            "log" => {
                                "root.div.03h1.nested_zero.new_numerator_rewrite_to_one.partner_family.log"
                            }
                            "nonlog_additive" => {
                                "root.div.03h2.nested_zero.new_numerator_rewrite_to_one.partner_family.nonlog_additive"
                            }
                            _ => {
                                "root.div.03h3.nested_zero.new_numerator_rewrite_to_one.partner_family.other"
                            }
                        };
                        if profile_nested_zero {
                            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                label,
                                format!(
                                    "{}  ||  {}",
                                    render_expr_for_orchestrator_profile(ctx, core),
                                    render_expr_for_orchestrator_profile(ctx, other)
                                ),
                            );
                            run_profiled_orchestrator_bool_section(label, || {
                                child_isolated_exact_zero(options, ctx, core)
                            })
                        } else {
                            child_isolated_exact_zero(options, ctx, core)
                        }
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        })
        .then_some(one)
    });
    let new_denominator = denominator_core.and_then(|core| {
        profiled_nested_zero_bool!("root.div.03i.nested_zero.new_denominator_rewrite_to_one", {
            expr_contains_trig_or_hyperbolic_builtin_local(ctx, core)
                && numerator_core
                    .is_some_and(|other| is_supported_nested_zero_child_partner(ctx, other))
                && child_isolated_exact_zero(options, ctx, core)
        })
        .then_some(one)
    });

    if new_numerator.is_none() && new_denominator.is_none() {
        return None;
    }

    let rewritten = match (new_numerator, new_denominator) {
        (Some(num), Some(den))
            if compare_expr(ctx, num, one) == Ordering::Equal
                && compare_expr(ctx, den, one) == Ordering::Equal =>
        {
            one
        }
        (Some(num), Some(den)) => ctx.add(Expr::Div(num, den)),
        (Some(num), None) => ctx.add(Expr::Div(num, denominator)),
        (None, Some(den)) => ctx.add(Expr::Div(numerator, den)),
        (None, None) => return None,
    };

    Some(run_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        rewritten,
        collect_steps,
    ))
}

pub(super) fn try_standard_subtract_expanded_sum_diff_cubes_quotient_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    let rule = crate::rules::arithmetic::SubtractExpandedSumDiffCubesQuotientRule;
    let rewrite = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx)?;
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        rewrite,
        "Subtract Expanded Sum/Difference of Cubes Quotient",
        collect_steps,
    ))
}

pub(super) fn try_hidden_solve_root_identical_atom_fraction_shortcut(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;

    if !is_symbolic_atom(ctx, num) || !is_symbolic_atom(ctx, den) || !expr_eq(ctx, num, den) {
        return None;
    }

    Some(ctx.num(1))
}

pub(super) fn try_standard_small_polynomial_denominator_factor_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;

    if !matches!(ctx.get(den), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }
    if cas_ast::count_nodes(ctx, den) > 15 {
        return None;
    }
    if cas_ast::collect_variables(ctx, den).len() != 1 {
        return None;
    }

    let rewrite = try_rewrite_automatic_factor_expr(ctx, den)?;
    let rewritten = ctx.add(Expr::Div(num, rewrite.rewritten));
    Some(run_named_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        rewritten,
        "Factorizar denominador polinómico",
        "Factor Polynomial Denominator",
        collect_steps,
    ))
}
