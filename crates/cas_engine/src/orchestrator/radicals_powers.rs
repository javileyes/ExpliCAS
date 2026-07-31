//! Orquestador: familia `radicals_powers` (troceo P1).
//!
//! Ver la cabecera de `orchestrator.rs` para el contexto.

use super::*;

pub(super) fn is_plain_symbolic_power_after_core(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return false;
    };
    is_symbolic_atom(ctx, *base)
        && matches!(
            ctx.get(*exp),
            Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_)
        )
}

pub(super) fn is_symbolic_power_over_same_atom_noop_root(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Div(left, right) = ctx.get(expr) else {
        return false;
    };
    let Expr::Pow(base, exp) = ctx.get(*left) else {
        return false;
    };

    is_symbolic_atom(ctx, *base)
        && matches!(ctx.get(*exp), Expr::Variable(_) | Expr::Constant(_))
        && is_symbolic_atom(ctx, *right)
        && expr_eq(ctx, *base, *right)
}

pub(super) fn square_of_symbolic_atom(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !is_symbolic_atom(ctx, *base) {
        return None;
    }
    match ctx.get(*exp) {
        Expr::Number(n) if *n == BigRational::from_integer(2.into()) => Some(*base),
        _ => None,
    }
}

pub(super) fn matches_direct_abs_square_pair_root(
    ctx: &Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (abs_square, plain_square) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Expr::Pow(abs_base, abs_exp) = ctx.get(abs_square) else {
            continue;
        };
        if extract_i64_integer(ctx, *abs_exp) != Some(2) {
            continue;
        }
        let Some(abs_inner) = try_unwrap_abs_arg(ctx, *abs_base) else {
            continue;
        };

        let Expr::Pow(plain_base, plain_exp) = ctx.get(plain_square) else {
            continue;
        };
        if extract_i64_integer(ctx, *plain_exp) != Some(2) {
            continue;
        }

        if compare_expr(ctx, abs_inner, *plain_base) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn canonicalize_direct_reciprocal_sqrt_pair_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    try_rewrite_reciprocal_sqrt_canon_expr(ctx, expr).map(|rewrite| rewrite.rewritten)
}

pub(super) fn matches_direct_reciprocal_sqrt_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let Some(lhs_canon) = canonicalize_direct_reciprocal_sqrt_pair_root(ctx, lhs_core) else {
        return false;
    };
    let Some(rhs_canon) = canonicalize_direct_reciprocal_sqrt_pair_root(ctx, rhs_core) else {
        return false;
    };
    compare_expr(ctx, lhs_canon, rhs_canon) == Ordering::Equal
}

pub(super) fn matches_direct_difference_of_squares_quotient_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (quotient_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Expr::Div(numerator, denominator) = ctx.get(quotient_expr) else {
            continue;
        };
        let Some(plan) =
            cas_math::difference_of_squares_support::try_plan_difference_of_squares_division_expr(
                ctx,
                *numerator,
                *denominator,
                cas_math::difference_of_squares_support::DifferenceOfSquaresDivisionPolicy::default(
                ),
            )
        else {
            continue;
        };

        if compare_expr(ctx, plan.final_result, target_expr) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, plan.final_result) <= 24
            && cas_ast::count_nodes(ctx, target_expr) <= 24
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                plan.final_result,
                target_expr,
            )
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_perfect_square_trinomial_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if let Some(plan) =
            cas_math::expansion_rule_support::try_expand_binomial_pow_expr(ctx, source_expr, 2, 2)
        {
            if cas_math::poly_compare::poly_eq(ctx, plan.expanded, target_expr) {
                return true;
            }
        }
    }

    let lhs_minus_rhs = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if matches_direct_perfect_square_trinomial_zero_identity_root(ctx, lhs_minus_rhs) {
        return true;
    }

    let rhs_minus_lhs = ctx.add(Expr::Sub(rhs_core, lhs_core));
    matches_direct_perfect_square_trinomial_zero_identity_root(ctx, rhs_minus_lhs)
}

fn extract_sum_of_two_squared_atoms_root(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 || !terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
        return None;
    }

    let mut first_atom = None;
    let mut second_atom = None;
    for (index, (term_expr, _)) in terms.into_iter().enumerate() {
        let Expr::Pow(base, exp) = ctx.get(term_expr) else {
            return None;
        };
        if !matches!(ctx.get(*exp), Expr::Number(n) if n.is_integer() && n.to_integer() == 2.into())
        {
            return None;
        }
        if index == 0 {
            first_atom = Some(*base);
        } else {
            second_atom = Some(*base);
        }
    }
    Some((first_atom?, second_atom?))
}

fn build_sum_of_squares_product_target_root(
    ctx: &mut Context,
    p: ExprId,
    q: ExprId,
    r: ExprId,
    s: ExprId,
) -> ExprId {
    let first_product = smart_mul(ctx, p, r);
    let second_product = smart_mul(ctx, q, s);
    let first_sum = ctx.add(Expr::Add(first_product, second_product));
    let third_product = smart_mul(ctx, p, s);
    let fourth_product = smart_mul(ctx, q, r);
    let second_diff = ctx.add(Expr::Sub(third_product, fourth_product));
    let two = ctx.num(2);
    let first_square = ctx.add(Expr::Pow(first_sum, two));
    let second_square = ctx.add(Expr::Pow(second_diff, two));
    ctx.add(Expr::Add(first_square, second_square))
}

fn extract_squared_binomial_terms_root(ctx: &Context, expr: ExprId) -> Option<[(ExprId, Sign); 2]> {
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !matches!(ctx.get(*exp), Expr::Number(n) if n.is_integer() && n.to_integer() == 2.into()) {
        return None;
    }

    let base = match ctx.get(*base) {
        Expr::Neg(inner) => *inner,
        _ => *base,
    };
    let terms = AddView::from_expr(ctx, base).terms;
    if terms.len() != 2 {
        return None;
    }
    Some([terms[0], terms[1]])
}

fn matches_squared_sum_pair_root(
    ctx: &mut Context,
    expr: ExprId,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    let Some([(first_term, first_sign), (second_term, second_sign)]) =
        extract_squared_binomial_terms_root(ctx, expr)
    else {
        return false;
    };
    if first_sign != Sign::Pos || second_sign != Sign::Pos {
        return false;
    }
    matches_unordered_expr_pair_root(ctx, first_term, second_term, lhs, rhs)
}

fn matches_squared_difference_pair_root(
    ctx: &mut Context,
    expr: ExprId,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    let Some([(first_term, first_sign), (second_term, second_sign)]) =
        extract_squared_binomial_terms_root(ctx, expr)
    else {
        return false;
    };
    if first_sign == second_sign {
        return false;
    }

    (first_sign == Sign::Pos
        && second_sign == Sign::Neg
        && ((compare_expr(ctx, first_term, lhs) == Ordering::Equal
            && compare_expr(ctx, second_term, rhs) == Ordering::Equal)
            || (compare_expr(ctx, first_term, rhs) == Ordering::Equal
                && compare_expr(ctx, second_term, lhs) == Ordering::Equal)))
        || (first_sign == Sign::Neg
            && second_sign == Sign::Pos
            && ((compare_expr(ctx, second_term, lhs) == Ordering::Equal
                && compare_expr(ctx, first_term, rhs) == Ordering::Equal)
                || (compare_expr(ctx, second_term, rhs) == Ordering::Equal
                    && compare_expr(ctx, first_term, lhs) == Ordering::Equal)))
}

pub(super) fn rewrite_sum_of_squares_product_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Mul(left, right) = ctx.get(expr) else {
        return None;
    };
    let (p, q) = extract_sum_of_two_squared_atoms_root(ctx, *left)?;
    let (r, s) = extract_sum_of_two_squared_atoms_root(ctx, *right)?;
    Some(build_sum_of_squares_product_target_root(ctx, p, q, r, s))
}

pub(super) fn matches_direct_sum_of_squares_product_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (product_expr, square_sum_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Expr::Mul(left, right) = ctx.get(product_expr) else {
            continue;
        };
        let Some((p, q)) = extract_sum_of_two_squared_atoms_root(ctx, *left) else {
            continue;
        };
        let Some((r, s)) = extract_sum_of_two_squared_atoms_root(ctx, *right) else {
            continue;
        };

        let sum_view = AddView::from_expr(ctx, square_sum_expr).terms;
        if sum_view.len() == 2 && sum_view.iter().all(|(_, sign)| *sign == Sign::Pos) {
            let pr = smart_mul(ctx, p, r);
            let qs = smart_mul(ctx, q, s);
            let ps = smart_mul(ctx, p, s);
            let qr = smart_mul(ctx, q, r);

            let square_terms = [sum_view[0].0, sum_view[1].0];
            if (matches_squared_sum_pair_root(ctx, square_terms[0], pr, qs)
                && matches_squared_difference_pair_root(ctx, square_terms[1], ps, qr))
                || (matches_squared_sum_pair_root(ctx, square_terms[1], pr, qs)
                    && matches_squared_difference_pair_root(ctx, square_terms[0], ps, qr))
            {
                return true;
            }
        }

        for (first_a, first_b) in [(p, q), (q, p)] {
            for (second_a, second_b) in [(r, s), (s, r)] {
                let expected = build_sum_of_squares_product_target_root(
                    ctx, first_a, first_b, second_a, second_b,
                );
                if compare_expr(ctx, square_sum_expr, expected) == Ordering::Equal {
                    return true;
                }
                if SemanticEqualityChecker::new(ctx).are_equal(square_sum_expr, expected) {
                    return true;
                }
                if cas_ast::count_nodes(ctx, expected) <= 48
                    && cas_ast::count_nodes(ctx, square_sum_expr) <= 48
                    && isolated_simplify_rewrites_to_target(
                        &crate::phase::SimplifyOptions::default(),
                        ctx,
                        expected,
                        square_sum_expr,
                    )
                {
                    return true;
                }
            }
        }
    }

    false
}

pub(super) fn build_square_preserving_one_root(ctx: &mut Context, expr: ExprId) -> ExprId {
    if extract_i64_integer(ctx, expr) == Some(1) {
        expr
    } else {
        let two = ctx.num(2);
        ctx.add(Expr::Pow(expr, two))
    }
}

pub(super) fn matches_direct_rationalized_sum_of_sqrts_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let one = ctx.num(1);

    for (fraction_side, target_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Expr::Div(source_numerator, source_denominator) = ctx.get(fraction_side).clone() else {
            continue;
        };
        if compare_expr(ctx, source_numerator, one) != Ordering::Equal {
            continue;
        }

        let Expr::Add(left_sqrt, right_sqrt) = ctx.get(source_denominator).clone() else {
            continue;
        };
        let Some(left_base) = extract_square_root_base(ctx, left_sqrt) else {
            continue;
        };
        let Some(right_base) = extract_square_root_base(ctx, right_sqrt) else {
            continue;
        };

        let canonical_numerator = ctx.add(Expr::Sub(left_sqrt, right_sqrt));
        let canonical_denominator = ctx.add(Expr::Sub(left_base, right_base));
        let canonical_target = ctx.add(Expr::Div(canonical_numerator, canonical_denominator));
        if compare_expr(ctx, canonical_target, target_side) == Ordering::Equal {
            return true;
        }

        let swapped_numerator = ctx.add(Expr::Sub(right_sqrt, left_sqrt));
        let swapped_denominator = ctx.add(Expr::Sub(right_base, left_base));
        let swapped_target = ctx.add(Expr::Div(swapped_numerator, swapped_denominator));
        if compare_expr(ctx, swapped_target, target_side) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_sqrt_perfect_square_abs_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    // REAL-ONLY (mirrors SimplifySquareRootRule): the blessed pair is
    // `√(X²) ≡ |X|`, false over ℂ (`√(i²) = i ≠ 1 = |i|`). The matcher
    // signature predates the value-domain axis; the ambient pipeline domain
    // carries it here (audit 2026-07-30, ficha S4-001).
    if crate::rules::arithmetic::ambient_pipeline_value_domain()
        != crate::semantics::ValueDomain::RealOnly
    {
        return false;
    }
    let view = AddView::from_expr(ctx, expr);
    if !(2..=4).contains(&view.terms.len()) {
        return false;
    }

    for candidate_index in 0..view.terms.len() {
        let (sqrt_like_term, sqrt_like_sign) = view.terms[candidate_index];
        let Some(rewrite) = cas_math::perfect_square_support::try_rewrite_sqrt_perfect_square_expr(
            ctx,
            sqrt_like_term,
        ) else {
            continue;
        };

        let normalized_remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, (term, sign))| {
                (index != candidate_index).then_some((
                    term,
                    if sqrt_like_sign == Sign::Pos {
                        match sign {
                            Sign::Pos => Sign::Neg,
                            Sign::Neg => Sign::Pos,
                        }
                    } else {
                        sign
                    },
                ))
            })
            .collect();
        if normalized_remaining_terms.is_empty() {
            continue;
        }

        let remaining_expr = AddView {
            root: expr,
            terms: normalized_remaining_terms,
        }
        .rebuild(ctx);
        if rewrite.rewritten == remaining_expr {
            return true;
        }
        if compare_expr(ctx, rewrite.rewritten, remaining_expr) == Ordering::Equal {
            return true;
        }
        if let Some(abs_rewrite) =
            cas_math::abs_support::try_rewrite_abs_sum_nonnegative_expr(ctx, rewrite.rewritten)
        {
            if abs_rewrite.rewritten == remaining_expr {
                return true;
            }
            if compare_expr(ctx, abs_rewrite.rewritten, remaining_expr) == Ordering::Equal {
                return true;
            }
        }
    }

    false
}

pub(super) fn try_standard_sqrt_perfect_square_abs_subset_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let view = AddView::from_expr(ctx, expr);
    if !(5..=8).contains(&view.terms.len()) {
        return None;
    }
    if !expr_contains_sqrt_or_half_power_local(ctx, expr)
        || expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
        || !expr_contains_log_builtin_local(ctx, expr)
    {
        return None;
    }
    let top_level_sqrt_like_terms = view
        .terms
        .iter()
        .filter(|(term, _)| expr_contains_sqrt_or_half_power_local(ctx, *term))
        .count();
    let top_level_log_like_terms = view
        .terms
        .iter()
        .filter(|(term, _)| expr_contains_log_builtin_local(ctx, *term))
        .count();
    if top_level_sqrt_like_terms < 2 || top_level_log_like_terms < 2 {
        return None;
    }

    for subset_size in [2usize, 3usize] {
        for first_index in 0..view.terms.len() {
            for second_index in (first_index + 1)..view.terms.len() {
                if subset_size == 2 {
                    let subset_terms = [view.terms[first_index], view.terms[second_index]];
                    let subset_expr = build_signed_sum_expr_root(ctx, &subset_terms);
                    if !matches_direct_sqrt_perfect_square_abs_zero_identity_root(ctx, subset_expr)
                    {
                        continue;
                    }

                    let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
                        .terms
                        .iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(index, term)| {
                            (index != first_index && index != second_index).then_some(term)
                        })
                        .collect();
                    if !(2..=6).contains(&remaining_terms.len()) {
                        continue;
                    }

                    let remaining_expr = AddView {
                        root: expr,
                        terms: remaining_terms,
                    }
                    .rebuild(ctx);
                    if expr_contains_sqrt_or_half_power_local(ctx, remaining_expr)
                        || !is_supported_nested_zero_child_partner(ctx, remaining_expr)
                    {
                        continue;
                    }

                    let remaining_rewrites_to_zero =
                        matches_direct_small_zero_identity_root(ctx, remaining_expr)
                            || try_standard_atanh_square_ratio_log_subset_zero_shortcut(
                                options,
                                ctx,
                                remaining_expr,
                                false,
                            )
                            .is_some()
                            || (cas_ast::count_nodes(ctx, remaining_expr) <= 48
                                && isolated_simplify_rewrites_to_zero(
                                    options,
                                    ctx,
                                    remaining_expr,
                                ));
                    if !remaining_rewrites_to_zero {
                        continue;
                    }

                    let zero = ctx.num(0);
                    return Some(finish_standard_root_shortcut(
                        ctx,
                        expr,
                        crate::rule::Rewrite::with_local(
                            zero,
                            "Collapse Exact Zero Additive Subexpression",
                            subset_expr,
                            zero,
                        ),
                        "Collapse Exact Zero Additive Subexpression",
                        collect_steps,
                    ));
                }

                for third_index in (second_index + 1)..view.terms.len() {
                    let subset_terms = [
                        view.terms[first_index],
                        view.terms[second_index],
                        view.terms[third_index],
                    ];
                    let subset_expr = build_signed_sum_expr_root(ctx, &subset_terms);
                    if !matches_direct_sqrt_perfect_square_abs_zero_identity_root(ctx, subset_expr)
                    {
                        continue;
                    }

                    let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = view
                        .terms
                        .iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(index, term)| {
                            (index != first_index && index != second_index && index != third_index)
                                .then_some(term)
                        })
                        .collect();
                    if !(2..=5).contains(&remaining_terms.len()) {
                        continue;
                    }

                    let remaining_expr = AddView {
                        root: expr,
                        terms: remaining_terms,
                    }
                    .rebuild(ctx);
                    if expr_contains_sqrt_or_half_power_local(ctx, remaining_expr)
                        || !is_supported_nested_zero_child_partner(ctx, remaining_expr)
                    {
                        continue;
                    }

                    let remaining_rewrites_to_zero =
                        matches_direct_small_zero_identity_root(ctx, remaining_expr)
                            || try_standard_atanh_square_ratio_log_subset_zero_shortcut(
                                options,
                                ctx,
                                remaining_expr,
                                false,
                            )
                            .is_some()
                            || (cas_ast::count_nodes(ctx, remaining_expr) <= 48
                                && isolated_simplify_rewrites_to_zero(
                                    options,
                                    ctx,
                                    remaining_expr,
                                ));
                    if !remaining_rewrites_to_zero {
                        continue;
                    }

                    let zero = ctx.num(0);
                    return Some(finish_standard_root_shortcut(
                        ctx,
                        expr,
                        crate::rule::Rewrite::with_local(
                            zero,
                            "Collapse Exact Zero Additive Subexpression",
                            subset_expr,
                            zero,
                        ),
                        "Collapse Exact Zero Additive Subexpression",
                        collect_steps,
                    ));
                }
            }
        }
    }

    None
}

fn extract_odd_half_power_outer_factor_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, i64)> {
    if let Some(inner) = extract_unary_builtin_arg_root(ctx, expr, BuiltinFn::Abs) {
        return Some((inner, 1));
    }

    match ctx.get(expr) {
        Expr::Pow(base, exponent) => {
            let power = extract_i64_integer(ctx, *exponent)?;
            if power < 1 {
                return None;
            }
            if let Some(inner) = extract_unary_builtin_arg_root(ctx, *base, BuiltinFn::Abs) {
                Some((inner, power))
            } else {
                Some((*base, power))
            }
        }
        _ => Some((expr, 1)),
    }
}

fn extract_odd_half_power_product_form_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<OddHalfPowerProductFormRoot> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for (sqrt_index, sqrt_factor) in factors.iter().copied().enumerate() {
        let Some(base) = extract_unary_builtin_arg_root(ctx, sqrt_factor, BuiltinFn::Sqrt) else {
            continue;
        };
        let outer_factor = factors[1 - sqrt_index];
        let Some((outer_base, outside_power)) =
            extract_odd_half_power_outer_factor_root(ctx, outer_factor)
        else {
            continue;
        };
        if compare_expr(ctx, outer_base, base) == Ordering::Equal {
            return Some(OddHalfPowerProductFormRoot {
                base,
                outside_power,
            });
        }
    }

    None
}

fn extract_odd_half_power_radical_form_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<OddHalfPowerProductFormRoot> {
    let radicand = extract_unary_builtin_arg_root(ctx, expr, BuiltinFn::Sqrt)?;
    let Expr::Pow(base, exponent) = ctx.get(radicand) else {
        return None;
    };
    let power = extract_i64_integer(ctx, *exponent)?;
    if power < 3 || power % 2 == 0 {
        return None;
    }

    Some(OddHalfPowerProductFormRoot {
        base: *base,
        outside_power: (power - 1) / 2,
    })
}

fn odd_half_power_domain_equivalent_target_match_root(
    ctx: &mut Context,
    rewritten: ExprId,
    target_expr: ExprId,
) -> bool {
    let Some(rewritten_form) = extract_odd_half_power_product_form_root(ctx, rewritten) else {
        return false;
    };
    let Some(target_form) = extract_odd_half_power_product_form_root(ctx, target_expr) else {
        return false;
    };

    rewritten_form.outside_power == target_form.outside_power
        && compare_expr(ctx, rewritten_form.base, target_form.base) == Ordering::Equal
}

pub(super) fn matches_direct_odd_half_power_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let parent_ctx = crate::ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let rule = crate::rules::arithmetic::ExpandOddHalfPowerToEnableCancellationRule;

    for candidate_index in 0..view.terms.len() {
        let (focus_expr, focus_sign) = normalize_signed_add_term_root(
            ctx,
            view.terms[candidate_index].0,
            view.terms[candidate_index].1,
        );
        let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, focus_expr, &parent_ctx) else {
            continue;
        };

        let (other_expr, other_sign) = normalize_signed_add_term_root(
            ctx,
            view.terms[1 - candidate_index].0,
            view.terms[1 - candidate_index].1,
        );
        if focus_sign == other_sign {
            continue;
        }

        if compare_expr(ctx, rewrite.new_expr, other_expr) == Ordering::Equal
            || odd_half_power_domain_equivalent_target_match_root(ctx, rewrite.new_expr, other_expr)
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_odd_half_power_zero_scope_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() == 2 {
        let (lhs_expr, lhs_sign) =
            normalize_signed_add_term_root(ctx, view.terms[0].0, view.terms[0].1);
        let (rhs_expr, rhs_sign) =
            normalize_signed_add_term_root(ctx, view.terms[1].0, view.terms[1].1);
        if lhs_sign != rhs_sign {
            let lhs_matches_rhs = extract_odd_half_power_radical_form_root(ctx, lhs_expr)
                .zip(extract_odd_half_power_product_form_root(ctx, rhs_expr))
                .map(|(lhs_form, rhs_form)| {
                    lhs_form.outside_power == rhs_form.outside_power
                        && compare_expr(ctx, lhs_form.base, rhs_form.base) == Ordering::Equal
                })
                .unwrap_or(false);
            let rhs_matches_lhs = extract_odd_half_power_radical_form_root(ctx, rhs_expr)
                .zip(extract_odd_half_power_product_form_root(ctx, lhs_expr))
                .map(|(rhs_form, lhs_form)| {
                    rhs_form.outside_power == lhs_form.outside_power
                        && compare_expr(ctx, rhs_form.base, lhs_form.base) == Ordering::Equal
                })
                .unwrap_or(false);
            if lhs_matches_rhs || rhs_matches_lhs {
                return true;
            }
        }
    }

    if view.terms.len() != 2 || !expr_contains_sqrt_or_half_power_local(ctx, expr) {
        return false;
    }

    let parent_ctx = crate::ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let expand_rule = crate::rules::arithmetic::ExpandOddHalfPowerToEnableCancellationRule;
    let Some(expanded) = crate::rule::Rule::apply(&expand_rule, ctx, expr, &parent_ctx) else {
        return false;
    };
    let cancel_rule = crate::rules::arithmetic::SubSelfToZeroRule;
    let Some(cancelled) =
        crate::rule::Rule::apply(&cancel_rule, ctx, expanded.new_expr, &parent_ctx)
    else {
        return false;
    };
    let zero = ctx.num(0);
    compare_expr(ctx, cancelled.final_expr(), zero) == Ordering::Equal
}

fn is_square_power_root(ctx: &Context, expr: ExprId) -> bool {
    let Expr::Pow(_, exponent) = ctx.get(expr) else {
        return false;
    };
    let Expr::Number(n) = ctx.get(*exponent) else {
        return false;
    };
    *n == BigRational::from_integer(2.into())
}

pub(super) fn extract_square_power_base_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    let Expr::Number(n) = ctx.get(*exponent) else {
        return None;
    };
    (*n == BigRational::from_integer(2.into())).then_some(*base)
}

pub(super) fn extract_difference_of_square_bases_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut positive_base = None;
    let mut negative_base = None;
    for (term, sign) in terms {
        let base = extract_square_power_base_root(ctx, term)?;
        match sign {
            Sign::Pos if positive_base.is_none() => positive_base = Some(base),
            Sign::Neg if negative_base.is_none() => negative_base = Some(base),
            _ => return None,
        }
    }

    Some((positive_base?, negative_base?))
}

pub(super) fn build_direct_perfect_square_from_terms_root(
    ctx: &mut Context,
    terms: &[(ExprId, Sign)],
) -> Option<ExprId> {
    if terms.len() != 3 {
        return None;
    }

    let expr = build_signed_sum_expr_root(ctx, terms);
    cas_math::factor::factor_perfect_square_trinomial(ctx, expr)
}

pub(super) fn matches_direct_perfect_square_trinomial_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 4 {
        return false;
    }

    for candidate_index in 0..view.terms.len() {
        let (square_term, square_sign) = view.terms[candidate_index];
        if !is_square_power_root(ctx, square_term) {
            continue;
        }

        let remaining_terms = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != candidate_index).then_some(term))
            .collect::<Vec<_>>();

        let normalized_remaining_terms = if square_sign == Sign::Neg {
            remaining_terms
        } else {
            remaining_terms
                .into_iter()
                .map(|(term_expr, term_sign)| {
                    let flipped_sign = match term_sign {
                        Sign::Pos => Sign::Neg,
                        Sign::Neg => Sign::Pos,
                    };
                    (term_expr, flipped_sign)
                })
                .collect()
        };

        let remaining_expr = build_signed_sum_expr_root(ctx, &normalized_remaining_terms);
        let Some(factored_square) =
            build_direct_perfect_square_from_terms_root(ctx, &normalized_remaining_terms)
                .or_else(|| cas_math::factor::factor_perfect_square_trinomial(ctx, remaining_expr))
        else {
            continue;
        };

        if compare_expr(ctx, factored_square, square_term) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_squared_exact_one_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let mut positive_expr = None;
    let mut negative_expr = None;
    for (term_expr, term_sign) in view.terms {
        match term_sign {
            Sign::Pos if positive_expr.is_none() => positive_expr = Some(term_expr),
            Sign::Neg if negative_expr.is_none() => negative_expr = Some(term_expr),
            _ => return false,
        }
    }

    let Some(positive_expr) = positive_expr else {
        return false;
    };
    let Some(negative_expr) = negative_expr else {
        return false;
    };

    let positive_base = extract_plain_pow2_base_root(ctx, positive_expr).unwrap_or(positive_expr);
    let negative_base = extract_plain_pow2_base_root(ctx, negative_expr).unwrap_or(negative_expr);
    if compare_expr(ctx, positive_expr, positive_base) == Ordering::Equal
        && compare_expr(ctx, negative_expr, negative_base) == Ordering::Equal
    {
        return false;
    }

    let base_difference = ctx.add(Expr::Sub(positive_base, negative_base));
    matches_direct_small_zero_or_known_pair_base_root(ctx, base_difference)
}

pub(super) fn extract_power_of_base_exponent_root(
    ctx: &mut Context,
    expr: ExprId,
    base: ExprId,
) -> Option<i64> {
    if compare_expr(ctx, expr, base) == Ordering::Equal {
        return Some(1);
    }

    let Expr::Pow(pow_base, exponent) = ctx.get(expr) else {
        return None;
    };
    if compare_expr(ctx, *pow_base, base) != Ordering::Equal {
        return None;
    }

    let exponent = extract_i64_integer(ctx, *exponent)?;
    (exponent >= 1).then_some(exponent)
}

fn format_standard_simplify_square_root_shortcut_desc(
    kind: SimplifySquareRootRewriteKind,
) -> &'static str {
    match kind {
        SimplifySquareRootRewriteKind::PerfectSquare => "Simplify perfect square root",
        SimplifySquareRootRewriteKind::SquareRootFactors => "Simplify square root factors",
        SimplifySquareRootRewriteKind::AdditiveCommonFactor => {
            "Extract common square factor from additive radicand"
        }
        SimplifySquareRootRewriteKind::QuotientOfSquares => {
            "Simplify square root of quotient of squares"
        }
    }
}

pub(super) fn try_standard_simplify_square_root_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    // REAL-ONLY (mirrors SimplifySquareRootRule): the helper emits |·| forms of
    // symbolic squares, a real-domain identity family (`√(i²) = i ≠ |i|`).
    if options.shared.semantics.value_domain != crate::semantics::ValueDomain::RealOnly {
        return None;
    }
    let rewrite = try_rewrite_simplify_square_root_expr(ctx, expr)?;
    // Confluence: when the rewrite lands on an |·| form, the phase pipeline
    // still owes it a canonicalization chain (Abs Positive Factor,
    // Abs Distribute Over Odd Power, ...) that this single-shot shortcut
    // does not replicate — returning here froze `√(4x²)` at `|2x|` while
    // steps mode reached `2|x|` (a steps-mode divergence). Decline and let
    // the full pipeline own every abs-bearing result.
    if expr_contains_any_builtin_local(ctx, rewrite.rewritten, &[BuiltinFn::Abs]) {
        return None;
    }
    let rewrite = crate::rule::Rewrite::new(rewrite.rewritten).desc(
        format_standard_simplify_square_root_shortcut_desc(rewrite.kind),
    );
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        rewrite,
        "Simplify Square Root",
        collect_steps,
    ))
}

pub(super) fn try_standard_sum_of_squares_product_subset_factor_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 3 {
        return None;
    }

    for i in 0..factors.len() {
        for j in (i + 1)..factors.len() {
            let subset = build_mul_expr_from_factors_root(ctx, &[factors[i], factors[j]]);
            let Some(rewritten_subset) = rewrite_sum_of_squares_product_root(ctx, subset) else {
                continue;
            };
            if compare_expr(ctx, subset, rewritten_subset) == Ordering::Equal {
                continue;
            }
            let mut rewritten_factors = Vec::with_capacity(factors.len() - 1);
            rewritten_factors.push(rewritten_subset);
            for (index, factor) in factors.iter().copied().enumerate() {
                if index != i && index != j {
                    rewritten_factors.push(factor);
                }
            }
            let rewritten = build_mul_expr_from_factors_root(ctx, &rewritten_factors);
            return Some(run_named_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                rewritten,
                "Sum of Squares Product Factor",
                "Sum of Squares Product Factor",
                collect_steps,
            ));
        }
    }

    None
}

pub(super) fn try_standard_perfect_square_trinomial_factor_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (index, factor) in factors.iter().copied().enumerate() {
        let Some(squared_factor) = cas_math::factor::factor_perfect_square_trinomial(ctx, factor)
            .or_else(|| {
                let view = AddView::from_expr(ctx, factor);
                build_direct_perfect_square_from_terms_root(ctx, &view.terms)
            })
        else {
            continue;
        };
        if squared_factor == factor {
            continue;
        }

        let remaining_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(other_index, factor_expr)| (other_index != index).then_some(factor_expr))
            .collect();
        let combined_partner = build_mul_expr_from_factors_root(ctx, &remaining_factors);

        if extract_square_power_base_root(ctx, squared_factor).is_some() {
            if let Some(partner_linear_shift) =
                factor_small_linear_shift_product_partner_root(ctx, combined_partner)
            {
                let rewritten = ctx.add(Expr::Mul(squared_factor, partner_linear_shift));
                let rewrite = crate::rule::Rewrite::new(rewritten)
                    .desc("Canonizar cuadrado perfecto con partner lineal pequeño");
                return Some(finish_standard_root_shortcut(
                    ctx,
                    expr,
                    rewrite,
                    "Perfect Square Trinomial Factor",
                    collect_steps,
                ));
            }
        }

        if let Some(partner_canonical) = canonicalize_direct_pair_factor_root(ctx, combined_partner)
        {
            if partner_canonical != combined_partner {
                let rewritten =
                    build_mul_expr_from_factors_root(ctx, &[squared_factor, partner_canonical]);
                return Some(run_named_rebuilt_root_shortcut_simplify(
                    options,
                    ctx,
                    expr,
                    rewritten,
                    "Perfect Square Trinomial Factor",
                    "Perfect Square Trinomial Factor",
                    collect_steps,
                ));
            }
        }

        if remaining_factors.len() == 1 {
            if let Some(partner_simplified) =
                isolated_simplify_expr_if_changed(options, ctx, combined_partner)
            {
                let rewritten =
                    build_mul_expr_from_factors_root(ctx, &[squared_factor, partner_simplified]);
                return Some(run_named_rebuilt_root_shortcut_simplify(
                    options,
                    ctx,
                    expr,
                    rewritten,
                    "Perfect Square Trinomial Factor",
                    "Perfect Square Trinomial Factor",
                    collect_steps,
                ));
            }
        }

        let rewritten = build_mul_expr_from_factors_root(ctx, &[squared_factor, combined_partner]);
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            rewritten,
            "Perfect Square Trinomial Factor",
            "Perfect Square Trinomial Factor",
            collect_steps,
        ));
    }

    None
}

pub(super) fn extract_square_plus_one_base_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut base = None;
    let mut saw_one = false;
    let two = ctx.num(2);
    for (term_expr, sign) in terms {
        if sign != Sign::Pos {
            return None;
        }
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if saw_one {
                return None;
            }
            saw_one = true;
            continue;
        }
        let Expr::Pow(candidate_base, exponent) = ctx.get(term_expr).clone() else {
            return None;
        };
        if compare_expr(ctx, exponent, two) != Ordering::Equal
            || base.replace(candidate_base).is_some()
        {
            return None;
        }
    }

    saw_one.then_some(base?)
}

fn is_potential_square_anchor_source_root(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(_, _) => true,
        Expr::Add(_, _) | Expr::Sub(_, _) => is_function_free_arithmetic_expr_root(ctx, expr),
        Expr::Neg(inner) => is_potential_square_anchor_source_root(ctx, *inner),
        _ => false,
    }
}

pub(super) fn canonicalize_small_surd_like_term_root(ctx: &mut Context, expr: ExprId) -> ExprId {
    let canonical = if let Some(rewrite) = try_rewrite_canonical_root_expr(ctx, expr) {
        rewrite.rewritten
    } else {
        expr
    };
    if let Some(extract) = try_rewrite_extract_perfect_power_from_radicand_expr(ctx, canonical) {
        return strip_multiplicative_one_root(ctx, extract.rewritten);
    }
    strip_multiplicative_one_root(ctx, canonical)
}

fn extract_small_constant_sqrt_function_value_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(fn_id, BuiltinFn::Sqrt) {
        return None;
    }
    if !cas_ast::collect_variables(ctx, args[0]).is_empty() {
        return None;
    }
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let rewritten = ctx.add(Expr::Pow(args[0], half));
    Some(strip_multiplicative_one_root(ctx, rewritten))
}

pub(super) fn try_rewrite_small_constant_sqrt_function_wrapper_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Function(_, _) => extract_small_constant_sqrt_function_value_root(ctx, expr),
        Expr::Neg(inner) => {
            let rewritten_inner = extract_small_constant_sqrt_function_value_root(ctx, inner)?;
            let rewritten = ctx.add(Expr::Neg(rewritten_inner));
            Some(simplify_shallow_small_constant_expr_root(ctx, rewritten))
        }
        Expr::Mul(lhs, rhs) => {
            if matches!(ctx.get(lhs), Expr::Number(_)) {
                let rewritten_rhs = extract_small_constant_sqrt_function_value_root(ctx, rhs)?;
                let rewritten = smart_mul(ctx, lhs, rewritten_rhs);
                return Some(simplify_shallow_small_constant_expr_root(ctx, rewritten));
            }
            if matches!(ctx.get(rhs), Expr::Number(_)) {
                let rewritten_lhs = extract_small_constant_sqrt_function_value_root(ctx, lhs)?;
                let rewritten = smart_mul(ctx, rewritten_lhs, rhs);
                return Some(simplify_shallow_small_constant_expr_root(ctx, rewritten));
            }
            None
        }
        Expr::Div(lhs, rhs) => {
            if matches!(ctx.get(rhs), Expr::Number(_)) {
                let rewritten_lhs = extract_small_constant_sqrt_function_value_root(ctx, lhs)?;
                let rewritten = ctx.add(Expr::Div(rewritten_lhs, rhs));
                return Some(simplify_shallow_small_constant_expr_root(ctx, rewritten));
            }
            if matches!(ctx.get(lhs), Expr::Number(_)) {
                let rewritten_rhs = extract_small_constant_sqrt_function_value_root(ctx, rhs)?;
                let rewritten = ctx.add(Expr::Div(lhs, rewritten_rhs));
                return Some(simplify_shallow_small_constant_expr_root(ctx, rewritten));
            }
            None
        }
        _ => None,
    }
}

pub(super) fn is_direct_small_surd_like_term_root(ctx: &Context, expr: ExprId) -> bool {
    if extract_square_root_base(ctx, expr).is_some() {
        return true;
    }
    matches!(
        ctx.get(expr),
        Expr::Pow(_, exp) if matches!(ctx.get(*exp), Expr::Number(n) if *n == BigRational::new(1.into(), 2.into()))
    )
}

pub(super) fn is_stable_small_direct_surd_container_root(ctx: &mut Context, expr: ExprId) -> bool {
    if !cas_ast::collect_variables(ctx, expr).is_empty() {
        return false;
    }

    match ctx.get(expr).clone() {
        Expr::Mul(lhs, rhs) => {
            if matches!(ctx.get(lhs), Expr::Number(_))
                && is_direct_small_surd_like_term_root(ctx, rhs)
            {
                let canonical = canonicalize_small_surd_like_term_root(ctx, rhs);
                return compare_expr(ctx, canonical, rhs) == Ordering::Equal;
            }
            if matches!(ctx.get(rhs), Expr::Number(_))
                && is_direct_small_surd_like_term_root(ctx, lhs)
            {
                let canonical = canonicalize_small_surd_like_term_root(ctx, lhs);
                return compare_expr(ctx, canonical, lhs) == Ordering::Equal;
            }
            false
        }
        Expr::Div(lhs, rhs) => {
            if matches!(ctx.get(rhs), Expr::Number(_))
                && is_direct_small_surd_like_term_root(ctx, lhs)
            {
                let canonical = canonicalize_small_surd_like_term_root(ctx, lhs);
                return compare_expr(ctx, canonical, lhs) == Ordering::Equal;
            }
            false
        }
        _ if is_direct_small_surd_like_term_root(ctx, expr) => {
            let canonical = canonicalize_small_surd_like_term_root(ctx, expr);
            compare_expr(ctx, canonical, expr) == Ordering::Equal
        }
        _ => false,
    }
}

pub(super) fn try_standard_square_anchor_linear_shift_partner_shortcut(
    options: &crate::phase::SimplifyOptions,
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
        .any(|factor| is_potential_square_anchor_source_root(ctx, factor))
    {
        return None;
    }
    if !factors
        .iter()
        .copied()
        .all(|factor| is_function_free_arithmetic_expr_root(ctx, factor))
    {
        return None;
    }

    for anchor_index in 0..factors.len() {
        if !is_potential_square_anchor_source_root(ctx, factors[anchor_index]) {
            continue;
        }
        let remaining_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, factor)| (index != anchor_index).then_some(factor))
            .collect();
        let partner_expr = build_mul_expr_from_factors_root(ctx, &remaining_factors);
        if !is_potential_small_linear_shift_product_partner_source_root(ctx, partner_expr) {
            continue;
        }
        let anchor_canonical = canonicalize_direct_pair_factor_root(ctx, factors[anchor_index])
            .or_else(|| factor_known_small_polynomial_partner_root(ctx, factors[anchor_index]))
            .or_else(|| isolated_simplify_expr_if_changed(options, ctx, factors[anchor_index]))
            .unwrap_or(factors[anchor_index]);
        if extract_square_power_base_root(ctx, anchor_canonical).is_none() {
            continue;
        }
        let Some(partner_canonical) =
            factor_small_linear_shift_product_partner_root(ctx, partner_expr)
        else {
            continue;
        };

        let anchor_changed =
            compare_expr(ctx, anchor_canonical, factors[anchor_index]) != Ordering::Equal;
        let partner_changed = compare_expr(ctx, partner_canonical, partner_expr) != Ordering::Equal;
        if !anchor_changed && !partner_changed {
            continue;
        }

        let rewritten = ctx.add(Expr::Mul(anchor_canonical, partner_canonical));
        let rewrite = crate::rule::Rewrite::new(rewritten)
            .desc("Canonizar producto con ancla cuadrada y partner lineal pequeño");
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Square Anchor Linear Shift Product",
            collect_steps,
        ));
    }

    None
}

pub(super) fn try_standard_extract_perfect_square_root_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if extract_shortcut_declines_for_value_domain(options, ctx, expr) {
        return None;
    }
    let canonical = try_rewrite_canonical_root_expr(ctx, expr)?;
    let extract = try_rewrite_extract_perfect_power_from_radicand_expr(ctx, canonical.rewritten)?;

    let rewrite = crate::rule::Rewrite::new(extract.rewritten)
        .desc("Extract perfect square from under radical");
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        rewrite,
        "Extract Perfect Square from Radicand",
        collect_steps,
    ))
}

pub(super) fn try_hidden_solve_root_binomial_square_shortcut(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;

    let Expr::Pow(base, exp) = ctx.get(den) else {
        return None;
    };
    if !matches!(ctx.get(*exp), Expr::Number(n) if *n == BigRational::from_integer(2.into())) {
        return None;
    }
    let Expr::Add(a, b) = ctx.get(*base) else {
        return None;
    };
    let a = *a;
    let b = *b;

    let exp_two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, exp_two));
    let exp_two_b = ctx.num(2);
    let b_sq = ctx.add(Expr::Pow(b, exp_two_b));

    let terms = AddView::from_expr(ctx, num).terms;
    if terms.len() != 3 {
        return None;
    }

    let mut squares = [None, None];
    let mut squares_len = 0usize;
    let mut middle = None;

    for (term, sign) in terms {
        if sign != Sign::Pos {
            return None;
        }

        if expr_eq(ctx, term, a_sq) || expr_eq(ctx, term, b_sq) {
            if squares_len >= squares.len() {
                return None;
            }
            squares[squares_len] = Some(term);
            squares_len += 1;
        } else if middle.is_none() {
            middle = Some(term);
        } else {
            return None;
        }
    }

    let (Some(left_sq), Some(right_sq), Some(middle_term)) = (squares[0], squares[1], middle)
    else {
        return None;
    };

    if !multiset_matches_exact(ctx, &[left_sq, right_sq], &[a_sq, b_sq])
        || !is_exact_two_ab_product(ctx, middle_term, a, b)
    {
        return None;
    }

    Some(ctx.num(1))
}

pub(super) fn try_hidden_solve_root_perfect_square_minus_shortcut(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;

    let Expr::Sub(a, b) = ctx.get(den) else {
        return None;
    };
    let a = *a;
    let b = *b;

    let terms = AddView::from_expr(ctx, num).terms;
    if terms.len() != 3 {
        return None;
    }

    let exp_two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, exp_two));
    let exp_two_b = ctx.num(2);
    let b_sq = ctx.add(Expr::Pow(b, exp_two_b));

    let mut positives = [None, None];
    let mut positive_count = 0usize;
    let mut negative = None;

    for (term, sign) in terms {
        match sign {
            Sign::Pos => {
                if positive_count >= positives.len() {
                    return None;
                }
                positives[positive_count] = Some(term);
                positive_count += 1;
            }
            Sign::Neg => {
                if negative.is_some() {
                    return None;
                }
                negative = Some(term);
            }
        }
    }

    let (Some(left_pos), Some(right_pos), Some(negative_term)) =
        (positives[0], positives[1], negative)
    else {
        return None;
    };

    if !multiset_matches_exact(ctx, &[left_pos, right_pos], &[a_sq, b_sq])
        || !is_exact_two_ab_product(ctx, negative_term, a, b)
    {
        return None;
    }

    Some(den)
}

pub(super) fn try_hidden_solve_root_difference_of_squares_shortcut(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;

    let Expr::Sub(left, right) = ctx.get(num) else {
        return None;
    };
    let left = *left;
    let right = *right;
    let a = square_of_symbolic_atom(ctx, left)?;
    let b = square_of_symbolic_atom(ctx, right)?;

    if expr_eq(ctx, a, b) {
        return None;
    }

    match ctx.get(den) {
        Expr::Sub(dl, dr) if expr_eq(ctx, *dl, a) && expr_eq(ctx, *dr, b) => {
            Some(ctx.add(Expr::Add(a, b)))
        }
        Expr::Add(dl, dr) if expr_eq(ctx, *dl, a) && expr_eq(ctx, *dr, b) => {
            Some(ctx.add(Expr::Sub(a, b)))
        }
        _ => None,
    }
}

pub(super) fn try_hidden_solve_root_power_quotient_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    _domain_mode: crate::DomainMode,
) -> Option<ExprId> {
    let plan = try_rewrite_cancel_same_base_powers_div_expr(ctx, expr)?;
    Some(plan.rewritten)
}

pub(super) fn is_symbolic_power_over_same_atom_noop_after_core(
    ctx: &Context,
    expr: ExprId,
) -> bool {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    let Expr::Pow(base, exp) = ctx.get(*num) else {
        return false;
    };
    if cas_ast::ordering::compare_expr(ctx, *base, *den) != std::cmp::Ordering::Equal {
        return false;
    }

    matches!(ctx.get(*base), Expr::Variable(_) | Expr::Constant(_))
        && matches!(ctx.get(*exp), Expr::Variable(_) | Expr::Constant(_))
}
