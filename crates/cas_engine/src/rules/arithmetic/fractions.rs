//! `arithmetic`: familia `fractions`.
//!
//! Ver la cabecera de `arithmetic.rs` para el contexto.

use super::*;

pub(super) fn exprs_equal_up_to_same_denominator(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let Expr::Div(lhs_num, lhs_den) = ctx.get(lhs).clone() else {
        return false;
    };
    let Expr::Div(rhs_num, rhs_den) = ctx.get(rhs).clone() else {
        return false;
    };

    compare_expr(ctx, lhs_den, rhs_den) == Ordering::Equal
        && (exprs_match_for_cancellation_leaf(ctx, lhs_num, rhs_num)
            || exprs_equal_up_to_add_term_multiset_for_cancellation(ctx, lhs_num, rhs_num))
}

pub(super) fn exprs_equal_up_to_fraction_parts_for_cancellation(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let Expr::Div(lhs_num, lhs_den) = ctx.get(lhs).clone() else {
        return false;
    };
    let Expr::Div(rhs_num, rhs_den) = ctx.get(rhs).clone() else {
        return false;
    };

    exprs_match_for_cancellation(ctx, lhs_num, rhs_num)
        && exprs_match_for_cancellation(ctx, lhs_den, rhs_den)
}

pub(super) fn maybe_fraction_telescoping_zero_scope_candidate(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 || !expr_contains_division_node(ctx, expr) {
        return false;
    }

    view.terms
        .iter()
        .all(|(term_expr, _)| expr_contains_division_node(ctx, *term_expr))
}

// DivZeroRule: 0/d → 0
// Domain Mode Policy: 0/d → 0 changes the domain of definition if d can be 0.
// Uses unified DomainOracle via oracle_allows_with_hint:
// - Strict: only apply if prove_nonzero(d) == Proven
// - Generic: apply with NonZero(d) assumption (Definability class)
// - Assume: apply with NonZero(d) assumption
pub(crate) fn try_build_exact_zero_radical_numerator_const_division_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
        return None;
    };
    let denominator_value = cas_ast::views::as_rational_const(ctx, denominator, 8)?;
    if denominator_value.is_zero() || !expr_contains_sqrt_or_half_power(ctx, numerator) {
        return None;
    }

    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, numerator)?;
    let child_rewrite =
        try_build_direct_scaled_reciprocal_half_power_product_rewrite(ctx, lhs_core, rhs_core)
            .or_else(|| try_build_direct_core_equivalence_rewrite(ctx, lhs_core, rhs_core))?;

    Some(
        Rewrite::with_local(
            ctx.num(0),
            child_rewrite.description.clone(),
            expr,
            ctx.num(0),
        )
        .requires_all(child_rewrite.required_conditions.clone())
        .assume_all(child_rewrite.assumption_events.clone()),
    )
}

fn two_term_fraction_sum_parts_for_finite_evaluation_match(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let expr = strip_trivial_one_product_factors_for_core_difference(ctx, expr);
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let (first_num, first_den) = as_div(ctx, terms[0].0)?;
    let (second_num, second_den) = as_div(ctx, terms[1].0)?;
    let first_cross = smart_mul(ctx, first_num, second_den);
    let second_cross = smart_mul(ctx, second_num, first_den);
    let numerator_terms = [(first_cross, terms[0].1), (second_cross, terms[1].1)];
    let numerator = build_signed_sum_expr(ctx, &numerator_terms);
    let denominator = smart_mul(ctx, first_den, second_den);
    Some((numerator, denominator))
}

fn finite_evaluation_fraction_parts_for_match(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let mut numerator_factors = Vec::new();
    let mut denominator_factors = Vec::new();

    for factor in flatten_mul_chain(ctx, expr) {
        if is_one_expr(ctx, factor) {
            continue;
        }
        if let Some((num, den)) = as_div(ctx, factor) {
            if !is_one_expr(ctx, num) {
                numerator_factors.push(num);
            }
            denominator_factors.push(den);
        } else {
            numerator_factors.push(factor);
        }
    }

    if denominator_factors.is_empty() {
        return None;
    }

    let denominator = build_mul_expr_from_factors(ctx, &denominator_factors);
    for (index, factor) in numerator_factors.iter().copied().enumerate() {
        let Some((inner_num, inner_den)) =
            two_term_fraction_sum_parts_for_finite_evaluation_match(ctx, factor)
        else {
            continue;
        };

        let adjusted_numerator_factors: Vec<_> = numerator_factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(factor_index, factor)| (factor_index != index).then_some(factor))
            .chain(std::iter::once(inner_num))
            .collect();
        let numerator = build_mul_expr_from_factors(ctx, &adjusted_numerator_factors);
        return Some((numerator, smart_mul(ctx, denominator, inner_den)));
    }

    let numerator = build_mul_expr_from_factors(ctx, &numerator_factors);
    if let Some((inner_num, inner_den)) =
        two_term_fraction_sum_parts_for_finite_evaluation_match(ctx, numerator)
    {
        return Some((inner_num, smart_mul(ctx, denominator, inner_den)));
    }

    Some((numerator, denominator))
}

pub(super) fn finite_evaluation_fraction_parts_match(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let Some((lhs_num, lhs_den)) = finite_evaluation_fraction_parts_for_match(ctx, lhs) else {
        return false;
    };
    let Some((rhs_num, rhs_den)) = finite_evaluation_fraction_parts_for_match(ctx, rhs) else {
        return false;
    };

    let lhs_cross = smart_mul(ctx, lhs_num, rhs_den);
    let rhs_cross = smart_mul(ctx, rhs_num, lhs_den);
    exprs_match_for_cancellation(ctx, lhs_cross, rhs_cross)
        || poly_eq(ctx, lhs_cross, rhs_cross)
        || exprs_match_after_default_simplify(ctx, lhs_cross, rhs_cross)
}

pub(super) fn extract_unit_fraction_denominator(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    (extract_i64_integer(ctx, *num) == Some(1)).then_some(*den)
}

fn extract_reciprocal_sum_difference_nested_fraction_target(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(
    cas_ast::ExprId,
    cas_ast::ExprId,
    cas_ast::ExprId,
    cas_ast::ExprId,
)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    let numerator_view = AddView::from_expr(ctx, numerator);
    let denominator_view = AddView::from_expr(ctx, denominator);
    if numerator_view.terms.len() != 2 || denominator_view.terms.len() != 2 {
        return None;
    }

    let mut numerator_denoms = smallvec::SmallVec::<[cas_ast::ExprId; 2]>::new();
    for (term_expr, term_sign) in numerator_view.terms {
        if term_sign != Sign::Pos {
            return None;
        }
        numerator_denoms.push(extract_unit_fraction_denominator(ctx, term_expr)?);
    }

    let mut positive_den = None;
    let mut negative_den = None;
    for (term_expr, term_sign) in denominator_view.terms {
        let unit_den = extract_unit_fraction_denominator(ctx, term_expr)?;
        match term_sign {
            Sign::Pos if positive_den.is_none() => positive_den = Some(unit_den),
            Sign::Neg if negative_den.is_none() => negative_den = Some(unit_den),
            _ => return None,
        }
    }

    let positive_den = positive_den?;
    let negative_den = negative_den?;
    if !numerator_denoms
        .iter()
        .any(|den| compare_expr(ctx, *den, positive_den) == Ordering::Equal)
        || !numerator_denoms
            .iter()
            .any(|den| compare_expr(ctx, *den, negative_den) == Ordering::Equal)
    {
        return None;
    }

    let target_numerator = ctx.add(Expr::Add(positive_den, negative_den));
    let target_denominator = ctx.add(Expr::Sub(negative_den, positive_den));
    let target_expr = ctx.add(Expr::Div(target_numerator, target_denominator));
    Some((target_expr, positive_den, negative_den, denominator))
}

pub(super) fn try_build_direct_reciprocal_sum_difference_nested_fraction_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, expr)?;

    for (fraction_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((target_candidate, positive_den, negative_den, nested_denominator)) =
            extract_reciprocal_sum_difference_nested_fraction_target(ctx, fraction_expr)
        else {
            continue;
        };

        if !(exprs_match_for_cancellation(ctx, target_candidate, target_expr)
            || exprs_match_after_default_simplify(ctx, target_candidate, target_expr))
        {
            continue;
        }

        let target_denominator = match ctx.get(target_candidate).clone() {
            Expr::Div(_, target_denominator) => target_denominator,
            _ => unreachable!("direct reciprocal target must be a division"),
        };

        return Some(
            Rewrite::with_local(ctx.num(0), "Simplify Nested Fraction", fraction_expr, target_expr)
                .requires(crate::ImplicitCondition::NonZero(positive_den))
                .requires(crate::ImplicitCondition::NonZero(negative_den))
                .requires(crate::ImplicitCondition::NonZero(nested_denominator))
                .requires(crate::ImplicitCondition::NonZero(target_denominator))
                .substep(
                    "Llevar el numerador y el denominador a común denominador",
                    vec![
                        "La suma y la resta de recíprocos comparten el mismo denominador común."
                            .to_string(),
                    ],
                )
                .substep(
                    "Cancelar el denominador común de numerador y denominador",
                    vec![
                        "Tras limpiar la fracción compleja, ambos lados quedan en la misma forma racional."
                            .to_string(),
                    ],
                )
                .substep(
                    "Cancelar términos iguales",
                    vec![
                        "Una vez simplificada la fracción anidada, la diferencia restante es exacta."
                            .to_string(),
                    ],
                ),
        );
    }

    None
}

pub(super) fn try_build_direct_nested_fraction_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, expr)?;

    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if expr_contains_any_function_call(ctx, source_expr)
            || expr_contains_any_function_call(ctx, target_expr)
            || !matches!(ctx.get(target_expr), Expr::Div(_, _))
        {
            continue;
        }

        let Some(rewrite) = try_rewrite_simplify_nested_fraction_expr(ctx, source_expr) else {
            continue;
        };
        let rewritten = rewrite.rewritten;
        let residual = ctx.add(Expr::Sub(rewritten, target_expr));
        if exprs_match_for_cancellation(ctx, rewritten, target_expr)
            || exprs_match_after_default_simplify(ctx, rewritten, target_expr)
            || is_zero_after_default_simplify(ctx, residual)
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Simplify Nested Fraction",
                source_expr,
                target_expr,
            ));
        }
    }

    None
}

pub(super) fn try_build_direct_fraction_telescoping_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    if !maybe_fraction_telescoping_zero_scope_candidate(ctx, expr) {
        return None;
    }
    let view = AddView::from_expr(ctx, expr);

    let one = ctx.num(1);
    let positive_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .filter(|(_, sign)| *sign == Sign::Pos)
        .collect();
    let negative_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .filter(|(_, sign)| *sign == Sign::Neg)
        .collect();
    if positive_terms.len() == 2 && negative_terms.len() == 1 {
        let negative_den = extract_unit_fraction_denominator(ctx, negative_terms[0].0);
        let positive_denoms = [
            extract_unit_fraction_denominator(ctx, positive_terms[0].0),
            extract_unit_fraction_denominator(ctx, positive_terms[1].0),
        ];

        if let (Some(base), Some(pos_a), Some(pos_b)) =
            (negative_den, positive_denoms[0], positive_denoms[1])
        {
            let successor = ctx.add(Expr::Add(base, one));
            let product = smart_mul(ctx, base, successor);
            let matches = (compare_expr(ctx, pos_a, successor) == Ordering::Equal
                && compare_expr(ctx, pos_b, product) == Ordering::Equal)
                || (compare_expr(ctx, pos_b, successor) == Ordering::Equal
                    && compare_expr(ctx, pos_a, product) == Ordering::Equal);
            if matches {
                return Some(
                    Rewrite::with_local(ctx.num(0), "Subtract Fractions", expr, ctx.num(0))
                        .substep(
                            "Combinar la parte telescópica",
                            vec![
                                "La combinación 1/(a*(a+1)) - 1/a + 1/(a+1) es telescópica exacta, así que vale 0."
                                    .to_string(),
                            ],
                        ),
                );
            }
        }
    }

    for first_index in 0..view.terms.len().saturating_sub(1) {
        for second_index in (first_index + 1)..view.terms.len() {
            let focus_terms: Vec<_> = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index == first_index || index == second_index).then_some(term)
                })
                .collect();
            let remaining_terms: Vec<_> = view
                .terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index != first_index && index != second_index).then_some(term)
                })
                .collect();
            if focus_terms.len() != 2 || remaining_terms.len() != 1 {
                continue;
            }

            let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);
            let mut focus_orders = vec![focus_terms.clone()];
            let mut reversed_focus_terms = focus_terms.clone();
            reversed_focus_terms.reverse();
            focus_orders.push(reversed_focus_terms);

            for focus_order in focus_orders {
                let focus_expr = build_signed_sum_expr(ctx, &focus_order);
                let focus_variants = [
                    (focus_expr, 1_i64),
                    (ctx.add(Expr::Neg(focus_expr)), -1_i64),
                ];

                for (candidate_focus, focus_sign) in focus_variants {
                    let Some(rewritten) =
                        try_rewrite_scaled_sub_fraction_combination_for_cancellation(
                            ctx,
                            candidate_focus,
                        )
                    else {
                        continue;
                    };

                    let adjusted_rewritten = apply_sign_to_expr(ctx, focus_sign, rewritten);
                    if expr_matches_negation_for_cancellation(
                        ctx,
                        adjusted_rewritten,
                        remaining_expr,
                    ) {
                        return Some(
                            Rewrite::with_local(
                                ctx.num(0),
                                "Subtract Fractions",
                                focus_expr,
                                adjusted_rewritten,
                            )
                            .substep(
                                "Cancelar términos iguales",
                                vec![
                                    "Tras combinar las fracciones adecuadas, el término restante es el opuesto y toda la expresión se anula."
                                        .to_string(),
                                ],
                            ),
                        );
                    }
                }
            }
        }
    }

    None
}

fn try_rewrite_sub_fraction_combination_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let (lhs, rhs) = match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) => (lhs, rhs),
        Expr::Add(lhs, rhs) => match ctx.get(rhs).clone() {
            Expr::Neg(inner) => (lhs, inner),
            _ => return None,
        },
        _ => return None,
    };

    let parts = extract_fraction_pair(ctx, lhs, rhs);
    if !(parts.is_frac1 && parts.is_frac2) {
        return None;
    }
    if extract_i64_integer(ctx, parts.n1) != Some(1)
        || extract_i64_integer(ctx, parts.n2) != Some(1)
    {
        return None;
    }
    if !matches!(ctx.get(parts.d1), Expr::Add(_, _) | Expr::Sub(_, _))
        || !matches!(ctx.get(parts.d2), Expr::Add(_, _) | Expr::Sub(_, _))
    {
        return None;
    }

    Some(
        plan_sub_fraction_rewrite_with(
            ctx,
            parts.n1,
            parts.n2,
            parts.d1,
            parts.d2,
            crate::expand::expand,
        )
        .rewritten,
    )
}

pub(super) fn try_rewrite_scaled_sub_fraction_combination_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(rewritten) = try_rewrite_sub_fraction_combination_for_cancellation(ctx, expr) {
        return Some(rewritten);
    }

    let Expr::Mul(lhs, rhs) = ctx.get(expr).clone() else {
        return None;
    };

    if let Some(rewritten) = try_rewrite_sub_fraction_combination_for_cancellation(ctx, lhs) {
        return Some(build_scaled_expr(ctx, rhs, rewritten));
    }

    if let Some(rewritten) = try_rewrite_sub_fraction_combination_for_cancellation(ctx, rhs) {
        return Some(build_scaled_expr(ctx, lhs, rewritten));
    }

    None
}

pub(super) fn reciprocal_three_half_quotient_parts(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let view = MulView::from_expr(ctx, expr);
    let mut quotient_base = None;
    let mut denominator_base = None;

    for factor in view.factors {
        if cas_ast::views::as_rational_const(ctx, factor, 8).is_some_and(|value| value.is_one()) {
            continue;
        }

        if let Some(base) = reciprocal_half_power_base(ctx, factor) {
            if quotient_base.replace(base).is_some() {
                return None;
            }
            continue;
        }

        let (base, exponent) = negative_even_root_power_parts(ctx, factor)?;
        if exponent != BigRational::new(3.into(), 2.into())
            || denominator_base.replace(base).is_some()
        {
            return None;
        }
    }

    Some((quotient_base?, denominator_base?))
}

pub(super) fn scale_division_numerator_denominator_by_rational(
    ctx: &mut cas_ast::Context,
    numerator: cas_ast::ExprId,
    denominator: cas_ast::ExprId,
    scale: BigRational,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    if scale.is_zero() {
        return None;
    }

    let numerator_coeff = scale.numer().clone();
    let denominator_coeff = scale.denom().clone();
    let numerator = match numerator_coeff {
        coeff if coeff == BigInt::one() => numerator,
        coeff if coeff == -BigInt::one() => ctx.add(Expr::Neg(numerator)),
        coeff => {
            let coeff_expr = ctx.add(Expr::Number(BigRational::from_integer(coeff)));
            build_balanced_mul(ctx, &[coeff_expr, numerator])
        }
    };
    let denominator = if denominator_coeff == BigInt::one() {
        denominator
    } else {
        let coeff_expr = ctx.add(Expr::Number(BigRational::from_integer(denominator_coeff)));
        build_balanced_mul(ctx, &[coeff_expr, denominator])
    };

    Some((numerator, denominator))
}

pub(super) fn reject_noncall_product_vs_division_shared_numerator_scale_before_default_simplify(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<bool> {
    for (product_expr, division_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let product_expr =
            strip_unit_negation_for_phase_shift(ctx, product_expr).unwrap_or(product_expr);
        let division_expr =
            strip_unit_negation_for_phase_shift(ctx, division_expr).unwrap_or(division_expr);

        let product_factors = flatten_mul_chain(ctx, product_expr);
        if product_factors.len() != 2
            || !product_factors
                .iter()
                .all(|factor| expr_is_atomic_noncall(ctx, *factor))
        {
            continue;
        }

        let Some((numerator, denominator)) = as_div(ctx, division_expr) else {
            continue;
        };
        if !expr_is_atomic_noncall(ctx, denominator) || is_minus_one_expr(ctx, denominator) {
            continue;
        }
        let one = ctx.num(1);
        if compare_expr(ctx, denominator, one) == Ordering::Equal {
            continue;
        }

        let numerator_factors = flatten_mul_chain(ctx, numerator);
        if numerator_factors.len() != 2
            || !numerator_factors
                .iter()
                .all(|factor| expr_is_atomic_noncall(ctx, *factor))
        {
            continue;
        }

        let shared_factor_count = product_factors
            .iter()
            .filter(|product_factor| {
                numerator_factors.iter().any(|numerator_factor| {
                    compare_expr(ctx, **product_factor, *numerator_factor) == Ordering::Equal
                })
            })
            .count();
        if shared_factor_count >= 1 {
            return Some(false);
        }
    }

    None
}

pub(super) fn extract_same_denominator_residual_cores(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId)> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut denominator = None;
    let mut residual_terms = Vec::with_capacity(2);

    for (term_expr, term_sign) in view.terms.iter().copied() {
        let term_expr = cas_ast::hold::unwrap_internal_hold(ctx, term_expr);
        let Expr::Div(num, den) = ctx.get(term_expr).clone() else {
            return None;
        };

        if let Some(existing_den) = denominator {
            if compare_expr(ctx, den, existing_den) != Ordering::Equal
                && !exprs_equal_up_to_mul_factor_order_and_sign(ctx, den, existing_den)
            {
                return None;
            }
        } else {
            denominator = Some(den);
        }

        residual_terms.push((num, term_sign));
    }

    if residual_terms.len() != 2 {
        return None;
    }

    let lhs_core = apply_sign_to_expr(ctx, sign_to_i64(residual_terms[0].1), residual_terms[0].0);
    let rhs_core = apply_sign_to_expr(
        ctx,
        sign_to_i64(residual_terms[1].1).checked_neg()?,
        residual_terms[1].0,
    );

    Some((denominator?, lhs_core, rhs_core))
}

pub(super) fn try_build_exact_zero_same_denominator_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let (den, lhs_core, rhs_core) = extract_same_denominator_residual_cores(ctx, expr)?;
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let pair_sample = profiling.then(|| {
        format!(
            "{}  ||  {}",
            render_expr_for_orchestrator_profile(ctx, lhs_core),
            render_expr_for_orchestrator_profile(ctx, rhs_core)
        )
    });
    let profile_route = |label: &'static str| {
        if profiling {
            let _ =
                run_profiled_orchestrator_option_section(label, pair_sample.clone(), || Some(()));
        }
    };

    if let Some(child_rewrite) =
        try_build_direct_sub_fraction_combination_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.same_denominator_zero.route.sub_fraction");
        let mut rewrite = Rewrite::with_local(
            ctx.num(0),
            child_rewrite.description.clone(),
            ctx.add(Expr::Sub(lhs_core, rhs_core)),
            ctx.num(0),
        )
        .requires(crate::ImplicitCondition::NonZero(den))
        .requires_all(child_rewrite.required_conditions.clone())
        .assume_all(child_rewrite.assumption_events.clone());

        if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
            rewrite = rewrite.poly_proof(poly_proof);
        }

        rewrite.substeps = child_rewrite.substeps.clone();
        return Some(rewrite);
    }
    if let Some(child_rewrite) =
        try_build_direct_tanh_exp_definition_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.same_denominator_zero.route.tanh_exp");
        let mut rewrite = Rewrite::with_local(
            ctx.num(0),
            child_rewrite.description.clone(),
            ctx.add(Expr::Sub(lhs_core, rhs_core)),
            ctx.num(0),
        )
        .requires(crate::ImplicitCondition::NonZero(den))
        .requires_all(child_rewrite.required_conditions.clone())
        .assume_all(child_rewrite.assumption_events.clone());

        if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
            rewrite = rewrite.poly_proof(poly_proof);
        }

        rewrite.substeps = child_rewrite.substeps.clone();
        return Some(rewrite);
    }
    if let Some(child_rewrite) =
        try_build_direct_trig_square_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.same_denominator_zero.route.trig_square");
        let mut rewrite = Rewrite::with_local(
            ctx.num(0),
            child_rewrite.description.clone(),
            ctx.add(Expr::Sub(lhs_core, rhs_core)),
            ctx.num(0),
        )
        .requires(crate::ImplicitCondition::NonZero(den))
        .requires_all(child_rewrite.required_conditions.clone())
        .assume_all(child_rewrite.assumption_events.clone());

        if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
            rewrite = rewrite.poly_proof(poly_proof);
        }

        rewrite.substeps = child_rewrite.substeps.clone();
        return Some(rewrite);
    }
    if let Some(child_rewrite) =
        try_build_direct_hyperbolic_sinh_cubic_polynomial_equivalence_rewrite(
            ctx, lhs_core, rhs_core,
        )
    {
        profile_route("rule.same_denominator_zero.route.sinh_cubic");
        let mut rewrite = Rewrite::with_local(
            ctx.num(0),
            child_rewrite.description.clone(),
            ctx.add(Expr::Sub(lhs_core, rhs_core)),
            ctx.num(0),
        )
        .requires(crate::ImplicitCondition::NonZero(den))
        .requires_all(child_rewrite.required_conditions.clone())
        .assume_all(child_rewrite.assumption_events.clone());

        if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
            rewrite = rewrite.poly_proof(poly_proof);
        }

        rewrite.substeps = child_rewrite.substeps.clone();
        return Some(rewrite);
    }
    if let Some(child_rewrite) =
        try_build_direct_trig_exact_quarter_phase_shift_pair_equivalence_rewrite(
            ctx, lhs_core, rhs_core,
        )
    {
        profile_route("rule.same_denominator_zero.route.phase_shift_quarter_pair");
        let mut rewrite = Rewrite::with_local(
            ctx.num(0),
            child_rewrite.description.clone(),
            ctx.add(Expr::Sub(lhs_core, rhs_core)),
            ctx.num(0),
        )
        .requires(crate::ImplicitCondition::NonZero(den))
        .requires_all(child_rewrite.required_conditions.clone())
        .assume_all(child_rewrite.assumption_events.clone());

        if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
            rewrite = rewrite.poly_proof(poly_proof);
        }

        rewrite.substeps = child_rewrite.substeps.clone();
        return Some(rewrite);
    }
    if let Some(child_rewrite) =
        try_build_direct_safe_hyperbolic_core_equivalence_rewrite(ctx, lhs_core, rhs_core)
    {
        profile_route("rule.same_denominator_zero.route.safe_hyperbolic");
        let mut rewrite = Rewrite::with_local(
            ctx.num(0),
            child_rewrite.description.clone(),
            ctx.add(Expr::Sub(lhs_core, rhs_core)),
            ctx.num(0),
        )
        .requires(crate::ImplicitCondition::NonZero(den))
        .requires_all(child_rewrite.required_conditions.clone())
        .assume_all(child_rewrite.assumption_events.clone());

        if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
            rewrite = rewrite.poly_proof(poly_proof);
        }

        rewrite.substeps = child_rewrite.substeps.clone();
        return Some(rewrite);
    }
    let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if let Some(child_rewrite) =
        try_build_exact_zero_shared_passthrough_difference_rewrite(ctx, residual_expr)
    {
        profile_route("rule.same_denominator_zero.route.shared_passthrough");
        let mut rewrite = Rewrite::with_local(
            ctx.num(0),
            child_rewrite.description.clone(),
            residual_expr,
            ctx.num(0),
        )
        .requires(crate::ImplicitCondition::NonZero(den))
        .requires_all(child_rewrite.required_conditions.clone())
        .assume_all(child_rewrite.assumption_events.clone());

        if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
            rewrite = rewrite.poly_proof(poly_proof);
        }

        rewrite.substeps = child_rewrite.substeps.clone();
        return Some(rewrite);
    }
    let residual_term_count = AddView::from_expr(ctx, residual_expr).terms.len();
    if residual_term_count <= 4 {
        if let Some(child_rewrite) =
            try_build_exact_zero_identity_rewrite_direct(ctx, residual_expr)
        {
            let zero = ctx.num(0);
            if compare_expr(ctx, child_rewrite.final_expr(), zero) == Ordering::Equal {
                profile_route("rule.same_denominator_zero.route.direct_identity");
                let mut rewrite = Rewrite::with_local(
                    zero,
                    child_rewrite.description.clone(),
                    residual_expr,
                    zero,
                )
                .requires(crate::ImplicitCondition::NonZero(den))
                .requires_all(child_rewrite.required_conditions.clone())
                .assume_all(child_rewrite.assumption_events.clone());

                if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
                    rewrite = rewrite.poly_proof(poly_proof);
                }

                rewrite.substeps = child_rewrite.substeps.clone();
                return Some(rewrite);
            }
        }
    }
    let child_rewrite = try_build_repeated_trig_phase_shift_pair_zero_rewrite(ctx, residual_expr)
        .inspect(|_| {
            profile_route("rule.same_denominator_zero.route.tail_phase_shift_pair");
        })
        .or_else(|| {
            try_build_stripped_zero_log_identity_child_rewrite(ctx, residual_expr).inspect(|_| {
                profile_route("rule.same_denominator_zero.route.tail_stripped_zero_log");
            })
        })
        .or_else(|| {
            try_build_fast_multiterm_hyperbolic_residual_child_rewrite(ctx, residual_expr).inspect(
                |_| {
                    profile_route(
                        "rule.same_denominator_zero.route.tail_fast_multiterm_hyperbolic",
                    );
                },
            )
        })
        .or_else(|| {
            try_build_direct_safe_hyperbolic_core_equivalence_rewrite(ctx, lhs_core, rhs_core)
                .inspect(|_| {
                    profile_route("rule.same_denominator_zero.route.tail_safe_hyperbolic");
                })
        })
        .or_else(|| {
            try_build_exact_zero_shared_passthrough_difference_rewrite(ctx, residual_expr).inspect(
                |_| {
                    profile_route("rule.same_denominator_zero.route.tail_shared_passthrough");
                },
            )
        })
        .or_else(|| {
            try_build_fast_trig_residual_identity_child_rewrite(ctx, residual_expr).inspect(|_| {
                profile_route("rule.same_denominator_zero.route.tail_fast_trig_residual");
            })
        })
        .or_else(|| {
            try_build_fast_small_polynomial_residual_child_rewrite(ctx, residual_expr).inspect(
                |_| {
                    profile_route("rule.same_denominator_zero.route.tail_fast_small_polynomial");
                },
            )
        })
        .or_else(|| {
            try_build_same_denominator_tail_trig_ratio_equivalence_rewrite(ctx, lhs_core, rhs_core)
                .inspect(|_| {
                    profile_route("rule.same_denominator_zero.route.tail_trig_ratio_core");
                })
        })
        .or_else(|| {
            try_build_direct_core_equivalence_rewrite(ctx, lhs_core, rhs_core).inspect(|_| {
                profile_route("rule.same_denominator_zero.route.tail_direct_core_equivalence");
                profile_same_denominator_tail_direct_core_equivalence_family(
                    ctx,
                    lhs_core,
                    rhs_core,
                    pair_sample.clone(),
                );
            })
        })
        .or_else(|| {
            try_build_exact_zero_identity_rewrite(ctx, residual_expr).inspect(|_| {
                profile_route("rule.same_denominator_zero.route.tail_exact_zero_identity");
            })
        })?;

    let mut rewrite = Rewrite::with_local(
        ctx.num(0),
        child_rewrite.description.clone(),
        residual_expr,
        ctx.num(0),
    )
    .requires(crate::ImplicitCondition::NonZero(den))
    .requires_all(child_rewrite.required_conditions.clone())
    .assume_all(child_rewrite.assumption_events.clone());

    if let Some(poly_proof) = child_rewrite.poly_proof.clone() {
        rewrite = rewrite.poly_proof(poly_proof);
    }

    rewrite.substeps = child_rewrite.substeps.clone();
    Some(rewrite)
}

pub(super) fn extract_fraction_like_add_term(
    ctx: &mut cas_ast::Context,
    term_expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    match ctx.get(term_expr).clone() {
        Expr::Div(num, den) => Some((num, den)),
        Expr::Neg(inner) => {
            let Expr::Div(num, den) = ctx.get(inner).clone() else {
                return None;
            };
            Some((ctx.add(Expr::Neg(num)), den))
        }
        _ => None,
    }
}
