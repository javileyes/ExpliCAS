//! Orquestador: familia `pairing` (troceo P1).
//!
//! Ver la cabecera de `orchestrator.rs` para el contexto.

use super::*;

pub(super) fn is_nested_additive_pair_root(ctx: &Context, expr: ExprId) -> bool {
    let (lhs, rhs) = match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => (*lhs, *rhs),
        _ => return false,
    };

    matches!(ctx.get(lhs), Expr::Add(_, _) | Expr::Sub(_, _))
        && matches!(ctx.get(rhs), Expr::Add(_, _) | Expr::Sub(_, _))
}

pub(super) fn sort_direct_pair_args_root(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> (ExprId, ExprId) {
    if compare_expr(ctx, lhs, rhs) == Ordering::Greater {
        (rhs, lhs)
    } else {
        (lhs, rhs)
    }
}

pub(super) fn matches_direct_small_pow_expansion_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    if expr_contains_trig_or_hyperbolic_builtin_local(ctx, lhs_core)
        || expr_contains_trig_or_hyperbolic_builtin_local(ctx, rhs_core)
    {
        return false;
    }

    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(expanded) = try_expand_small_pow_sum_expr(
            ctx,
            source,
            SmallPowExpandPolicy {
                max_vars: 3,
                ..SmallPowExpandPolicy::default()
            },
        )
        .map(|expanded| cas_ast::hold::unwrap_hold(ctx, expanded))
        .or_else(|| try_expand_binomial_pow_expr(ctx, source, 2, 6).map(|plan| plan.expanded)) else {
            continue;
        };
        if compare_expr(ctx, expanded, target) == Ordering::Equal {
            return true;
        }
        if cas_math::poly_compare::poly_eq(ctx, expanded, target) {
            return true;
        }
        if cas_ast::count_nodes(ctx, expanded) <= 24 && cas_ast::count_nodes(ctx, target) <= 24 {
            if isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                expanded,
                target,
            ) {
                return true;
            }

            let difference = ctx.add(Expr::Sub(expanded, target));
            if isolated_simplify_rewrites_to_zero(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                difference,
            ) {
                return true;
            }
        }
    }

    false
}

fn matches_direct_short_geometric_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (expanded_expr, factored_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if let Some(factored) = factor_short_geometric_sum_partner_root(ctx, expanded_expr) {
            if compare_expr(ctx, factored, factored_expr) == Ordering::Equal {
                return true;
            }
        }

        let Some(base) = extract_direct_short_geometric_product_base_root(ctx, factored_expr)
        else {
            continue;
        };
        let expanded = build_direct_short_geometric_sum_expanded_target_root(ctx, base);
        if compare_expr(ctx, expanded, expanded_expr) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn extract_direct_factored_linear_shift_pair_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for (base_index, shifted_index) in [(0usize, 1usize), (1usize, 0usize)] {
        let base = factors[base_index];
        let Some((shifted_base, constant)) =
            extract_base_plus_constant_root(ctx, factors[shifted_index])
        else {
            continue;
        };
        if compare_expr(ctx, base, shifted_base) == Ordering::Equal {
            return Some((base, constant));
        }
    }

    None
}

pub(super) fn extract_direct_expanded_linear_shift_pair_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut square_base = None;
    let mut linear_term = None;
    for (term_expr, term_sign) in view.terms {
        if let Some(base) = extract_plain_pow2_base_root(ctx, term_expr) {
            if term_sign == Sign::Neg || square_base.is_some() {
                return None;
            }
            square_base = Some(base);
            continue;
        }

        let (mut coeff, base) = extract_coef_and_base(ctx, term_expr);
        if term_sign == Sign::Neg {
            coeff = -coeff;
        }
        if linear_term.is_some() || coeff.is_zero() {
            return None;
        }
        linear_term = Some((base, coeff));
    }

    let (Some(square_base), Some((linear_base, constant))) = (square_base, linear_term) else {
        return None;
    };
    (compare_expr(ctx, square_base, linear_base) == Ordering::Equal)
        .then_some((square_base, constant))
}

pub(super) fn matches_direct_linear_factoring_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let lhs = extract_direct_expanded_linear_shift_pair_root(ctx, lhs_core)
        .or_else(|| extract_direct_factored_linear_shift_pair_root(ctx, lhs_core));
    let rhs = extract_direct_expanded_linear_shift_pair_root(ctx, rhs_core)
        .or_else(|| extract_direct_factored_linear_shift_pair_root(ctx, rhs_core));

    let (Some((lhs_base, lhs_constant)), Some((rhs_base, rhs_constant))) = (lhs, rhs) else {
        return false;
    };
    compare_expr(ctx, lhs_base, rhs_base) == Ordering::Equal && lhs_constant == rhs_constant
}

pub(super) fn matches_direct_two_linear_shift_product_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (product_expr, expanded_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((product_base, product_constants)) =
            extract_direct_two_linear_shift_product_root(ctx, product_expr)
        else {
            continue;
        };

        let Some(rewrite) = try_rewrite_automatic_factor_expr(ctx, expanded_expr) else {
            continue;
        };
        let factored = strip_multiplicative_one_root(ctx, rewrite.rewritten);
        if compare_expr(ctx, factored, product_expr) == Ordering::Equal {
            return true;
        }

        let Some((factored_base, factored_constants)) =
            extract_direct_two_linear_shift_product_root(ctx, factored)
        else {
            continue;
        };

        if compare_expr(ctx, product_base, factored_base) == Ordering::Equal
            && product_constants == factored_constants
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_three_linear_shift_product_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (product_expr, expanded_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((product_base, product_constants)) =
            extract_direct_three_linear_shift_product_root(ctx, product_expr)
        else {
            continue;
        };

        let Some(rewrite) = try_rewrite_automatic_factor_expr(ctx, expanded_expr) else {
            continue;
        };
        let factored = strip_multiplicative_one_root(ctx, rewrite.rewritten);
        if compare_expr(ctx, factored, product_expr) == Ordering::Equal {
            return true;
        }

        let Some((factored_base, factored_constants)) =
            extract_direct_three_linear_shift_product_root(ctx, factored)
        else {
            continue;
        };

        if compare_expr(ctx, product_base, factored_base) == Ordering::Equal
            && product_constants == factored_constants
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_quartic_gcf_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (expanded_expr, factored_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(expanded_base) = extract_direct_quartic_gcf_base_expanded_root(ctx, expanded_expr)
        else {
            continue;
        };
        let Some(factored_base) = extract_direct_quartic_gcf_base_factored_root(ctx, factored_expr)
        else {
            continue;
        };
        if compare_expr(ctx, expanded_base, factored_base) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_weierstrass_contraction_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_weierstrass_contraction_div_expr(ctx, source_expr) else {
            continue;
        };
        let rewritten = strip_multiplicative_one_root(ctx, rewrite.rewritten);
        let target = strip_multiplicative_one_root(ctx, target_expr);
        if compare_expr(ctx, rewritten, target) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, rewritten) <= 24
            && cas_ast::count_nodes(ctx, target) <= 24
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                rewritten,
                target,
            )
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_small_exact_constant_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if !matches!(ctx.get(target_expr), Expr::Number(_)) {
            continue;
        }
        if cas_ast::count_nodes(ctx, source_expr) > 16 {
            continue;
        }
        if isolated_simplify_rewrites_to_target(
            &crate::phase::SimplifyOptions::default(),
            ctx,
            source_expr,
            target_expr,
        ) {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_sum_diff_cubes_product_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    fn extract_sum_diff_cubes_compact_bases_root(
        ctx: &mut Context,
        expr: ExprId,
    ) -> Option<(ExprId, ExprId, bool)> {
        let view = AddView::from_expr(ctx, expr);
        if view.terms.len() != 2 {
            return None;
        }

        let mut positive_bases = Vec::with_capacity(2);
        let mut negative_bases = Vec::with_capacity(1);
        for (term_expr, term_sign) in view.terms {
            let base = extract_plain_cube_base_root(ctx, term_expr)?;
            match term_sign {
                Sign::Pos => positive_bases.push(base),
                Sign::Neg => negative_bases.push(base),
            }
        }

        match (&positive_bases[..], &negative_bases[..]) {
            ([a, b], []) => Some((*a, *b, false)),
            ([a], [b]) => Some((*a, *b, true)),
            _ => None,
        }
    }

    fn matches_sum_diff_cubes_binomial_root(
        ctx: &mut Context,
        expr: ExprId,
        lhs: ExprId,
        rhs: ExprId,
        is_difference: bool,
    ) -> bool {
        let view = AddView::from_expr(ctx, expr);
        if view.terms.len() != 2 {
            return false;
        }

        if is_difference {
            let mut positive = None;
            let mut negative = None;
            for (term, sign) in view.terms {
                match sign {
                    Sign::Pos => positive = Some(term),
                    Sign::Neg => negative = Some(term),
                }
            }
            return positive.is_some_and(|term| compare_expr(ctx, term, lhs) == Ordering::Equal)
                && negative.is_some_and(|term| compare_expr(ctx, term, rhs) == Ordering::Equal);
        }

        let lhs_matches = compare_expr(ctx, view.terms[0].0, lhs) == Ordering::Equal
            && compare_expr(ctx, view.terms[1].0, rhs) == Ordering::Equal;
        let rhs_matches = compare_expr(ctx, view.terms[0].0, rhs) == Ordering::Equal
            && compare_expr(ctx, view.terms[1].0, lhs) == Ordering::Equal;
        view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) && (lhs_matches || rhs_matches)
    }

    fn matches_sum_diff_cubes_trinomial_root(
        ctx: &mut Context,
        expr: ExprId,
        lhs: ExprId,
        rhs: ExprId,
        is_difference: bool,
    ) -> bool {
        let terms = AddView::from_expr(ctx, expr).terms;
        if terms.len() != 3 {
            return false;
        }

        let two = ctx.num(2);
        let lhs_sq = ctx.add(Expr::Pow(lhs, two));
        let rhs_sq = ctx.add(Expr::Pow(rhs, two));
        let middle = smart_mul(ctx, lhs, rhs);
        let expected_middle_sign = if is_difference { Sign::Pos } else { Sign::Neg };

        let mut found_lhs_sq = false;
        let mut found_rhs_sq = false;
        let mut found_middle = false;
        for (term, sign) in terms {
            if sign == Sign::Pos
                && !found_lhs_sq
                && compare_expr(ctx, term, lhs_sq) == Ordering::Equal
            {
                found_lhs_sq = true;
                continue;
            }
            if sign == Sign::Pos
                && !found_rhs_sq
                && compare_expr(ctx, term, rhs_sq) == Ordering::Equal
            {
                found_rhs_sq = true;
                continue;
            }
            if sign == expected_middle_sign
                && !found_middle
                && compare_expr(ctx, term, middle) == Ordering::Equal
            {
                found_middle = true;
            }
        }

        found_lhs_sq && found_rhs_sq && found_middle
    }

    fn matches_sum_diff_cubes_product_from_compact_root(
        ctx: &mut Context,
        product_expr: ExprId,
        lhs: ExprId,
        rhs: ExprId,
        is_difference: bool,
    ) -> bool {
        let factors = flatten_mul_chain(ctx, product_expr);
        if factors.len() != 2 {
            return false;
        }

        [(factors[0], factors[1]), (factors[1], factors[0])]
            .into_iter()
            .any(|(binomial, trinomial)| {
                matches_sum_diff_cubes_binomial_root(ctx, binomial, lhs, rhs, is_difference)
                    && matches_sum_diff_cubes_trinomial_root(
                        ctx,
                        trinomial,
                        lhs,
                        rhs,
                        is_difference,
                    )
            })
    }

    for (product_expr, compact_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((lhs, rhs, is_difference)) =
            extract_sum_diff_cubes_compact_bases_root(ctx, compact_expr)
        else {
            continue;
        };

        if matches_sum_diff_cubes_product_from_compact_root(
            ctx,
            product_expr,
            lhs,
            rhs,
            is_difference,
        ) {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_higher_degree_difference_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    fn matches_geometric_quadratic_factor_root(
        ctx: &mut Context,
        expr: ExprId,
        base: ExprId,
        positive_linear: bool,
    ) -> bool {
        let terms = AddView::from_expr(ctx, expr).terms;
        if terms.len() != 3 {
            return false;
        }

        let two = ctx.num(2);
        let base_sq = ctx.add(Expr::Pow(base, two));
        let mut found_base_sq = false;
        let mut found_linear = false;
        let mut found_one = false;
        for (term, sign) in terms {
            if sign == Sign::Pos
                && !found_base_sq
                && compare_expr(ctx, term, base_sq) == Ordering::Equal
            {
                found_base_sq = true;
                continue;
            }
            if sign == Sign::Pos && !found_one && extract_i64_integer(ctx, term) == Some(1) {
                found_one = true;
                continue;
            }
            if sign
                == if positive_linear {
                    Sign::Pos
                } else {
                    Sign::Neg
                }
                && !found_linear
                && compare_expr(ctx, term, base) == Ordering::Equal
            {
                found_linear = true;
                continue;
            }
        }

        found_base_sq && found_linear && found_one
    }

    fn extract_base_from_sixth_power_minus_one_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
        let Expr::Sub(lhs, rhs) = ctx.get(expr) else {
            return None;
        };
        if extract_i64_integer(ctx, *rhs) != Some(1) {
            return None;
        }
        let Expr::Pow(base, exponent) = ctx.get(*lhs) else {
            return None;
        };
        (extract_i64_integer(ctx, *exponent) == Some(6)).then_some(*base)
    }

    for (compact_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(base) = extract_base_from_sixth_power_minus_one_root(ctx, compact_expr) else {
            continue;
        };
        let product_factors = flatten_mul_chain(ctx, product_expr);
        if product_factors.len() != 4 {
            continue;
        }

        let mut saw_plus_one = false;
        let mut saw_minus_one = false;
        let mut saw_positive_quadratic = false;
        let mut saw_negative_quadratic = false;
        let mut invalid = false;
        for factor in product_factors {
            if let Some((factor_base, constant)) = extract_base_plus_constant_root(ctx, factor) {
                if compare_expr(ctx, factor_base, base) == Ordering::Equal {
                    if constant == BigRational::one() && !saw_plus_one {
                        saw_plus_one = true;
                        continue;
                    }
                    if constant == -BigRational::one() && !saw_minus_one {
                        saw_minus_one = true;
                        continue;
                    }
                }
            }
            if !saw_positive_quadratic
                && matches_geometric_quadratic_factor_root(ctx, factor, base, true)
            {
                saw_positive_quadratic = true;
                continue;
            }
            if !saw_negative_quadratic
                && matches_geometric_quadratic_factor_root(ctx, factor, base, false)
            {
                saw_negative_quadratic = true;
                continue;
            }
            invalid = true;
            break;
        }

        if !invalid
            && saw_plus_one
            && saw_minus_one
            && saw_positive_quadratic
            && saw_negative_quadratic
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_sophie_germain_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (compact_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if let Some((a, b)) = extract_sophie_germain_bases_root(ctx, compact_expr) {
            let product_factors = flatten_mul_chain(ctx, product_expr);
            if product_factors.len() == 2 {
                let positive_quadratic = build_sophie_germain_quadratic_expr_root(ctx, a, b, true);
                let negative_quadratic = build_sophie_germain_quadratic_expr_root(ctx, a, b, false);
                if (compare_expr(ctx, product_factors[0], positive_quadratic) == Ordering::Equal
                    && compare_expr(ctx, product_factors[1], negative_quadratic) == Ordering::Equal)
                    || (compare_expr(ctx, product_factors[0], negative_quadratic)
                        == Ordering::Equal
                        && compare_expr(ctx, product_factors[1], positive_quadratic)
                            == Ordering::Equal)
                    || (matches_sophie_germain_quadratic_root(ctx, product_factors[0], a, b, true)
                        && matches_sophie_germain_quadratic_root(
                            ctx,
                            product_factors[1],
                            a,
                            b,
                            false,
                        ))
                    || (matches_sophie_germain_quadratic_root(ctx, product_factors[0], a, b, false)
                        && matches_sophie_germain_quadratic_root(
                            ctx,
                            product_factors[1],
                            a,
                            b,
                            true,
                        ))
                {
                    return true;
                }
            }

            if cas_ast::count_nodes(ctx, compact_expr) + cas_ast::count_nodes(ctx, product_expr)
                <= 48
                && cas_math::poly_compare::poly_eq(ctx, compact_expr, product_expr)
            {
                return true;
            }
        }

        if let Some(factored_expr) = factor_sophie_germain_partner_root(ctx, compact_expr) {
            if compare_expr(ctx, factored_expr, product_expr) == Ordering::Equal {
                return true;
            }
        }
    }

    false
}

pub(super) fn matches_known_direct_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    matches_direct_addition_of_successive_unit_fractions_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_reciprocal_sqrt_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_rationalized_sum_of_sqrts_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_exponential_combination_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_exp_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_normalized_trig_product_to_sum_sin_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_numeric_general_phase_shift_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_phase_shift_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_tangent_addition_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_tan_angle_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_weierstrass_contraction_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_sum_to_product_contraction_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_power_mixed_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_half_angle_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_scaled_half_angle_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_half_angle_tan_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cos_fourth_power_reduction_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_half_angle_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_sin_sin_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_sin_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_cos_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_pythagorean_identity_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_pythagorean_extended_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_nested_fraction_simplified_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_reciprocal_sum_difference_nested_fraction_pair_root(
            ctx, lhs_core, rhs_core,
        )
        || matches_direct_hyperbolic_sinh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_cosh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_cosh_difference_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_recursive_hyperbolic_sinh_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_recursive_hyperbolic_cosh_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_sinh_double_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_cosh_double_angle_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_double_angle_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_pure_double_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_double_angle_inverse_trig_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_sine_double_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_mixed_double_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_quintuple_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_cubic_cosine_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_binomial_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_abs_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_abs_trig_half_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_positive_double_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_angle_sum_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_small_pow_expansion_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_short_geometric_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_quartic_gcf_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_perfect_square_trinomial_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_linear_factoring_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_two_linear_shift_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_three_linear_shift_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_difference_of_squares_quotient_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_sum_diff_cubes_quotient_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_tan_to_sec_pythagorean_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_sec_tan_pythagorean_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cot_to_csc_pythagorean_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_csc_cot_pythagorean_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_pythagorean_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_reciprocal_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_ratio_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_ratio_alias_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_reciprocal_trig_product_one_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_triple_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_triple_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_special_angle_exact_value_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_inverse_composition_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_from_exp_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_tanh_to_sinh_cosh_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cube_root_rationalization_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_sum_diff_cubes_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_higher_degree_difference_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_sophie_germain_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cos_product_telescoping_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_tanh_pythagorean_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_small_exact_constant_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_rational_plus_minus_one_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_tanh_double_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_sum_of_squares_product_pair_root(ctx, lhs_core, rhs_core)
}

pub(super) fn matches_direct_tangent_addition_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (tan_sum_expr, fraction_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((lhs_arg, rhs_arg)) =
            extract_direct_tangent_addition_target_root(ctx, tan_sum_expr)
        else {
            continue;
        };
        let Some((fraction_lhs, fraction_rhs)) =
            extract_direct_tangent_addition_fraction_target_root(ctx, fraction_expr)
        else {
            continue;
        };
        if matches_unordered_expr_pair_root(ctx, lhs_arg, rhs_arg, fraction_lhs, fraction_rhs) {
            return true;
        }
    }

    false
}

pub(super) fn extract_mul_pair_root(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    Some((factors[0], factors[1]))
}

pub(super) fn factors_match_by_equality_or_direct_pair_root(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    fn normalize_negative_unit_product_root(ctx: &mut Context, expr: ExprId) -> ExprId {
        let factors = flatten_mul_chain(ctx, expr);
        if factors.len() != 2 {
            return expr;
        }

        if extract_i64_integer(ctx, factors[0]) == Some(-1) {
            return ctx.add(Expr::Neg(factors[1]));
        }
        if extract_i64_integer(ctx, factors[1]) == Some(-1) {
            return ctx.add(Expr::Neg(factors[0]));
        }

        expr
    }

    let lhs = normalize_negative_unit_product_root(ctx, lhs);
    let rhs = normalize_negative_unit_product_root(ctx, rhs);

    if compare_expr(ctx, lhs, rhs) == Ordering::Equal
        || matches_direct_addition_of_successive_unit_fractions_pair_root(ctx, lhs, rhs)
        || matches_known_direct_pair_root(ctx, lhs, rhs)
        || matches_direct_half_angle_binomial_square_pair_root(ctx, lhs, rhs)
    {
        return true;
    }

    if expr_contains_trig_or_hyperbolic_builtin_local(ctx, lhs)
        || expr_contains_trig_or_hyperbolic_builtin_local(ctx, rhs)
        || cas_ast::count_nodes(ctx, lhs) > 48
        || cas_ast::count_nodes(ctx, rhs) > 48
        || (matches!(ctx.get(lhs), Expr::Mul(_, _)) && matches!(ctx.get(rhs), Expr::Mul(_, _)))
    {
        return false;
    }

    let difference = ctx.add(Expr::Sub(lhs, rhs));
    cas_ast::count_nodes(ctx, difference) <= 96
        && isolated_simplify_rewrites_to_zero(
            &crate::phase::SimplifyOptions::default(),
            ctx,
            difference,
        )
}

pub(super) fn try_standard_common_scale_known_pair_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    // Factoring a non-finite common scale and zeroing the residual is unsound:
    // `inf - inf` is `inf * (1 - 1)`, not `inf * 0 = 0`. The same holds for a
    // `(1/0) - (1/0)` or `undefined - undefined` difference. Decline so the
    // expression stays symbolic rather than collapsing to `0`.
    if crate::rules::arithmetic::additive_term_is_nonfinite_or_undefined(ctx, expr) {
        return None;
    }

    let (_common_factor, residual_expr) =
        extract_common_multiplicative_residual_sum_root(ctx, expr)?;
    if !matches_direct_small_zero_or_known_pair_residual_root(ctx, residual_expr) {
        return None;
    }

    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    Some(finish_common_scale_zero_shortcut_with_domain_meta(
        ctx,
        expr,
        &parent_ctx,
        collect_steps,
    ))
}

pub(super) fn factor_sum_diff_cubes_partner_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut positive_bases = Vec::with_capacity(2);
    let mut negative_bases = Vec::with_capacity(1);
    for (term_expr, term_sign) in view.terms {
        let base = extract_plain_cube_base_root(ctx, term_expr)?;
        match term_sign {
            Sign::Pos => positive_bases.push(base),
            Sign::Neg => negative_bases.push(base),
        }
    }

    let (a, b, is_difference) = match (&positive_bases[..], &negative_bases[..]) {
        ([a, b], []) => (*a, *b, false),
        ([a], [b]) => (*a, *b, true),
        _ => return None,
    };

    let two = ctx.num(2);
    let a_sq = ctx.add(Expr::Pow(a, two));
    let b_sq = ctx.add(Expr::Pow(b, two));
    let ab = smart_mul(ctx, a, b);
    let trinomial = if is_difference {
        let inner = ctx.add(Expr::Add(ab, b_sq));
        ctx.add(Expr::Add(a_sq, inner))
    } else {
        let neg_ab = ctx.add(Expr::Neg(ab));
        let inner = ctx.add(Expr::Add(neg_ab, b_sq));
        ctx.add(Expr::Add(a_sq, inner))
    };
    let binomial = if is_difference {
        ctx.add(Expr::Sub(a, b))
    } else {
        ctx.add(Expr::Add(a, b))
    };

    Some(build_mul_expr_from_factors_root(
        ctx,
        &[binomial, trinomial],
    ))
}

pub(super) fn factor_higher_degree_difference_partner_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (lhs, rhs) = match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) => (lhs, rhs),
        _ => return None,
    };
    if extract_i64_integer(ctx, rhs) != Some(1) {
        return None;
    }
    let (base, exponent) = match ctx.get(lhs).clone() {
        Expr::Pow(base, exponent) => (base, exponent),
        _ => return None,
    };
    if extract_i64_integer(ctx, exponent) != Some(6) {
        return None;
    }

    let one = ctx.num(1);
    let plus_one = ctx.add(Expr::Add(base, one));
    let minus_one = ctx.add(Expr::Sub(base, one));
    let two = ctx.num(2);
    let base_sq = ctx.add(Expr::Pow(base, two));
    let positive_quad = build_balanced_add(ctx, &[base_sq, base, one]);
    let negative_base = ctx.add(Expr::Neg(base));
    let negative_quad = build_balanced_add(ctx, &[base_sq, negative_base, one]);
    Some(build_mul_expr_from_factors_root(
        ctx,
        &[positive_quad, negative_quad, plus_one, minus_one],
    ))
}

pub(super) fn factor_sophie_germain_partner_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factored = cas_math::factor::factor_sophie_germain(ctx, expr)?;
    (compare_expr(ctx, factored, expr) != Ordering::Equal).then_some(factored)
}

fn is_potential_known_small_polynomial_partner_source_root(ctx: &Context, expr: ExprId) -> bool {
    is_function_free_arithmetic_expr_root(ctx, expr)
        && cas_ast::count_nodes(ctx, expr) <= 24
        && matches!(
            ctx.get(expr),
            Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_) | Expr::Pow(_, _)
        )
}

pub(super) fn factor_short_geometric_sum_partner_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 4 || terms.iter().any(|(_, sign)| *sign != Sign::Pos) {
        return None;
    }

    for (term_expr, _) in &terms {
        let base = match ctx.get(*term_expr).clone() {
            Expr::Pow(base, exponent) => {
                let exponent = extract_i64_integer(ctx, exponent)?;
                if (1..=3).contains(&exponent) {
                    base
                } else {
                    continue;
                }
            }
            _ if extract_i64_integer(ctx, *term_expr) != Some(1) => *term_expr,
            _ => continue,
        };

        if !matches_geometric_series_sum_root(ctx, expr, base, 3) {
            continue;
        }

        let one = ctx.num(1);
        let two = ctx.num(2);
        let linear = ctx.add(Expr::Add(base, one));
        let squared = ctx.add(Expr::Pow(base, two));
        let quadratic = ctx.add(Expr::Add(squared, one));
        return Some(ctx.add(Expr::Mul(linear, quadratic)));
    }

    None
}

pub(super) fn factor_small_linear_shift_product_partner_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if extract_direct_two_linear_shift_product_root(ctx, expr).is_some()
        || extract_direct_three_linear_shift_product_root(ctx, expr).is_some()
    {
        return Some(expr);
    }
    if !matches!(
        ctx.get(expr),
        Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_)
    ) {
        return None;
    }
    if cas_ast::collect_variables(ctx, expr).len() != 1 || cas_ast::count_nodes(ctx, expr) > 24 {
        return None;
    }

    let rewrite = try_rewrite_automatic_factor_expr(ctx, expr)?;
    let factored = strip_multiplicative_one_root(ctx, rewrite.rewritten);
    if extract_direct_two_linear_shift_product_root(ctx, factored).is_some()
        || extract_direct_three_linear_shift_product_root(ctx, factored).is_some()
    {
        return Some(factored);
    }

    None
}

pub(super) fn is_potential_small_linear_shift_product_partner_source_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    extract_direct_two_linear_shift_product_root(ctx, expr).is_some()
        || extract_direct_three_linear_shift_product_root(ctx, expr).is_some()
        || (matches!(
            ctx.get(expr),
            Expr::Add(_, _) | Expr::Sub(_, _) | Expr::Neg(_)
        ) && cas_ast::collect_variables(ctx, expr).len() == 1
            && cas_ast::count_nodes(ctx, expr) <= 24)
}

pub(super) fn is_safe_direct_pair_anchor_target_root(ctx: &mut Context, expr: ExprId) -> bool {
    (cas_ast::collect_variables(ctx, expr).is_empty() && cas_ast::count_nodes(ctx, expr) <= 16)
        || extract_plain_sinh_or_cosh_arg_root(ctx, expr).is_some()
        || extract_unary_builtin_arg_root(ctx, expr, BuiltinFn::Tanh).is_some()
        || extract_direct_tangent_addition_fraction_target_root(ctx, expr).is_some()
}

fn canonicalize_safe_anchor_direct_partner_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Some((trig_fn, full_arg)) =
        extract_direct_scaled_half_angle_square_target_root(ctx, expr)
    {
        if trig_fn == BuiltinFn::Cos {
            return Some(build_scaled_half_angle_pow2_target_root(
                ctx, trig_fn, full_arg,
            ));
        }
    }

    if let Some((trig_fn, full_arg)) = extract_direct_abs_trig_half_angle_target_root(ctx, expr) {
        return Some(build_direct_sqrt_abs_trig_half_angle_target_root(
            ctx, trig_fn, full_arg,
        ));
    }

    if let Some(base) = extract_addition_of_successive_unit_fractions_arg_root(ctx, expr) {
        return Some(build_collapsed_successive_unit_fractions_expr_root(
            ctx, base,
        ));
    }

    if matches!(ctx.get(expr), Expr::Function(_, _)) {
        let expanded = expand_logs_collect_positive_assumptions(ctx, expr).rewritten;
        if compare_expr(ctx, expanded, expr) != Ordering::Equal {
            return Some(strip_multiplicative_one_root(ctx, expanded));
        }
    }

    if let Some(rewritten) = rewrite_small_exp_product_root(ctx, expr) {
        if compare_expr(ctx, rewritten, expr) != Ordering::Equal {
            return Some(strip_multiplicative_one_root(ctx, rewritten));
        }
    }

    None
}

pub(super) fn try_standard_two_factor_small_partner_canonicalization_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (numeric_coeff, factors) = extract_signed_two_factor_product_root(ctx, expr)?;
    if !factors
        .iter()
        .copied()
        .any(|factor| is_potential_known_small_polynomial_partner_source_root(ctx, factor))
    {
        return None;
    }

    for partner_index in 0..2 {
        let other_index = 1 - partner_index;
        if !is_potential_known_small_polynomial_partner_source_root(ctx, factors[partner_index]) {
            continue;
        }
        let Some(partner_factored) =
            factor_known_small_polynomial_partner_root(ctx, factors[partner_index])
        else {
            continue;
        };
        if extract_special_angle_exact_value_root(ctx, factors[other_index]).is_some() {
            continue;
        }
        if is_potential_known_small_polynomial_partner_source_root(ctx, factors[other_index])
            && factor_known_small_polynomial_partner_root(ctx, factors[other_index]).is_some()
        {
            continue;
        }

        let other_canonical = canonicalize_direct_pair_factor_root(ctx, factors[other_index])
            .unwrap_or(factors[other_index]);
        if compare_expr(ctx, partner_factored, factors[partner_index]) == Ordering::Equal
            && compare_expr(ctx, other_canonical, factors[other_index]) == Ordering::Equal
        {
            continue;
        }

        let base_factors = if partner_index == 0 {
            vec![partner_factored, other_canonical]
        } else {
            vec![other_canonical, partner_factored]
        };
        let rewritten_factors = if numeric_coeff == BigRational::from_integer((-1).into()) {
            let minus_one = ctx.num(-1);
            let mut factors_with_sign = Vec::with_capacity(3);
            factors_with_sign.push(minus_one);
            factors_with_sign.extend(base_factors);
            factors_with_sign
        } else {
            base_factors
        };
        let rewritten =
            build_locally_simplified_mul_expr_from_factors_root(ctx, &rewritten_factors);
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            rewritten,
            "Canonizar producto binario firmado con partner polinómico pequeño",
            "Canonical Two-Factor Partner",
            collect_steps,
        ));
    }

    None
}

pub(super) fn try_standard_three_linear_shift_anchor_direct_partner_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 4 {
        return None;
    }

    for i in 0..factors.len() {
        for j in (i + 1)..factors.len() {
            for k in (j + 1)..factors.len() {
                let anchor_subset = [factors[i], factors[j], factors[k]];
                let anchor_subset_expr = build_mul_expr_from_factors_root(ctx, &anchor_subset);
                let Some((base, constants)) =
                    extract_direct_three_linear_shift_product_root(ctx, anchor_subset_expr)
                else {
                    continue;
                };
                let Some(anchor_expanded) =
                    build_direct_three_linear_shift_expanded_target_root(ctx, base, &constants)
                else {
                    continue;
                };

                let remaining_factors = factors
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, factor)| {
                        (index != i && index != j && index != k).then_some(factor)
                    })
                    .collect::<Vec<_>>();
                if remaining_factors.is_empty() {
                    continue;
                }

                let partner_expr = build_mul_expr_from_factors_root(ctx, &remaining_factors);
                let partner_canonical = if let Some(rewritten) =
                    rewrite_direct_double_angle_inverse_trig_target_root(ctx, partner_expr)
                {
                    strip_multiplicative_one_root(ctx, rewritten)
                } else if let Some(rewrite) =
                    try_rewrite_trig_inverse_composition_expr(ctx, partner_expr)
                {
                    strip_multiplicative_one_root(ctx, rewrite.rewritten)
                } else if let Some((lhs_arg, rhs_arg)) =
                    extract_direct_tangent_addition_target_root(ctx, partner_expr)
                {
                    build_tangent_addition_fraction_root(ctx, lhs_arg, rhs_arg)
                } else if expr_contains_sqrt_or_half_power_local(ctx, partner_expr)
                    && cas_ast::count_nodes(ctx, partner_expr) <= 16
                {
                    isolated_simplify_expr_if_changed(
                        &crate::phase::SimplifyOptions::default(),
                        ctx,
                        partner_expr,
                    )
                    .map(|rewritten| strip_multiplicative_one_root(ctx, rewritten))
                    .unwrap_or(partner_expr)
                } else {
                    continue;
                };

                let rewritten =
                    build_mul_expr_from_factors_root(ctx, &[anchor_expanded, partner_canonical]);
                let rewrite = crate::rule::Rewrite::new(rewritten).desc(
                    "Canonizar producto de tres desplazamientos lineales con partner directo pequeño",
                );
                return Some(finish_standard_root_shortcut(
                    ctx,
                    expr,
                    rewrite,
                    "Three Linear Shift Anchor Product",
                    collect_steps,
                ));
            }
        }
    }

    None
}

pub(super) fn try_standard_two_factor_direct_pair_anchor_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    macro_rules! profiled_two_factor_bool {
        ($name:literal, $body:expr) => {{
            if profiling {
                run_profiled_orchestrator_bool_section($name, || $body)
            } else {
                $body
            }
        }};
    }

    fn canonicalize_small_constant_like_direct_pair_anchor_root(
        ctx: &mut Context,
        expr: ExprId,
    ) -> Option<ExprId> {
        let profiling =
            crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();

        let sqrt_function = if profiling {
            run_profiled_root_shortcut(
                "root.mul.14c0.constant_like.sqrt_function_fast_path",
                || try_rewrite_small_constant_sqrt_function_wrapper_root(ctx, expr),
            )
        } else {
            try_rewrite_small_constant_sqrt_function_wrapper_root(ctx, expr)
        };
        if let Some(rewritten) = sqrt_function {
            return Some(rewritten);
        }

        let special_angle = if profiling {
            run_profiled_root_shortcut(
                "root.mul.14c1.constant_like.special_angle_exact_value",
                || extract_special_angle_exact_value_root(ctx, expr),
            )
        } else {
            extract_special_angle_exact_value_root(ctx, expr)
        };
        if let Some(rewritten) = special_angle {
            return Some(strip_multiplicative_one_root(ctx, rewritten));
        }
        let inverse_trig_plan = if profiling {
            run_profiled_root_shortcut("root.mul.14c2.constant_like.inverse_trig_plan", || {
                cas_math::inverse_trig_composition_support::try_plan_inverse_trig_composition_expr(
                    ctx, expr, false, false,
                )
            })
        } else {
            cas_math::inverse_trig_composition_support::try_plan_inverse_trig_composition_expr(
                ctx, expr, false, false,
            )
        };
        if let Some(plan) = inverse_trig_plan {
            return Some(strip_multiplicative_one_root(ctx, plan.rewritten));
        }
        let trig_inverse = if profiling {
            run_profiled_root_shortcut("root.mul.14c3.constant_like.trig_inverse_rewrite", || {
                try_rewrite_trig_inverse_composition_expr(ctx, expr)
            })
        } else {
            try_rewrite_trig_inverse_composition_expr(ctx, expr)
        };
        if let Some(rewrite) = trig_inverse {
            return Some(strip_multiplicative_one_root(ctx, rewrite.rewritten));
        }
        let exp_log_inverse = if profiling {
            run_profiled_root_shortcut("root.mul.14c4.constant_like.exp_log_inverse", || {
                try_rewrite_exponential_log_inverse_expr(ctx, expr)
            })
        } else {
            try_rewrite_exponential_log_inverse_expr(ctx, expr)
        };
        if let Some(rewrite) = exp_log_inverse {
            return Some(strip_multiplicative_one_root(ctx, rewrite.rewritten));
        }
        let log_inverse_match = if profiling {
            run_profiled_root_shortcut("root.mul.14c5.constant_like.log_exp_inverse", || {
                cas_math::logarithm_inverse_support::try_match_log_exp_inverse_expr(ctx, expr)
            })
        } else {
            cas_math::logarithm_inverse_support::try_match_log_exp_inverse_expr(ctx, expr)
        };
        if let Some(log_inverse_match) = log_inverse_match {
            match log_inverse_match {
                cas_math::logarithm_inverse_support::LogExpInverseMatch::Numeric {
                    rewritten,
                    ..
                } => return Some(strip_multiplicative_one_root(ctx, rewritten)),
                cas_math::logarithm_inverse_support::LogExpInverseMatch::Symbolic {
                    base,
                    exponent,
                } => {
                    let e = ctx.add(Expr::Constant(Constant::E));
                    if compare_expr(ctx, base, e) == Ordering::Equal {
                        return Some(strip_multiplicative_one_root(ctx, exponent));
                    }
                }
            }
        }
        if is_stable_small_constant_like_exact_anchor_root(ctx, expr) {
            return None;
        }
        let wrapped_constant_function = if profiling {
            run_profiled_root_shortcut(
                "root.mul.14c5a.constant_like.function_wrapper_exact_eval",
                || try_rewrite_small_constant_function_wrapper_root(ctx, expr),
            )
        } else {
            try_rewrite_small_constant_function_wrapper_root(ctx, expr)
        };
        if let Some(rewritten) = wrapped_constant_function {
            return Some(rewritten);
        }
        let isolated = if profiling {
            let label = constant_like_isolated_simplify_profile_label_root(ctx, expr);
            if label.starts_with("root.mul.14c6") {
                crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                    label,
                    render_expr_for_orchestrator_profile(ctx, expr),
                );
            }
            run_profiled_root_shortcut(label, || {
                isolated_simplify_expr_if_changed(
                    &crate::phase::SimplifyOptions::default(),
                    ctx,
                    expr,
                )
            })
        } else {
            isolated_simplify_expr_if_changed(&isolated_probe_options(), ctx, expr)
        };
        isolated.map(|rewritten| strip_multiplicative_one_root(ctx, rewritten))
    }

    fn is_additive_sin_or_cos_partner_root(ctx: &Context, expr: ExprId) -> bool {
        let Expr::Function(fn_id, args) = ctx.get(expr) else {
            return false;
        };
        args.len() == 1
            && (ctx.is_builtin(*fn_id, BuiltinFn::Sin) || ctx.is_builtin(*fn_id, BuiltinFn::Cos))
            && matches!(ctx.get(args[0]), Expr::Add(_, _) | Expr::Sub(_, _))
    }

    fn should_skip_two_factor_partner_direct_pair_fallback_root(
        ctx: &Context,
        expr: ExprId,
    ) -> bool {
        matches!(ctx.get(expr), Expr::Variable(_) | Expr::Constant(_))
            || is_additive_sin_or_cos_partner_root(ctx, expr)
    }

    fn is_special_angle_exact_value_function_builtin_root(ctx: &Context, expr: ExprId) -> bool {
        let Expr::Function(fn_id, args) = ctx.get(expr) else {
            return false;
        };
        args.len() == 1
            && (ctx.is_builtin(*fn_id, BuiltinFn::Sin)
                || ctx.is_builtin(*fn_id, BuiltinFn::Cos)
                || ctx.is_builtin(*fn_id, BuiltinFn::Tan)
                || ctx.is_builtin(*fn_id, BuiltinFn::Asin)
                || ctx.is_builtin(*fn_id, BuiltinFn::Acos)
                || ctx.is_builtin(*fn_id, BuiltinFn::Atan)
                || ctx.is_builtin(*fn_id, BuiltinFn::Arcsin)
                || ctx.is_builtin(*fn_id, BuiltinFn::Arccos)
                || ctx.is_builtin(*fn_id, BuiltinFn::Arctan))
    }

    fn is_simple_unary_function_partner_root(ctx: &Context, expr: ExprId) -> bool {
        matches!(ctx.get(expr), Expr::Function(_, args) if args.len() == 1)
    }

    fn canonicalize_two_factor_partner_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
        factor_sum_diff_cubes_partner_root(ctx, expr)
            .or_else(|| factor_higher_degree_difference_partner_root(ctx, expr))
            .or_else(|| factor_sophie_germain_partner_root(ctx, expr))
            .or_else(|| factor_known_small_polynomial_partner_root(ctx, expr))
            .or_else(|| {
                if should_skip_two_factor_partner_direct_pair_fallback_root(ctx, expr) {
                    return None;
                }
                if crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled()
                {
                    crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                        "root.mul.14d.two_factor.partner.direct_pair_fallback",
                        render_expr_for_orchestrator_profile(ctx, expr),
                    );
                    run_profiled_root_shortcut(
                        "root.mul.14d.two_factor.partner.direct_pair_fallback",
                        || canonicalize_direct_pair_factor_root(ctx, expr),
                    )
                } else {
                    canonicalize_direct_pair_factor_root(ctx, expr)
                }
            })
    }

    fn canonicalize_two_factor_anchor_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
        if let Some((lhs_arg, rhs_arg)) = extract_direct_tangent_addition_target_root(ctx, expr) {
            return if crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled(
            ) {
                run_profiled_root_shortcut(
                    "root.mul.14a.two_factor.anchor.tangent_addition",
                    || Some(build_tangent_addition_fraction_root(ctx, lhs_arg, rhs_arg)),
                )
            } else {
                Some(build_tangent_addition_fraction_root(ctx, lhs_arg, rhs_arg))
            };
        }
        if is_potential_tanh_ratio_anchor_source_root(ctx, expr) {
            let rewrite =
                if crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled()
                {
                    run_profiled_root_shortcut("root.mul.14b.two_factor.anchor.tanh_ratio", || {
                        try_rewrite_recognize_hyperbolic_from_exp(ctx, expr)
                    })
                } else {
                    try_rewrite_recognize_hyperbolic_from_exp(ctx, expr)
                };
            if let Some(rewrite) = rewrite {
                return Some(rewrite.rewritten);
            }
        }
        if is_small_constant_like_direct_pair_anchor_source_root(ctx, expr) {
            return if crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled(
            ) {
                crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                    "root.mul.14c.two_factor.anchor.direct_pair_fallback",
                    render_expr_for_orchestrator_profile(ctx, expr),
                );
                run_profiled_root_shortcut(
                    "root.mul.14c.two_factor.anchor.direct_pair_fallback",
                    || canonicalize_small_constant_like_direct_pair_anchor_root(ctx, expr),
                )
            } else {
                canonicalize_small_constant_like_direct_pair_anchor_root(ctx, expr)
            };
        }
        if crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                "root.mul.14c.two_factor.anchor.direct_pair_fallback",
                render_expr_for_orchestrator_profile(ctx, expr),
            );
            run_profiled_root_shortcut(
                "root.mul.14c.two_factor.anchor.direct_pair_fallback",
                || canonicalize_direct_pair_factor_root(ctx, expr),
            )
        } else {
            canonicalize_direct_pair_factor_root(ctx, expr)
        }
    }

    fn try_canonicalize_additive_trig_partner_argument_root(
        ctx: &mut Context,
        expr: ExprId,
    ) -> Option<ExprId> {
        let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
            return None;
        };
        if args.len() != 1
            || !(ctx.is_builtin(fn_id, BuiltinFn::Sin) || ctx.is_builtin(fn_id, BuiltinFn::Cos))
        {
            return None;
        }

        let arg = args[0];
        let view = AddView::from_expr(ctx, arg);
        if view.terms.len() != 2 {
            return None;
        }

        let mut pi_term = None;
        let mut other_term = None;
        for (term_expr, term_sign) in view.terms {
            if let Some(k) = extract_rational_pi_multiple(ctx, term_expr) {
                if pi_term.is_some() {
                    return None;
                }
                pi_term = Some((k, term_sign));
            } else {
                if other_term.is_some() || term_sign != Sign::Pos {
                    return None;
                }
                other_term = Some(term_expr);
            }
        }

        let (pi_coeff, pi_sign) = pi_term?;
        let other = other_term?;
        let abs_coeff = pi_coeff.abs();
        let pi = ctx.add(Expr::Constant(Constant::Pi));
        let pi_expr = if abs_coeff == BigRational::one() {
            pi
        } else {
            let coeff = ctx.add(Expr::Number(abs_coeff));
            ctx.add(Expr::Mul(coeff, pi))
        };
        let rewritten_arg = match pi_sign {
            Sign::Pos => ctx.add(Expr::Add(other, pi_expr)),
            Sign::Neg => ctx.add(Expr::Sub(other, pi_expr)),
        };
        let rewritten = ctx.call_builtin(ctx.builtin_of(fn_id)?, vec![rewritten_arg]);
        (compare_expr(ctx, rewritten, expr) != Ordering::Equal).then_some(rewritten)
    }

    let (numeric_coeff, factors) = if profiling {
        run_profiled_root_shortcut("root.mul.14e.two_factor.extract_signed_product", || {
            extract_signed_two_factor_product_root(ctx, expr)
        })?
    } else {
        extract_signed_two_factor_product_root(ctx, expr)?
    };

    for anchor_index in 0..2 {
        let partner_index = 1 - anchor_index;
        if !profiled_two_factor_bool!(
            "root.mul.14g.two_factor.anchor_candidate_gate",
            is_potential_direct_pair_anchor_source_root(ctx, factors[anchor_index])
        ) {
            continue;
        }
        if !profiled_two_factor_bool!(
            "root.mul.14h.two_factor.partner_candidate_gate",
            is_potential_direct_pair_partner_source_root(ctx, factors[partner_index])
        ) {
            continue;
        }
        let function_special_angle_skip =
            profiled_two_factor_bool!("root.mul.14i.two_factor.function_special_angle_skip", {
                if profiling
                    && is_special_angle_exact_value_function_builtin_root(
                        ctx,
                        factors[anchor_index],
                    )
                {
                    crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                        "root.mul.14i.two_factor.function_special_angle_skip",
                        render_expr_for_orchestrator_profile(ctx, factors[anchor_index]),
                    );
                }
                is_special_angle_exact_value_function_builtin_root(ctx, factors[anchor_index])
                    && extract_special_angle_exact_value_root(ctx, factors[anchor_index]).is_some()
            });
        if function_special_angle_skip {
            continue;
        }
        let arithmetic_constant_skip = profiled_two_factor_bool!(
            "root.mul.14j.two_factor.arithmetic_constant_skip",
            is_pure_arithmetic_constant_expr_root(ctx, factors[anchor_index])
                || is_pure_arithmetic_constant_expr_root(ctx, factors[partner_index])
        );
        if arithmetic_constant_skip {
            continue;
        }
        let anchor_canonical = if profiling {
            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                "root.mul.14k.two_factor.anchor_canonicalize",
                render_expr_for_orchestrator_profile(ctx, factors[anchor_index]),
            );
            run_profiled_root_shortcut("root.mul.14k.two_factor.anchor_canonicalize", || {
                canonicalize_two_factor_anchor_root(ctx, factors[anchor_index])
            })
            .unwrap_or(factors[anchor_index])
        } else {
            canonicalize_two_factor_anchor_root(ctx, factors[anchor_index])
                .unwrap_or(factors[anchor_index])
        };
        let anchor_changed =
            compare_expr(ctx, anchor_canonical, factors[anchor_index]) != Ordering::Equal;
        if !profiled_two_factor_bool!(
            "root.mul.14m.two_factor.anchor_safe_target_gate",
            anchor_changed && is_safe_direct_pair_anchor_target_root(ctx, anchor_canonical)
        ) {
            continue;
        }

        let partner_canonical = if profiling {
            run_profiled_root_shortcut("root.mul.14l.two_factor.partner_canonicalize", || {
                canonicalize_two_factor_partner_root(ctx, factors[partner_index])
            })
            .unwrap_or(factors[partner_index])
        } else {
            canonicalize_two_factor_partner_root(ctx, factors[partner_index])
                .unwrap_or(factors[partner_index])
        };
        let stable_constant_anchor =
            is_stable_small_constant_like_exact_anchor_root(ctx, anchor_canonical);
        let partner_pre_simplified = if stable_constant_anchor
            && is_additive_sin_or_cos_partner_root(ctx, partner_canonical)
        {
            let partner_arg_canonical = if profiling {
                run_profiled_root_shortcut(
                    "root.mul.14f1a.two_factor.partner_only_trig_arg_canonicalize",
                    || try_canonicalize_additive_trig_partner_argument_root(ctx, partner_canonical),
                )
            } else {
                try_canonicalize_additive_trig_partner_argument_root(ctx, partner_canonical)
            };
            if partner_arg_canonical.is_some() {
                partner_arg_canonical
            } else if profiling {
                run_profiled_root_shortcut(
                    "root.mul.14f1b.two_factor.partner_only_isolated_simplify_fallback",
                    || isolated_simplify_expr_if_changed(options, ctx, partner_canonical),
                )
            } else {
                isolated_simplify_expr_if_changed(options, ctx, partner_canonical)
            }
        } else {
            None
        };
        let effective_partner = partner_pre_simplified.unwrap_or(partner_canonical);

        let base_factors = if anchor_index == 0 {
            vec![anchor_canonical, effective_partner]
        } else {
            vec![effective_partner, anchor_canonical]
        };
        let rewritten_factors = if numeric_coeff == BigRational::from_integer((-1).into()) {
            let minus_one = ctx.num(-1);
            let mut factors_with_sign = Vec::with_capacity(3);
            factors_with_sign.push(minus_one);
            factors_with_sign.extend(base_factors);
            factors_with_sign
        } else {
            base_factors
        };
        let rewritten_raw =
            build_locally_simplified_mul_expr_from_factors_root(ctx, &rewritten_factors);
        let skip_final_isolated = partner_pre_simplified.is_some()
            || (stable_constant_anchor
                && ((is_function_free_arithmetic_expr_root(ctx, effective_partner)
                    && AddView::from_expr(ctx, effective_partner).terms.len() == 1)
                    || is_simple_unary_function_partner_root(ctx, effective_partner)));
        let final_isolated = if skip_final_isolated {
            if profiling {
                let label = if partner_pre_simplified.is_some() {
                    "root.mul.14f0a.two_factor.skip_after_partner_only_simplify"
                } else if stable_constant_anchor
                    && is_simple_unary_function_partner_root(ctx, effective_partner)
                {
                    "root.mul.14f0b.two_factor.skip_stable_constant_times_unary_function"
                } else {
                    "root.mul.14f0.two_factor.skip_stable_constant_times_nonadditive_arithmetic"
                };
                run_profiled_root_shortcut(label, || Some(rewritten_raw));
            }
            None
        } else if profiling {
            let result = run_profiled_root_shortcut(
                "root.mul.14f.two_factor.final_isolated_simplify",
                || isolated_simplify_expr_if_changed(options, ctx, rewritten_raw),
            );
            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                if result.is_some() {
                    "root.mul.14f.changed.sample"
                } else {
                    "root.mul.14f.unchanged.sample"
                },
                render_expr_for_orchestrator_profile(ctx, rewritten_raw),
            );
            result
        } else {
            isolated_simplify_expr_if_changed(options, ctx, rewritten_raw)
        };
        let rewritten = final_isolated.unwrap_or(rewritten_raw);
        if compare_expr(ctx, rewritten, expr) == Ordering::Equal {
            continue;
        }

        let shortcut_steps = if collect_steps {
            vec![build_root_shortcut_compact_step(
                expr,
                rewritten,
                "Canonizar producto binario con ancla directa y partner equivalente",
                "Direct Pair Anchor Product",
            )]
        } else {
            Vec::new()
        };
        return Some((rewritten, shortcut_steps));
    }

    None
}

pub(super) fn try_standard_safe_anchor_small_polynomial_partner_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if !(2..=3).contains(&factors.len()) {
        return None;
    }

    for partner_index in 0..factors.len() {
        let Some(partner_canonical) =
            factor_known_small_polynomial_partner_root(ctx, factors[partner_index])
        else {
            continue;
        };

        let anchor_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, factor)| (index != partner_index).then_some(factor))
            .collect();
        if anchor_factors
            .iter()
            .copied()
            .any(|factor| extract_special_angle_exact_value_root(ctx, factor).is_some())
        {
            continue;
        }
        let anchor_expr = build_mul_expr_from_factors_root(ctx, &anchor_factors);
        let anchor_canonical = canonicalize_direct_pair_factor_root(ctx, anchor_expr)
            .or_else(|| isolated_simplify_expr_if_changed(options, ctx, anchor_expr))
            .unwrap_or(anchor_expr);
        if !is_safe_direct_pair_anchor_target_root(ctx, anchor_canonical) {
            continue;
        }

        let anchor_changed = compare_expr(ctx, anchor_canonical, anchor_expr) != Ordering::Equal;
        let partner_changed =
            compare_expr(ctx, partner_canonical, factors[partner_index]) != Ordering::Equal;
        if !anchor_changed && !partner_changed {
            continue;
        }

        let rewritten_raw = build_locally_simplified_mul_expr_from_factors_root(
            ctx,
            &[anchor_canonical, partner_canonical],
        );
        let rewritten =
            isolated_simplify_expr_if_changed(options, ctx, rewritten_raw).unwrap_or(rewritten_raw);
        if compare_expr(ctx, rewritten, expr) == Ordering::Equal {
            continue;
        }

        let shortcut_steps = if collect_steps {
            vec![build_root_shortcut_compact_step(
                expr,
                rewritten,
                "Canonizar producto con ancla segura y partner polinómico pequeño",
                "Safe Anchor Small Polynomial Partner",
            )]
        } else {
            Vec::new()
        };
        return Some((rewritten, shortcut_steps));
    }

    None
}

pub(super) fn try_standard_safe_anchor_direct_partner_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if !(2..=3).contains(&factors.len()) {
        return None;
    }

    for partner_index in 0..factors.len() {
        let Some(partner_canonical) =
            canonicalize_safe_anchor_direct_partner_root(ctx, factors[partner_index])
        else {
            continue;
        };

        let anchor_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, factor)| (index != partner_index).then_some(factor))
            .collect();
        if anchor_factors
            .iter()
            .copied()
            .any(|factor| extract_special_angle_exact_value_root(ctx, factor).is_some())
        {
            continue;
        }
        let anchor_expr = build_mul_expr_from_factors_root(ctx, &anchor_factors);
        let anchor_canonical = canonicalize_direct_pair_factor_root(ctx, anchor_expr)
            .or_else(|| isolated_simplify_expr_if_changed(options, ctx, anchor_expr))
            .unwrap_or(anchor_expr);
        if !is_safe_direct_pair_anchor_target_root(ctx, anchor_canonical) {
            continue;
        }

        let anchor_changed = compare_expr(ctx, anchor_canonical, anchor_expr) != Ordering::Equal;
        let partner_changed =
            compare_expr(ctx, partner_canonical, factors[partner_index]) != Ordering::Equal;
        if !anchor_changed && !partner_changed {
            continue;
        }

        let rewritten_raw = build_locally_simplified_mul_expr_from_factors_root(
            ctx,
            &[anchor_canonical, partner_canonical],
        );
        let rewritten =
            isolated_simplify_expr_if_changed(options, ctx, rewritten_raw).unwrap_or(rewritten_raw);
        if compare_expr(ctx, rewritten, expr) == Ordering::Equal {
            continue;
        }

        let shortcut_steps = if collect_steps {
            vec![build_root_shortcut_compact_step(
                expr,
                rewritten,
                "Canonizar producto con ancla segura y partner directo",
                "Safe Anchor Direct Partner",
            )]
        } else {
            Vec::new()
        };
        return Some((rewritten, shortcut_steps));
    }

    None
}

fn is_small_constant_like_direct_pair_anchor_source_root(ctx: &Context, expr: ExprId) -> bool {
    cas_ast::collect_variables(ctx, expr).is_empty() && cas_ast::count_nodes(ctx, expr) <= 16
}

fn is_stable_small_constant_like_exact_anchor_root(ctx: &mut Context, expr: ExprId) -> bool {
    if is_stable_small_direct_surd_container_root(ctx, expr) {
        return true;
    }

    if !is_function_free_arithmetic_expr_root(ctx, expr)
        || !expr_contains_sqrt_or_half_power_local(ctx, expr)
    {
        return false;
    }

    if let Some((_numeric, surd_term, _sign)) =
        cas_math::root_forms::split_numeric_plus_surd(ctx, expr)
    {
        let canonical = canonicalize_small_surd_like_term_root(ctx, surd_term);
        return compare_expr(ctx, canonical, surd_term) == Ordering::Equal;
    }

    match ctx.get(expr).clone() {
        Expr::Mul(_, _) | Expr::Div(_, _) => false,
        _ if is_direct_small_surd_like_term_root(ctx, expr) => false,
        _ => false,
    }
}

fn is_potential_tangent_addition_anchor_source_root(ctx: &Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let mut tan_count = 0usize;
    for (term_expr, term_sign) in view.terms {
        if term_sign != Sign::Pos {
            return false;
        }
        let Expr::Function(fn_id, args) = ctx.get(term_expr) else {
            return false;
        };
        if args.len() != 1 || !ctx.is_builtin(*fn_id, BuiltinFn::Tan) {
            return false;
        }
        tan_count += 1;
    }

    tan_count == 2
}

fn is_potential_direct_pair_partner_source_root(ctx: &Context, expr: ExprId) -> bool {
    if is_function_free_arithmetic_expr_root(ctx, expr) {
        return true;
    }

    match ctx.get(expr) {
        Expr::Neg(inner) => is_potential_direct_pair_partner_source_root(ctx, *inner),
        Expr::Div(_, _) | Expr::Pow(_, _) => true,
        Expr::Mul(_, _) => cas_ast::count_nodes(ctx, expr) <= 16,
        Expr::Function(fn_id, args) => {
            args.len() == 1
                && (ctx.is_builtin(*fn_id, BuiltinFn::Sin)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Cos)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Tan)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Sinh)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Cosh)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Tanh)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Exp)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Ln)
                    || ctx.is_builtin(*fn_id, BuiltinFn::Abs))
        }
        Expr::Add(_, _) | Expr::Sub(_, _) => {
            AddView::from_expr(ctx, expr).terms.len() <= 2 && cas_ast::count_nodes(ctx, expr) <= 16
        }
        Expr::Number(_)
        | Expr::Variable(_)
        | Expr::Constant(_)
        | Expr::Matrix { .. }
        | Expr::SessionRef(_)
        | Expr::Hold(_) => false,
    }
}

fn is_potential_direct_pair_anchor_source_root(ctx: &Context, expr: ExprId) -> bool {
    if is_small_constant_like_direct_pair_anchor_source_root(ctx, expr) {
        return true;
    }

    match ctx.get(expr) {
        Expr::Add(_, _) => is_potential_tangent_addition_anchor_source_root(ctx, expr),
        Expr::Div(_, _) => is_potential_tanh_ratio_anchor_source_root(ctx, expr),
        Expr::Neg(inner) => is_potential_direct_pair_anchor_source_root(ctx, *inner),
        _ => false,
    }
}

pub(super) fn is_supported_nested_direct_equivalence_partner(ctx: &Context, expr: ExprId) -> bool {
    !expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
}

pub(super) fn extract_shared_additive_passthrough_pair_cores_root(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> Option<(ExprId, ExprId)> {
    let lhs_terms = AddView::from_expr(ctx, lhs).terms;
    let rhs_terms = AddView::from_expr(ctx, rhs).terms;
    if lhs_terms.is_empty() || rhs_terms.is_empty() {
        return None;
    }

    let mut lhs_used = vec![false; lhs_terms.len()];
    let mut rhs_used = vec![false; rhs_terms.len()];
    let mut matched_any = false;

    for (lhs_index, (lhs_term, lhs_sign)) in lhs_terms.iter().copied().enumerate() {
        let Some(rhs_index) =
            rhs_terms
                .iter()
                .copied()
                .enumerate()
                .find_map(|(rhs_index, (rhs_term, rhs_sign))| {
                    (!rhs_used[rhs_index]
                        && lhs_sign == rhs_sign
                        && compare_expr(ctx, lhs_term, rhs_term) == Ordering::Equal)
                        .then_some(rhs_index)
                })
        else {
            continue;
        };
        lhs_used[lhs_index] = true;
        rhs_used[rhs_index] = true;
        matched_any = true;
    }

    if !matched_any {
        return None;
    }

    let remaining_lhs_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = lhs_terms
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, term)| (!lhs_used[index]).then_some(term))
        .collect();
    let remaining_rhs_terms: smallvec::SmallVec<[(ExprId, Sign); 8]> = rhs_terms
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, term)| (!rhs_used[index]).then_some(term))
        .collect();

    if remaining_lhs_terms.is_empty() || remaining_rhs_terms.is_empty() {
        return None;
    }

    let lhs_core = AddView {
        root: lhs,
        terms: remaining_lhs_terms,
    }
    .rebuild(ctx);
    let rhs_core = AddView {
        root: rhs,
        terms: remaining_rhs_terms,
    }
    .rebuild(ctx);
    Some((lhs_core, rhs_core))
}

fn term_pair_is_small_exact_equivalent_root(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> bool {
    compare_expr(ctx, lhs, rhs) == Ordering::Equal || matches_known_direct_pair_root(ctx, lhs, rhs)
}

pub(super) fn matches_composed_small_additive_pair_root(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    let lhs_terms = AddView::from_expr(ctx, lhs).terms;
    let rhs_terms = AddView::from_expr(ctx, rhs).terms;
    if !(2..=8).contains(&lhs_terms.len()) || !(2..=8).contains(&rhs_terms.len()) {
        return false;
    }

    let lhs_partitions = build_small_two_chunk_additive_partitions_root(ctx, &lhs_terms);
    let rhs_partitions = build_small_two_chunk_additive_partitions_root(ctx, &rhs_terms);
    for (lhs_a, lhs_b) in lhs_partitions {
        for (rhs_a, rhs_b) in rhs_partitions.iter().copied() {
            if (term_pair_is_small_exact_equivalent_root(ctx, lhs_a, rhs_a)
                && term_pair_is_small_exact_equivalent_root(ctx, lhs_b, rhs_b))
                || (term_pair_is_small_exact_equivalent_root(ctx, lhs_a, rhs_b)
                    && term_pair_is_small_exact_equivalent_root(ctx, lhs_b, rhs_a))
            {
                return true;
            }
        }
    }

    false
}

pub(super) fn try_standard_small_composed_additive_pair_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (lhs, rhs) = match ctx.get(expr) {
        Expr::Sub(lhs, rhs) => (*lhs, *rhs),
        Expr::Add(lhs, rhs) => match ctx.get(*rhs) {
            Expr::Neg(inner) => (*lhs, *inner),
            _ => return None,
        },
        _ => return None,
    };

    if !matches_composed_small_additive_pair_root(ctx, lhs, rhs) {
        return None;
    }

    let zero = ctx.num(0);
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::new(zero).desc("Parallel additive equivalence composition"),
        "Parallel additive equivalence composition",
        collect_steps,
    ))
}

pub(super) fn try_standard_exact_additive_pair_chain_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let allow = cas_solver_core::undefined_risk_policy_support::allow_cancellation_with_undefined_risk_mode_flags(
        matches!(options.shared.semantics.domain_mode, crate::DomainMode::Assume),
        matches!(options.shared.semantics.domain_mode, crate::DomainMode::Strict),
        crate::collect::has_undefined_risk(ctx, expr),
    );
    if !allow {
        return None;
    }

    let rewritten =
        crate::rules::arithmetic::try_rewrite_exact_additive_term_cancellation_expr(ctx, expr)?;
    let rewrite =
        crate::rule::Rewrite::with_local(rewritten, "Cancel exact additive pairs", expr, rewritten);
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        rewrite,
        RULE_CANCEL_EXACT_ADDITIVE_PAIRS,
        collect_steps,
    ))
}

pub(super) fn try_standard_exact_additive_pair_chain_pipeline_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (rewritten, mut shortcut_steps) =
        try_standard_exact_additive_pair_chain_shortcut(options, ctx, expr, collect_steps)?;

    let needs_domain_resimplify =
        exact_additive_pair_chain_result_needs_domain_resimplify(options, ctx, expr, rewritten);
    // A VARIABLE-FREE, non-literal residual (`1/e^0 - 1`, `ln(1)`, `sin(0)`)
    // always takes the re-pass: the cancellation can strand a foldable
    // constant one pass short of the fixpoint (caught by the dsolve O3
    // verification gate), and the shape-list triggers below cannot enumerate
    // every constant head. By construction, not by list: a constant tree is
    // small, the re-pass runs once, and a non-foldable constant (`pi - e`)
    // comes back unchanged.
    let constant_residual_needs_fold = !matches!(ctx.get(rewritten), Expr::Number(_))
        && cas_ast::traversal::collect_variables(ctx, rewritten).is_empty();
    let should_resimplify =
        expr_contains_any_builtin_local(ctx, rewritten, &[BuiltinFn::Atan, BuiltinFn::Arctan])
            || constant_residual_needs_fold
            || matches_direct_small_zero_identity_root(ctx, rewritten)
            || matches_direct_small_zero_or_known_pair_base_root(ctx, rewritten)
            || cas_math::numeric_eval::contains_i(ctx, rewritten)
            || expr_contains_named_function_local(ctx, rewritten, "diff")
            || needs_domain_resimplify;

    if !should_resimplify {
        return Some((rewritten, shortcut_steps));
    }

    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    if needs_domain_resimplify {
        simplifier.set_sticky_implicit_domain(expr, options.shared.semantics.value_domain);
    }
    let (result, inner_steps, _stats) = simplifier.simplify_with_stats(
        rewritten,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..options.clone()
        },
    );
    std::mem::swap(&mut simplifier.context, ctx);

    if collect_steps {
        shortcut_steps.extend(inner_steps);
    }

    Some((result, shortcut_steps))
}

fn exact_additive_pair_chain_result_needs_domain_resimplify(
    options: &crate::phase::SimplifyOptions,
    ctx: &Context,
    original: ExprId,
    rewritten: ExprId,
) -> bool {
    if !expr_contains_any_builtin_local(ctx, rewritten, &[BuiltinFn::Abs, BuiltinFn::Sqrt]) {
        return false;
    }

    let input_domain =
        crate::infer_implicit_domain(ctx, original, options.shared.semantics.value_domain);
    !input_domain.conditions().is_empty()
}

pub(super) fn passthrough_direct_pair_rule_name_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> Option<&'static str> {
    if matches_direct_trig_product_to_sum_sin_sin_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_sin_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_cos_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_pythagorean_identity_pair_root(ctx, lhs_core, rhs_core)
    {
        return Some("Aplicar suma a producto");
    }

    if matches_direct_angle_sum_diff_pair_root(ctx, lhs_core, rhs_core) {
        return Some("Angle Sum/Diff Identity");
    }

    if matches_direct_nested_fraction_simplified_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_sinh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_cosh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_cosh_difference_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_recursive_hyperbolic_sinh_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_recursive_hyperbolic_cosh_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_reciprocal_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_ratio_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_ratio_alias_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_sine_double_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_mixed_double_angle_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_cubic_cosine_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_binomial_square_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_small_pow_expansion_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_pythagorean_identity_pair_root(ctx, lhs_core, rhs_core)
    {
        return Some("Collapse Exact Zero Additive Subexpression");
    }

    if matches_known_direct_pair_root(ctx, lhs_core, rhs_core) {
        return Some("Collapse Exact Zero Additive Subexpression");
    }

    None
}

pub(super) fn try_standard_shared_passthrough_direct_pair_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if matches_direct_product_to_sum_sin_cos_factor_pair_zero_difference_root(ctx, expr) {
        return None;
    }

    let (lhs_core, rhs_core) = extract_shared_additive_passthrough_sub_cores_root(ctx, expr)?;
    let zero = ctx.num(0);
    let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));

    if let Some(rule_name) = passthrough_direct_pair_rule_name_root(ctx, lhs_core, rhs_core) {
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            crate::rule::Rewrite::with_local(zero, rule_name, residual_expr, zero),
            rule_name,
            collect_steps,
        ));
    }

    if matches_direct_nested_zero_trig_ratio_alias_residual_pair_root(ctx, residual_expr) {
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            crate::rule::Rewrite::with_local(
                zero,
                "Collapse Exact Zero Additive Subexpression",
                residual_expr,
                zero,
            ),
            "Collapse Exact Zero Additive Subexpression",
            collect_steps,
        ));
    }

    if expr_contains_any_builtin_local(
        ctx,
        residual_expr,
        &[BuiltinFn::Ln, BuiltinFn::Log, BuiltinFn::Log10],
    ) && try_standard_exact_zero_equivalence_shortcut(options, ctx, residual_expr, false)
        .is_some()
    {
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            zero,
            "Expandir logaritmos y cancelar términos iguales",
            "Expandir logaritmos y cancelar términos iguales",
            collect_steps,
        ));
    }

    if let Some(rule_name) = passthrough_residual_zero_rule_name_root(ctx, residual_expr) {
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            zero,
            rule_name,
            rule_name,
            collect_steps,
        ));
    }

    if try_standard_exact_zero_equivalence_shortcut(options, ctx, residual_expr, false).is_some() {
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            zero,
            "Collapse Exact Zero Additive Subexpression",
            "Collapse Exact Zero Additive Subexpression",
            collect_steps,
        ));
    }

    if expr_contains_division_node_local(ctx, residual_expr)
        && !expr_contains_hyperbolic_builtin_local(ctx, residual_expr)
        && !expr_contains_log_builtin_local(ctx, residual_expr)
        && cas_ast::count_nodes(ctx, residual_expr) <= 24
        && isolated_simplify_rewrites_to_zero(options, ctx, residual_expr)
    {
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            crate::rule::Rewrite::with_local(
                zero,
                "Collapse Exact Zero Additive Subexpression",
                residual_expr,
                zero,
            ),
            "Collapse Exact Zero Additive Subexpression",
            collect_steps,
        ));
    }

    if !expr_contains_trig_or_hyperbolic_builtin_local(ctx, residual_expr)
        && cas_ast::count_nodes(ctx, residual_expr) <= 64
        && isolated_simplify_rewrites_to_zero(options, ctx, residual_expr)
    {
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            zero,
            "Collapse Exact Zero Additive Subexpression",
            "Collapse Exact Zero Additive Subexpression",
            collect_steps,
        ));
    }

    None
}
