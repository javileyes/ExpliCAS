//! `focused_rule_substeps`: familia `fractions`.
//!
//! Ver la cabecera de `focused_rule_substeps.rs` para el contexto.

use super::*;

pub(super) fn generate_fraction_expansion_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let local_after = step.after_local().unwrap_or(step.after);
    let mut out = Vec::new();
    if before != local_after {
        let first_title = match ctx.get(before) {
            Expr::Div(numerator, _denominator) => {
                let numerator_terms = AddView::from_expr(ctx, *numerator);
                if numerator_terms.terms.len() >= 3 {
                    "Repartir el mismo denominador sobre cada término del numerador"
                } else {
                    "Repartir el denominador entre los términos del numerador"
                }
            }
            _ => "Repartir el denominador entre los términos del numerador",
        };
        out.push(concrete_expr_substep(ctx, first_title, before, local_after));
    }

    if step.before_local().is_none() {
        if let Some(intermediate) = step.after_local().or_else(|| {
            let mut work = ctx.clone();
            build_fraction_expansion_intermediate(&mut work, before)
        }) {
            if intermediate != step.after {
                out.push(
                    SubStep::new(
                        fraction_expansion_cleanup_title(ctx, intermediate),
                        human_expr(ctx, intermediate),
                        human_expr(ctx, step.after),
                    )
                    .with_before_latex(latex_expr(ctx, intermediate))
                    .with_after_latex(latex_expr(ctx, step.after)),
                );
            }
        }
    }

    out
}

fn build_fraction_expansion_intermediate(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
        return None;
    };

    let terms = AddView::from_expr(ctx, numerator).terms;
    if terms.len() < 2 {
        return None;
    }

    let distributed_terms = terms
        .into_iter()
        .map(|(term, sign)| (ctx.add(Expr::Div(term, denominator)), sign))
        .collect::<Vec<_>>();

    Some(build_add_from_signed_terms(ctx, &distributed_terms))
}

pub(super) fn is_fraction_expansion_simplify_pair(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> bool {
    let mut work = ctx.clone();
    let Some(intermediate) = build_fraction_expansion_intermediate(&mut work, before) else {
        return false;
    };
    let simplified = simplify_expr_in_context(&mut work, intermediate);
    compare_expr(&work, simplified, after) == Ordering::Equal
        || human_expr(&work, simplified) == human_expr(ctx, after)
}

pub(super) fn generate_add_subtract_fractions_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let unit_numerators = both_fraction_numerators_are_one(ctx, before);

    let mut work = ctx.clone();
    let Some(intermediate) = build_two_fraction_common_denominator_intermediate(&mut work, before)
    else {
        return Vec::new();
    };

    let intermediate_display = display_expr(&work, intermediate);
    let after_display = display_expr(ctx, after);
    let intermediate_latex = latex_expr(&work, intermediate);
    let after_latex = latex_expr(ctx, after);

    let mut out = vec![SubStep::keyed(
        "fraction.common_denominator",
        vec![],
        display_expr(ctx, before),
        intermediate_display.clone(),
    )
    .with_before_latex(latex_expr(ctx, before))
    .with_after_latex(intermediate_latex.clone())];

    if unit_numerators
        && (intermediate_display != after_display || intermediate_latex != after_latex)
    {
        out.push(
            SubStep::keyed(
                "fraction.simplify_numerator_and_denominator",
                vec![],
                intermediate_display,
                after_display,
            )
            .with_before_latex(intermediate_latex)
            .with_after_latex(after_latex),
        );
    }

    out
}

fn both_fraction_numerators_are_one(ctx: &Context, expr: ExprId) -> bool {
    let Some((left, right, _is_subtraction)) = extract_fraction_add_sub_operands(ctx, expr) else {
        return false;
    };

    let Some((left_num, _)) = as_div(ctx, left) else {
        return false;
    };
    let Some((right_num, _)) = as_div(ctx, right) else {
        return false;
    };

    is_one(ctx, left_num) && is_one(ctx, right_num)
}

pub(super) fn build_two_fraction_common_denominator_intermediate(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (left, right, is_subtraction) = extract_fraction_add_sub_operands(ctx, expr)?;

    let (left_num, left_den) = as_div(ctx, left)?;
    let (right_num, right_den) = as_div(ctx, right)?;

    let common_den = ctx.add(Expr::Mul(left_den, right_den));
    // The `Mul(1, den)` lifts are DELIBERATE and load-bearing: an opaque Mul
    // term blocks both node-level canonicalization AND the display layer's
    // term reordering, which is what keeps this didactic intermediate shaped
    // like the story it narrates (`x + 1 + x - 1` over the common
    // denominator) instead of folding into the step's own after and gating
    // the bridge sub-step off as a no-op. A cleaner-looking unit-free lift
    // was tried and measured: it DELETED the narration of every
    // unit-numerator Add pair in the corpus. The `2x` lie this shape once
    // caused lived in the renderers — the unit-elision-vs-parenthesization
    // mismatch — and is fixed there, where the class belongs.
    let lifted_left = ctx.add(Expr::Mul(left_num, right_den));
    let lifted_right = ctx.add(Expr::Mul(right_num, left_den));
    let numerator = if is_subtraction {
        ctx.add(Expr::Sub(lifted_left, lifted_right))
    } else {
        ctx.add(Expr::Add(lifted_left, lifted_right))
    };

    Some(ctx.add(Expr::Div(numerator, common_den)))
}

pub(super) fn extract_fraction_add_sub_operands(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, bool)> {
    match ctx.get(expr) {
        Expr::Add(left, right) => match ctx.get(*right) {
            Expr::Neg(inner) => Some((*left, *inner, true)),
            _ => Some((*left, *right, false)),
        },
        Expr::Sub(left, right) => Some((*left, *right, true)),
        _ => None,
    }
}

fn fraction_expansion_cleanup_title(ctx: &Context, intermediate: ExprId) -> String {
    match count_fraction_terms_with_common_factor(ctx, intermediate) {
        0 => "Simplificar las fracciones resultantes".to_string(),
        1 => "Cancelar los factores comunes en la fracción que queda".to_string(),
        _ => "Cancelar los factores comunes en las fracciones resultantes".to_string(),
    }
}

fn count_fraction_terms_with_common_factor(ctx: &Context, expr: ExprId) -> usize {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() <= 1 {
        return usize::from(fraction_term_has_common_factor(ctx, expr));
    }

    terms
        .into_iter()
        .filter(|(term, _sign)| fraction_term_has_common_factor(ctx, *term))
        .count()
}

fn fraction_term_has_common_factor(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Div(numerator, denominator) => {
            first_common_factor(ctx, *numerator, *denominator).is_some()
        }
        _ => false,
    }
}

pub(super) fn generate_mixed_fraction_split_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if before == after {
        return Vec::new();
    }

    generate_mixed_fraction_split_intermediate_substeps(ctx, before, after).unwrap_or_default()
}

fn generate_mixed_fraction_split_intermediate_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<Vec<SubStep>> {
    let Expr::Div(_, denominator) = ctx.get(before) else {
        return None;
    };
    let (whole, whole_sign, remainder, remainder_sign, remainder_denominator) =
        mixed_fraction_split_parts(ctx, after, *denominator)?;
    if !same_expr(ctx, *denominator, remainder_denominator) {
        return None;
    }

    let mut work = ctx.clone();
    let whole_term = signed_expr(&mut work, whole, whole_sign);
    let product = work.add_raw(Expr::Mul(whole_term, *denominator));
    let intermediate_numerator = if remainder_sign == Sign::Pos {
        work.add_raw(Expr::Add(product, remainder))
    } else {
        work.add_raw(Expr::Sub(product, remainder))
    };
    let intermediate = work.add_raw(Expr::Div(intermediate_numerator, *denominator));
    Some(vec![
        mixed_ctx_substep(
            "Reescribir el numerador como parte entera por denominador más resto",
            ctx,
            before,
            &work,
            intermediate,
        ),
        temp_ctx_substep(
            "Separar la suma del numerador sobre el denominador",
            &work,
            intermediate,
            after,
        ),
    ])
}

fn mixed_fraction_split_parts(
    ctx: &Context,
    after: ExprId,
    source_denominator: ExprId,
) -> Option<(ExprId, Sign, ExprId, Sign, ExprId)> {
    let terms = AddView::from_expr(ctx, after).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut whole = None;
    let mut remainder = None;
    for (term, sign) in terms {
        if let Some((numerator, denominator)) = as_div(ctx, term) {
            if same_expr(ctx, denominator, source_denominator) {
                if remainder.replace((numerator, sign, denominator)).is_some() {
                    return None;
                }
                continue;
            }
        }

        if whole.replace((term, sign)).is_some() {
            return None;
        }
    }

    let (whole, whole_sign) = whole?;
    let (remainder, remainder_sign, denominator) = remainder?;
    Some((whole, whole_sign, remainder, remainder_sign, denominator))
}

pub(super) fn generate_mixed_fraction_combine_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if before == after {
        return Vec::new();
    }

    vec![concrete_expr_substep(
        ctx,
        "Poner la parte entera sobre el mismo denominador y combinar",
        before,
        after,
    )]
}

pub(super) fn generate_same_base_power_quotient_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Vec<SubStep> {
    let Some((numerator, denominator)) = as_div(ctx, before) else {
        return Vec::new();
    };
    let Some((numerator_base, numerator_exponent)) = as_pow(ctx, numerator) else {
        return Vec::new();
    };
    let Some((denominator_base, denominator_exponent)) = as_pow(ctx, denominator) else {
        return Vec::new();
    };
    if compare_expr(ctx, numerator_base, denominator_base) != Ordering::Equal {
        return Vec::new();
    }

    let mut work = ctx.clone();
    let negative_denominator_exponent = work.add(Expr::Neg(denominator_exponent));
    let denominator_as_negative_power =
        work.add(Expr::Pow(numerator_base, negative_denominator_exponent));
    let numerator_power = work.add(Expr::Pow(numerator_base, numerator_exponent));
    let intermediate = work.add(Expr::Mul(numerator_power, denominator_as_negative_power));
    let merged_exponent = work.add(Expr::Sub(numerator_exponent, denominator_exponent));
    let expected_after = work.add(Expr::Pow(numerator_base, merged_exponent));
    if compare_expr(&work, expected_after, after) != Ordering::Equal {
        return Vec::new();
    }

    vec![
        temp_ctx_substep(
            "Reescribir la división como potencia negativa",
            &work,
            before,
            intermediate,
        ),
        temp_ctx_substep(
            "Sumar los exponentes de la misma base",
            &work,
            intermediate,
            after,
        ),
    ]
}

pub(super) fn generate_reverse_nested_fraction_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<Vec<SubStep>> {
    let pattern = reverse_nested_fraction_pattern(ctx, after)?;

    match pattern {
        NestedFractionPattern::OneOverSumWithFraction
        | NestedFractionPattern::FractionOverSumWithFraction => {
            let Expr::Div(_, before_den) = ctx.get(before) else {
                return None;
            };
            let Expr::Div(_, after_den) = ctx.get(after) else {
                return None;
            };
            let (_, common_den) = split_add_with_single_fraction(ctx, *after_den)?;
            if !factors_exactly(ctx, *before_den, common_den, *after_den) {
                return None;
            }
            let common_den_display = human_expr(ctx, common_den);
            let common_den_grouped_display = grouped_substitution_display(ctx, common_den);
            let common_den_grouped_latex = grouped_substitution_latex(ctx, common_den);
            let after_den_display = human_expr(ctx, *after_den);
            let after_den_latex = latex_expr(ctx, *after_den);

            return Some(vec![SubStep::keyed(
                "nested.rewrite_denominator_common_factor",
                vec![format!("{common_den_display}")],
                human_expr(ctx, *before_den),
                format!("{common_den_grouped_display} · ({after_den_display})"),
            )
            .with_before_latex(latex_expr(ctx, *before_den))
            .with_after_latex(format!(
                "{common_den_grouped_latex}\\cdot \\left({after_den_latex}\\right)"
            ))]);
        }
        NestedFractionPattern::SumWithFractionOverScalar => {
            let Expr::Div(before_num, _) = ctx.get(before) else {
                return None;
            };
            let Expr::Div(after_num, _) = ctx.get(after) else {
                return None;
            };
            let (_, common_den) = split_add_with_single_fraction(ctx, *after_num)?;
            if !factors_exactly(ctx, *before_num, common_den, *after_num) {
                return None;
            }
            let common_den_display = human_expr(ctx, common_den);
            let common_den_grouped_display = grouped_substitution_display(ctx, common_den);
            let common_den_grouped_latex = grouped_substitution_latex(ctx, common_den);
            let after_num_display = human_expr(ctx, *after_num);
            let after_num_latex = latex_expr(ctx, *after_num);

            return Some(vec![SubStep::keyed(
                "nested.rewrite_numerator_common_factor",
                vec![format!("{common_den_display}")],
                human_expr(ctx, *before_num),
                format!("{common_den_grouped_display} · ({after_num_display})"),
            )
            .with_before_latex(latex_expr(ctx, *before_num))
            .with_after_latex(format!(
                "{common_den_grouped_latex}\\cdot \\left({after_num_latex}\\right)"
            ))]);
        }
        NestedFractionPattern::OneOverSumWithUnitFraction | NestedFractionPattern::General => {}
    }

    None
}

pub(super) fn generate_reverse_nested_fraction_rule_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.global_before.unwrap_or(step.before);
    let after = step.global_after.unwrap_or(step.after);

    generate_reverse_nested_fraction_substeps(ctx, before, after)
        .or_else(|| {
            let before = step.before_local().unwrap_or(step.before);
            let after = step.after_local().unwrap_or(step.after);
            generate_reverse_nested_fraction_substeps(ctx, before, after)
        })
        .unwrap_or_default()
}

fn reverse_nested_fraction_pattern(ctx: &Context, after: ExprId) -> Option<NestedFractionPattern> {
    let pattern = super::super::nested_fraction_analysis::classify_nested_fraction(ctx, after)?;
    match pattern {
        NestedFractionPattern::OneOverSumWithFraction
        | NestedFractionPattern::FractionOverSumWithFraction
        | NestedFractionPattern::SumWithFractionOverScalar => {}
        _ => return None,
    }

    let Expr::Div(num, den) = ctx.get(after) else {
        return None;
    };

    match pattern {
        NestedFractionPattern::OneOverSumWithFraction
        | NestedFractionPattern::FractionOverSumWithFraction => {
            let denominator = *den;
            let _ = split_add_with_single_fraction(ctx, denominator)?;
        }
        NestedFractionPattern::SumWithFractionOverScalar => {
            let numerator = *num;
            let _ = split_add_with_single_fraction(ctx, numerator)?;
        }
        NestedFractionPattern::OneOverSumWithUnitFraction | NestedFractionPattern::General => {
            return None
        }
    }

    Some(pattern)
}

fn split_add_with_single_fraction(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };

    match (ctx.get(*left), ctx.get(*right)) {
        (Expr::Div(_, left_den), _) if !matches!(ctx.get(*right), Expr::Div(_, _)) => {
            Some((*left, *left_den))
        }
        (_, Expr::Div(_, right_den)) if !matches!(ctx.get(*left), Expr::Div(_, _)) => {
            Some((*right, *right_den))
        }
        _ => None,
    }
}

pub(super) fn generate_telescoping_fraction_combine_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Some((u, _gap_display, gap_is_one)) = telescoping_fraction_base_and_gap(ctx, after, before)
    else {
        return Vec::new();
    };
    let _ = u;

    if gap_is_one {
        return generate_consecutive_telescoping_common_denominator_substeps(
            ctx,
            after,
            before,
            TelescopingFractionSubstepDirection::Combine,
        )
        .unwrap_or_default();
    }

    generate_general_telescoping_common_denominator_substeps(
        ctx,
        after,
        before,
        TelescopingFractionSubstepDirection::Combine,
    )
    .unwrap_or_default()
}

pub(super) fn generate_telescoping_fraction_split_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    generate_consecutive_telescoping_fraction_substeps(ctx, before, after).unwrap_or_default()
}

pub(super) fn generate_consecutive_telescoping_fraction_substeps(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<Vec<SubStep>> {
    let (u, _gap_display, gap_is_one) = telescoping_fraction_base_and_gap(ctx, before, after)?;
    let _ = u;

    if gap_is_one {
        return generate_consecutive_telescoping_common_denominator_substeps(
            ctx,
            before,
            after,
            TelescopingFractionSubstepDirection::Split,
        );
    }

    generate_general_telescoping_common_denominator_substeps(
        ctx,
        before,
        after,
        TelescopingFractionSubstepDirection::Split,
    )
}

fn generate_consecutive_telescoping_common_denominator_substeps(
    ctx: &Context,
    compact_expr: ExprId,
    split_expr: ExprId,
    direction: TelescopingFractionSubstepDirection,
) -> Option<Vec<SubStep>> {
    let (work, common_fraction) =
        consecutive_telescoping_common_fraction_work(ctx, compact_expr, split_expr)?;

    Some(match direction {
        TelescopingFractionSubstepDirection::Split => vec![
            temp_ctx_substep(
                "Introducir el numerador telescópico",
                &work,
                compact_expr,
                common_fraction,
            ),
            temp_ctx_substep(
                "Separar sobre el denominador común",
                &work,
                common_fraction,
                split_expr,
            ),
        ],
        TelescopingFractionSubstepDirection::Combine => vec![
            temp_ctx_substep(
                "Llevar las fracciones al denominador común",
                &work,
                split_expr,
                common_fraction,
            ),
            temp_ctx_substep(
                "Simplificar el numerador telescópico",
                &work,
                common_fraction,
                compact_expr,
            ),
        ],
    })
}

fn consecutive_telescoping_common_fraction_work(
    ctx: &Context,
    compact_expr: ExprId,
    split_expr: ExprId,
) -> Option<(Context, ExprId)> {
    let (num, den) = as_div(ctx, compact_expr)?;
    if !is_one(ctx, num) {
        return None;
    }

    let (u, u_plus_gap, gap_expr) = extract_telescoping_fraction_split_pattern(ctx, split_expr)?;
    if gap_expr.is_some() || !unit_gap_relation_holds(ctx, u, u_plus_gap) {
        return None;
    }
    if !matches_telescoping_fraction_denominator(ctx, den, u, u_plus_gap) {
        return None;
    }

    let mut work = ctx.clone();
    let numerator_difference = work.add(Expr::Sub(u_plus_gap, u));
    let common_fraction = work.add(Expr::Div(numerator_difference, den));
    Some((work, common_fraction))
}

fn generate_general_telescoping_common_denominator_substeps(
    ctx: &Context,
    compact_expr: ExprId,
    split_expr: ExprId,
    direction: TelescopingFractionSubstepDirection,
) -> Option<Vec<SubStep>> {
    let (work, common_fraction) =
        general_telescoping_common_fraction_work(ctx, compact_expr, split_expr)?;

    Some(match direction {
        TelescopingFractionSubstepDirection::Split => vec![
            temp_ctx_substep(
                "Introducir el numerador telescópico",
                &work,
                compact_expr,
                common_fraction,
            ),
            temp_ctx_substep(
                "Separar sobre el denominador común",
                &work,
                common_fraction,
                split_expr,
            ),
        ],
        TelescopingFractionSubstepDirection::Combine => vec![
            temp_ctx_substep(
                "Llevar las fracciones al denominador común",
                &work,
                split_expr,
                common_fraction,
            ),
            temp_ctx_substep(
                "Simplificar el numerador telescópico",
                &work,
                common_fraction,
                compact_expr,
            ),
        ],
    })
}

fn general_telescoping_common_fraction_work(
    ctx: &Context,
    compact_expr: ExprId,
    split_expr: ExprId,
) -> Option<(Context, ExprId)> {
    let (num, den) = as_div(ctx, compact_expr)?;
    if !is_one(ctx, num) {
        return None;
    }

    let (u, u_plus_gap, gap_expr) = extract_telescoping_fraction_split_pattern(ctx, split_expr)?;
    let gap_expr = gap_expr?;
    if !additive_gap_relation_holds(ctx, u, gap_expr, u_plus_gap) {
        return None;
    }
    if !matches_telescoping_fraction_denominator(ctx, den, u, u_plus_gap) {
        return None;
    }

    let mut work = ctx.clone();
    let numerator_difference = work.add(Expr::Sub(u_plus_gap, u));
    let scaled_denominator = work.add(Expr::Mul(gap_expr, den));
    let common_fraction = work.add(Expr::Div(numerator_difference, scaled_denominator));
    Some((work, common_fraction))
}

fn telescoping_fraction_base_and_gap(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Option<(ExprId, String, bool)> {
    let (num, den) = as_div(ctx, before)?;
    if !is_one(ctx, num) {
        return None;
    }

    let (u, u_plus_gap, gap_expr) = extract_telescoping_fraction_split_pattern(ctx, after)?;
    if !matches_telescoping_fraction_denominator(ctx, den, u, u_plus_gap) {
        return None;
    }

    if let Some(gap_expr) = gap_expr {
        if !additive_gap_relation_holds(ctx, u, gap_expr, u_plus_gap) {
            return None;
        }
        Some((u, human_expr(ctx, gap_expr), false))
    } else {
        if !unit_gap_relation_holds(ctx, u, u_plus_gap) {
            return None;
        }
        Some((u, "1".to_string(), true))
    }
}

fn matches_telescoping_fraction_denominator(
    ctx: &Context,
    denominator: ExprId,
    u: ExprId,
    u_plus_gap: ExprId,
) -> bool {
    let factors = expr_nary::mul_leaves(ctx, denominator);
    if factors.len() == 2 {
        let same_order = cas_ast::ordering::compare_expr(ctx, factors[0], u)
            == std::cmp::Ordering::Equal
            && cas_ast::ordering::compare_expr(ctx, factors[1], u_plus_gap)
                == std::cmp::Ordering::Equal;
        let swapped_order = cas_ast::ordering::compare_expr(ctx, factors[1], u)
            == std::cmp::Ordering::Equal
            && cas_ast::ordering::compare_expr(ctx, factors[0], u_plus_gap)
                == std::cmp::Ordering::Equal;
        if same_order || swapped_order {
            return true;
        }
    }

    // Also accept denominators that are algebraically the same product even when
    // they are still expanded, like x^2 + 3x + 2 instead of (x + 1)(x + 2).
    let mut temp_ctx = ctx.clone();
    let expected_product = temp_ctx.add_raw(Expr::Mul(u, u_plus_gap));
    poly_eq(&temp_ctx, denominator, expected_product)
}

fn extract_telescoping_fraction_split_pattern(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, Option<ExprId>)> {
    if let Some((numerator, denominator)) = as_div(ctx, expr) {
        if let Some((u, u_plus_gap)) = extract_telescoping_fraction_core(ctx, numerator) {
            return Some((u, u_plus_gap, Some(denominator)));
        }
    }

    let factors = expr_nary::mul_leaves(ctx, expr);
    for core_index in 0..factors.len() {
        let Some((u, u_plus_gap)) = extract_telescoping_fraction_core(ctx, factors[core_index])
        else {
            continue;
        };

        let residual = factors
            .iter()
            .enumerate()
            .filter_map(|(index, factor)| (index != core_index).then_some(*factor))
            .collect::<Vec<_>>();
        match residual.as_slice() {
            [] => return Some((u, u_plus_gap, None)),
            [single] => {
                if let Some(denominator) = extract_unit_reciprocal_denominator(ctx, *single) {
                    return Some((u, u_plus_gap, Some(denominator)));
                }
            }
            _ => {}
        }
    }

    None
}

fn extract_telescoping_fraction_core(ctx: &Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut saw_u = None;
    let mut saw_u_plus_gap = None;
    for (term, sign) in terms {
        match sign {
            Sign::Pos => saw_u = Some(extract_unit_fraction_denominator(ctx, term)?),
            Sign::Neg => saw_u_plus_gap = Some(extract_unit_fraction_denominator(ctx, term)?),
        }
    }

    Some((saw_u?, saw_u_plus_gap?))
}

fn extract_unit_fraction_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let (num, den) = as_div(ctx, expr)?;
    is_one(ctx, num).then_some(den)
}

fn extract_unit_reciprocal_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let (num, den) = as_div(ctx, expr)?;
    is_one(ctx, num).then_some(den)
}

pub(super) fn reciprocal_rewrite_substep_for_factor(
    ctx: &mut Context,
    factor: ExprId,
) -> Option<(String, ExprId)> {
    let (builtin, arg) = match ctx.get(factor) {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(*fn_id), args[0]),
        _ => return None,
    };

    if matches!(builtin, Some(BuiltinFn::Sec)) {
        let cos_expr = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
        let one = ctx.num(1);
        let before = ctx.add(Expr::Div(one, cos_expr));
        return Some(("Usar 1 / cos(u) = sec(u)".to_string(), before));
    }

    if matches!(builtin, Some(BuiltinFn::Csc)) {
        let sin_expr = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
        let one = ctx.num(1);
        let before = ctx.add(Expr::Div(one, sin_expr));
        return Some(("Usar 1 / sin(u) = csc(u)".to_string(), before));
    }

    None
}

pub(super) fn generate_reciprocal_product_identity_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    // C1.8: the audit's first named lie — this title used to publish over ANY
    // pair. Now the pair must be tan(σu)·cot(σu) ⇒ 1 for some σ, or nothing
    // is published.
    named_identity_substep(ctx, "tan(u) · cot(u)", "1", before, after)
        .into_iter()
        .collect()
}

/// Migrated to the matcher (2026-07-27): the derive route maps TEN
/// description arms onto this rule (sec²−tan²⟹1, sec²−1⟹tan², the negated
/// 1−sec²⟹−tan² family, their csc twins and expansions) and the old emitter
/// only spoke for two of them. One row per identity; the matcher tries both
/// application orientations, so the expansion arms cite the same censused
/// row they invert. The six are structurally disjoint — no directed-mode
/// ambiguity between equivalents.
pub(super) fn generate_reciprocal_pythagorean_substeps(ctx: &Context, step: &Step) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    const RECIPROCAL_PYTHAGOREAN_TEMPLATES: [(&str, &str); 6] = [
        ("sec(u)^2 - tan(u)^2", "1"),
        ("csc(u)^2 - cot(u)^2", "1"),
        ("sec(u)^2 - 1", "tan(u)^2"),
        ("csc(u)^2 - 1", "cot(u)^2"),
        ("1 - sec(u)^2", "-tan(u)^2"),
        ("1 - csc(u)^2", "-cot(u)^2"),
    ];
    named_identity_from_table(ctx, &RECIPROCAL_PYTHAGOREAN_TEMPLATES, before, after)
        .into_iter()
        .collect()
}

pub(super) fn render_fraction_plain(numerator: &str, denominator: &str) -> String {
    format!("({numerator}) / ({denominator})")
}

pub(super) fn render_fraction_latex(numerator: &str, denominator: &str) -> String {
    format!("\\frac{{{numerator}}}{{{denominator}}}")
}

pub(super) fn render_unit_fraction_plain(denominator: &str) -> String {
    render_fraction_plain("1", denominator)
}

pub(super) fn render_unit_fraction_latex(denominator: &str) -> String {
    render_fraction_latex("1", denominator)
}

pub(super) fn generate_simplify_nested_fraction_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let repeated_factor = generate_perfect_square_fraction_cancel_substeps(ctx, step);
    if !repeated_factor.is_empty() {
        return repeated_factor;
    }

    let common_factor = generate_common_factor_cancel_substeps(ctx, step);
    if !common_factor.is_empty() {
        return common_factor;
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    if let Some(substeps) = generate_reverse_nested_fraction_substeps(ctx, before, after) {
        return substeps;
    }

    let Expr::Div(_, _) = ctx.get(before) else {
        return Vec::new();
    };

    {
        let mut work = ctx.clone();
        if let Some((den_before, den_after, full_intermediate)) =
            build_one_over_fraction_plus_minus_one_intermediates(&mut work, before)
        {
            return vec![
                SubStep::keyed(
                    "polynomial.common_denominator_within_denominator",
                    vec![],
                    display_expr(&work, den_before),
                    display_expr(&work, den_after),
                )
                .with_before_latex(latex_expr(&work, den_before))
                .with_after_latex(latex_expr(&work, den_after)),
                SubStep::keyed(
                    "polynomial.invert_denominator_fraction",
                    vec![],
                    display_expr(&work, full_intermediate),
                    display_expr(ctx, after),
                )
                .with_before_latex(latex_expr(&work, full_intermediate))
                .with_after_latex(latex_expr(ctx, after)),
            ];
        }
    }

    {
        let mut work = ctx.clone();
        if let Some(intermediate) =
            build_sum_difference_reciprocal_complex_fraction_intermediate(&mut work, before)
        {
            return vec![
                SubStep::new(
                    "Llevar el numerador y el denominador a común denominador",
                    display_expr(ctx, before),
                    display_expr(&work, intermediate),
                )
                .with_before_latex(latex_expr(ctx, before))
                .with_after_latex(latex_expr(&work, intermediate)),
                SubStep::new(
                    "Cancelar el denominador común de numerador y denominador",
                    display_expr(&work, intermediate),
                    display_expr(ctx, after),
                )
                .with_before_latex(latex_expr(&work, intermediate))
                .with_after_latex(latex_expr(ctx, after)),
            ];
        }
    }

    {
        let mut work = ctx.clone();
        if let Some((numerator_before, numerator_after, full_intermediate, divides_by_fraction)) =
            build_additive_numerator_nested_fraction_intermediates(&mut work, before)
        {
            let final_title = if divides_by_fraction {
                "Dividir entre una fracción es multiplicar por su inversa"
            } else {
                "Incorporar el denominador externo"
            };

            return vec![
                SubStep::new(
                    "Llevar el numerador a denominador común",
                    display_expr(&work, numerator_before),
                    display_expr(&work, numerator_after),
                )
                .with_before_latex(latex_expr(&work, numerator_before))
                .with_after_latex(latex_expr(&work, numerator_after)),
                SubStep::new(
                    final_title,
                    display_expr(&work, full_intermediate),
                    display_expr(ctx, after),
                )
                .with_before_latex(latex_expr(&work, full_intermediate))
                .with_after_latex(latex_expr(ctx, after)),
            ];
        }
    }

    vec![SubStep::new(
        "Cancelar los factores comunes del numerador y del denominador",
        display_expr(ctx, before),
        display_expr(ctx, after),
    )
    .with_before_latex(latex_expr(ctx, before))
    .with_after_latex(latex_expr(ctx, after))]
}

fn unit_fraction_denominator(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Div(num, den) if is_one(ctx, *num) => Some(*den),
        _ => None,
    }
}

fn build_reciprocal_pair_with_common_denominator(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            let left_den = unit_fraction_denominator(ctx, *left)?;
            let right_den = unit_fraction_denominator(ctx, *right)?;
            let common_den = build_balanced_mul(ctx, &[left_den, right_den]);
            let numerator = ctx.add(Expr::Add(right_den, left_den));
            Some(ctx.add(Expr::Div(numerator, common_den)))
        }
        Expr::Sub(left, right) => {
            let left_den = unit_fraction_denominator(ctx, *left)?;
            let right_den = unit_fraction_denominator(ctx, *right)?;
            let common_den = build_balanced_mul(ctx, &[left_den, right_den]);
            let numerator = ctx.add(Expr::Sub(right_den, left_den));
            Some(ctx.add(Expr::Div(numerator, common_den)))
        }
        _ => None,
    }
}

fn split_single_fraction_addend(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    if let Some((fraction_num, fraction_den)) = as_div(ctx, left) {
        return Some((fraction_num, fraction_den, right));
    }

    let (fraction_num, fraction_den) = as_div(ctx, right)?;
    Some((fraction_num, fraction_den, left))
}

fn build_additive_single_fraction_common_denominator(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(expr).clone() else {
        return None;
    };
    let (fraction_num, fraction_den, other) = split_single_fraction_addend(ctx, left, right)?;
    let scaled_other = ctx.add(Expr::Mul(other, fraction_den));
    let numerator = ctx.add(Expr::Add(scaled_other, fraction_num));
    Some(ctx.add(Expr::Div(numerator, fraction_den)))
}

fn build_additive_numerator_nested_fraction_intermediates(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId, bool)> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    let numerator_after = build_reciprocal_pair_with_common_denominator(ctx, numerator)
        .or_else(|| build_additive_single_fraction_common_denominator(ctx, numerator))?;
    let full_intermediate = ctx.add(Expr::Div(numerator_after, denominator));
    Some((
        numerator,
        numerator_after,
        full_intermediate,
        unit_fraction_denominator(ctx, denominator).is_some(),
    ))
}

fn build_sum_difference_reciprocal_complex_fraction_intermediate(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    let numerator = build_reciprocal_pair_with_common_denominator(ctx, numerator)?;
    let denominator = build_reciprocal_pair_with_common_denominator(ctx, denominator)?;
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn build_one_over_fraction_plus_minus_one_intermediates(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId)> {
    let (numerator, denominator) = as_div(ctx, expr)?;
    if !is_one(ctx, numerator) {
        return None;
    }

    let (_fraction_num, fraction_den, den_after) = match ctx.get(denominator).clone() {
        Expr::Add(left, right) => {
            if is_one(ctx, left) {
                let (frac_num, frac_den) = as_div(ctx, right)?;
                let combined_num = ctx.add(Expr::Add(frac_den, frac_num));
                let den_after = ctx.add(Expr::Div(combined_num, frac_den));
                (frac_num, frac_den, den_after)
            } else if is_one(ctx, right) {
                let (frac_num, frac_den) = as_div(ctx, left)?;
                let combined_num = ctx.add(Expr::Add(frac_num, frac_den));
                let den_after = ctx.add(Expr::Div(combined_num, frac_den));
                (frac_num, frac_den, den_after)
            } else {
                return None;
            }
        }
        Expr::Sub(left, right) => {
            if is_one(ctx, left) {
                let (frac_num, frac_den) = as_div(ctx, right)?;
                let combined_num = ctx.add(Expr::Sub(frac_den, frac_num));
                let den_after = ctx.add(Expr::Div(combined_num, frac_den));
                (frac_num, frac_den, den_after)
            } else if is_one(ctx, right) {
                let (frac_num, frac_den) = as_div(ctx, left)?;
                let combined_num = ctx.add(Expr::Sub(frac_num, frac_den));
                let den_after = ctx.add(Expr::Div(combined_num, frac_den));
                (frac_num, frac_den, den_after)
            } else {
                return None;
            }
        }
        _ => return None,
    };

    let full_intermediate = ctx.add(Expr::Div(numerator, den_after));
    let _ = fraction_den;
    Some((denominator, den_after, full_intermediate))
}

pub(super) fn generate_perfect_square_fraction_cancel_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if let (Some(before_local), Some(after_local)) = (step.before_local(), step.after_local()) {
        let substeps = generate_perfect_square_fraction_cancel_substeps_for_pair(
            ctx,
            before_local,
            after_local,
        );
        if !substeps.is_empty() {
            return substeps;
        }
    }

    generate_perfect_square_fraction_cancel_substeps_for_pair(ctx, step.before, step.after)
}

fn generate_perfect_square_fraction_cancel_substeps_for_pair(
    ctx: &Context,
    before: ExprId,
    after: ExprId,
) -> Vec<SubStep> {
    let Expr::Div(numerator, denominator) = ctx.get(before) else {
        return Vec::new();
    };
    if after != *denominator {
        return Vec::new();
    }

    let denominator_display = display_expr(ctx, *denominator);
    let denominator_squared_display = squared_display(ctx, *denominator);

    if is_repeated_factor_product(ctx, *numerator, *denominator) {
        return Vec::new();
    }

    if is_square_of_expr(ctx, *numerator, *denominator) {
        return Vec::new();
    }

    if let Some(square_latex) = perfect_square_form_latex(ctx, *numerator, *denominator) {
        return vec![
            SubStep::new(
                "Reconocer que el numerador es un cuadrado perfecto",
                display_expr(ctx, *numerator),
                squared_display(ctx, *denominator),
            )
            .with_before_latex(latex_expr(ctx, *numerator))
            .with_after_latex(square_latex),
            build_square_over_denominator_cancel_substep(
                ctx,
                format!(
                    "Si {} está dividido entre {}, queda una sola copia",
                    denominator_squared_display, denominator_display
                ),
                *denominator,
            ),
        ];
    }

    let mut temp_ctx = ctx.clone();
    let exponent = temp_ctx.num(2);
    let squared = temp_ctx.add_raw(Expr::Pow(*denominator, exponent));
    if poly_eq(&temp_ctx, *numerator, squared) {
        return vec![
            SubStep::new(
                "Reconocer que el numerador es un cuadrado perfecto",
                display_expr(ctx, *numerator),
                squared_display(ctx, *denominator),
            )
            .with_before_latex(latex_expr(ctx, *numerator))
            .with_after_latex(latex_expr(&temp_ctx, squared)),
            build_square_over_denominator_cancel_substep(
                ctx,
                format!(
                    "Si {} está dividido entre {}, queda una sola copia",
                    denominator_squared_display, denominator_display
                ),
                *denominator,
            ),
        ];
    }
    Vec::new()
}

fn build_square_over_denominator_cancel_substep(
    ctx: &Context,
    title: impl Into<String>,
    denominator: ExprId,
) -> SubStep {
    let mut work = ctx.clone();
    let two = work.num(2);
    let squared = work.add_raw(Expr::Pow(denominator, two));
    let fraction = work.add_raw(Expr::Div(squared, denominator));
    temp_ctx_substep(title, &work, fraction, denominator)
}

pub(super) fn change_of_base_quotient_arguments(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId, ExprId, ExprId)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    let argument = change_of_base_natural_log_argument(ctx, numerator)?;
    let base = change_of_base_natural_log_argument(ctx, denominator)?;
    Some((argument, base, numerator, denominator))
}

pub(super) fn generate_subtract_expanded_cubes_quotient_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let before = step.before_local().unwrap_or(step.before);
    let Some((left, right)) = difference_like_terms(ctx, before) else {
        return Vec::new();
    };
    let Some((numerator, denominator)) = as_div(ctx, left) else {
        return Vec::new();
    };
    let Some(plan) = cube_identity_plan_for_fraction_cancel(ctx, numerator, denominator) else {
        return Vec::new();
    };
    let base_left = plan.left_base;
    let base_right = plan.right_base;
    let kind = plan.kind;

    let identity_title = match kind {
        CubeIdentityKind::Sum => "Usar a^3 + b^3 = (a + b)(a^2 - ab + b^2)",
        CubeIdentityKind::Difference => "Usar a^3 - b^3 = (a - b)(a^2 + ab + b^2)",
    };
    // Build the factored form and the quotient as raw TREES and let the real
    // formatter parenthesize (audit 2026-07-30, ficha D2-004: hand-`format!`ed
    // text published `a^3 - b^3 / a - b` and, with compound bases,
    // `x + 1^3 - y - 2^3` — and the identity substep showed no factorization
    // at all, so its before/after were identical).
    let mut temp_ctx = ctx.clone();
    let two = temp_ctx.num(2);
    let left_sq = temp_ctx.add(Expr::Pow(base_left, two));
    let right_sq = temp_ctx.add(Expr::Pow(base_right, two));
    let cross = temp_ctx.add(Expr::Mul(base_left, base_right));
    let (linear_factor, quadratic_factor) = match kind {
        CubeIdentityKind::Sum => {
            let linear = temp_ctx.add(Expr::Add(base_left, base_right));
            let head = temp_ctx.add(Expr::Sub(left_sq, cross));
            (linear, temp_ctx.add(Expr::Add(head, right_sq)))
        }
        CubeIdentityKind::Difference => {
            let linear = temp_ctx.add(Expr::Sub(base_left, base_right));
            let head = temp_ctx.add(Expr::Add(left_sq, cross));
            (linear, temp_ctx.add(Expr::Add(head, right_sq)))
        }
    };
    let factored = temp_ctx.add(Expr::Mul(linear_factor, quadratic_factor));
    let quotient = temp_ctx.add(Expr::Div(factored, denominator));
    let factored_plain = human_expr(&temp_ctx, factored);
    let factored_latex = latex_expr(&temp_ctx, factored);
    let quotient_plain = human_expr(&temp_ctx, quotient);
    let quotient_latex = latex_expr(&temp_ctx, quotient);

    vec![
        formula_substep(
            identity_title,
            &human_expr(ctx, numerator),
            &factored_plain,
            &latex_expr(ctx, numerator),
            &factored_latex,
        ),
        formula_substep(
            "Cancelar el factor común del numerador y el denominador",
            &quotient_plain,
            &human_expr(ctx, right),
            &quotient_latex,
            &latex_expr(ctx, right),
        ),
    ]
}

pub(super) fn generate_cancel_reciprocal_exponents_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    let local_before = step.before_local().unwrap_or(step.before);
    let local_after = step.after_local().unwrap_or(step.after);
    let Some(_plan) = reciprocal_exponent_plan(ctx, local_before) else {
        return Vec::new();
    };

    let mut out = vec![concrete_expr_substep(
        ctx,
        "El cuadrado deshace la raíz",
        local_before,
        local_after,
    )];

    if let (Some(global_before), Some(global_after)) = (step.global_before, step.global_after) {
        if global_before != local_before || global_after != local_after {
            out.push(
                SubStep::keyed(
                    "polynomial.replace_block_in_expression",
                    vec![],
                    display_expr(ctx, global_before),
                    display_expr(ctx, global_after),
                )
                .with_before_latex(latex_expr(ctx, global_before))
                .with_after_latex(latex_expr(ctx, global_after)),
            );
            return out;
        }
    }

    if step.before != local_before || step.after != local_after {
        out.push(
            SubStep::keyed(
                "polynomial.replace_block_in_expression",
                vec![],
                display_expr(ctx, step.before),
                display_expr(ctx, step.after),
            )
            .with_before_latex(latex_expr(ctx, step.before))
            .with_after_latex(latex_expr(ctx, step.after)),
        );
    } else if local_before != local_after {
        out.push(
            SubStep::new(
                "Reemplazar la raíz al cuadrado por el radicando",
                display_expr(ctx, local_before),
                display_expr(ctx, local_after),
            )
            .with_before_latex(latex_expr(ctx, local_before))
            .with_after_latex(latex_expr(ctx, local_after)),
        );
    }

    out
}

pub(super) fn group_display_for_quotient_numerator(value: &str) -> String {
    if value.contains(" + ") || value.contains(" - ") {
        format!("({value})")
    } else {
        value.to_string()
    }
}

pub(super) fn group_display_for_quotient_denominator(value: &str) -> String {
    if value.contains(" + ") || value.contains(" - ") {
        format!("({value})")
    } else {
        value.to_string()
    }
}

/// Narrate the backend multi-quadratic partial-fraction family with the
/// REAL intermediate decomposition: the backend's own decomposition is
/// rebuilt on a scratch context, so the first substep shows
/// N/prod(q_i) -> sum (alpha_i*x+beta_i)/q_i and the second shows the
/// term-by-term integration to the verified result.
pub(super) fn generate_multi_quadratic_partial_fraction_integration_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if step.rule_name != "Symbolic Integration" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return Vec::new();
    }
    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_name = ctx.sym_name(*var_sym).to_string();

    let mut result = after;
    loop {
        let unwrapped = cas_ast::hold::unwrap_internal_hold(ctx, result);
        if unwrapped == result {
            break;
        }
        result = unwrapped;
    }
    if expr_contains_integrate_call(ctx, result) {
        return Vec::new();
    }

    let mut scratch = ctx.clone();
    let Some(decomposition) =
        cas_math::general_integration_backend::multi_quadratic_partial_fraction_decomposition_expr(
            &mut scratch,
            args[0],
            &var_name,
        )
    else {
        return Vec::new();
    };

    vec![
        SubStep::keyed(
            "partial_fractions.decompose",
            vec![],
            display_expr(ctx, args[0]),
            display_expr(&scratch, decomposition),
        )
        .with_before_latex(latex_expr(ctx, args[0]))
        .with_after_latex(latex_expr(&scratch, decomposition)),
        SubStep::keyed(
            "integral.integrate_simple_terms",
            vec![],
            display_expr(&scratch, decomposition),
            display_expr(ctx, result),
        )
        .with_before_latex(latex_expr(&scratch, decomposition))
        .with_after_latex(latex_expr(ctx, result)),
    ]
}

/// Narrate the mixed-numerator positive-quadratic family produced by the
/// algorithmic backend: integrate((m*(s*x+b)+c)/((s*x+b)^2+a), x) splits
/// into a log part (the derivative of the denominator) and an arctan
/// part. Fires only for a compact single positive-quadratic denominator
/// (Add with a squared side), so partial-fraction and expanded shapes
/// keep their own narrations.
pub(super) fn generate_positive_quadratic_mixed_numerator_integration_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if step.rule_name != "Symbolic Integration" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return Vec::new();
    }
    let Expr::Variable(_) = ctx.get(args[1]) else {
        return Vec::new();
    };

    let Expr::Div(_, denominator) = ctx.get(args[0]) else {
        return Vec::new();
    };
    let denominator = *denominator;
    let compact_denominator = is_compact_positive_quadratic_denominator(ctx, denominator);

    let mut result = after;
    loop {
        let unwrapped = cas_ast::hold::unwrap_internal_hold(ctx, result);
        if unwrapped == result {
            break;
        }
        result = unwrapped;
    }
    let mut terms = Vec::new();
    collect_additive_terms(ctx, result, &mut terms);
    if terms.is_empty()
        || terms
            .iter()
            .any(|term| expr_contains_integrate_call(ctx, *term))
    {
        return Vec::new();
    }
    let log_term = terms
        .iter()
        .copied()
        .find(|term| expr_contains_builtin(ctx, *term, BuiltinFn::Ln));
    let arctan_term = terms.iter().copied().find(|term| {
        expr_contains_builtin(ctx, *term, BuiltinFn::Arctan)
            || expr_contains_builtin(ctx, *term, BuiltinFn::Atan)
    });
    let Some(log_term) = log_term else {
        return Vec::new();
    };

    // Expanded denominators narrate completing the square first; the
    // compact form is recovered from the result's own ln argument (the
    // backend verified their equality).
    let mut substeps = Vec::new();
    if !compact_denominator {
        // The completing-the-square story only applies to a flat expanded
        // polynomial denominator; products of factors belong to the
        // partial-fraction narrators.
        if !matches!(ctx.get(denominator), Expr::Add(..)) {
            return Vec::new();
        }
        let Some(compact_form) = compact_quadratic_from_log_term(ctx, log_term) else {
            return Vec::new();
        };
        substeps.push(
            SubStep::new(
                "Completar el cuadrado en el denominador",
                display_expr(ctx, denominator),
                display_expr(ctx, compact_form),
            )
            .with_before_latex(latex_expr(ctx, denominator))
            .with_after_latex(latex_expr(ctx, compact_form)),
        );
    }

    if arctan_term.is_some() {
        substeps.push(
            SubStep::new(
                "Separar la parte logarítmica del numerador",
                display_expr(ctx, args[0]),
                display_expr(ctx, result),
            )
            .with_before_latex(latex_expr(ctx, args[0]))
            .with_after_latex(latex_expr(ctx, result)),
        );
    } else if compact_denominator || terms.len() > 1 {
        // The compact ln-only shape is owned by the existing log-table
        // narrators; only the expanded ln-only shape continues here.
        return Vec::new();
    }

    substeps.push(
        SubStep::new(
            "Integrar la derivada del denominador como logaritmo",
            display_expr(ctx, denominator),
            display_expr(ctx, log_term),
        )
        .with_before_latex(latex_expr(ctx, denominator))
        .with_after_latex(latex_expr(ctx, log_term)),
    );
    if let Some(arctan_term) = arctan_term {
        substeps.push(
            SubStep::new(
                "Usar la regla de arctan con derivada interna",
                display_expr(ctx, denominator),
                display_expr(ctx, arctan_term),
            )
            .with_before_latex(latex_expr(ctx, denominator))
            .with_after_latex(latex_expr(ctx, arctan_term)),
        );
    }
    if substeps.len() < 2 {
        return Vec::new();
    }
    substeps
}

/// A compact positive-quadratic denominator: Add with one side being a
/// square (Pow(_, 2)). Expanded quadratics and factor products are out of
/// scope here on purpose.
pub(super) fn is_compact_positive_quadratic_denominator(
    ctx: &Context,
    denominator: ExprId,
) -> bool {
    let Expr::Add(left, right) = ctx.get(denominator) else {
        return false;
    };
    [*left, *right].into_iter().any(|side| {
        matches!(ctx.get(side), Expr::Pow(_, exponent)
            if as_rational_const(ctx, *exponent, 4)
                .is_some_and(|value| value == BigRational::from_integer(2.into())))
    })
}

pub(super) fn generate_rational_linear_partial_fraction_integration_substeps(
    ctx: &Context,
    step: &Step,
) -> Vec<SubStep> {
    if step.rule_name != "Symbolic Integration" {
        return Vec::new();
    }

    let before = step.before_local().unwrap_or(step.before);
    let after = step.after_local().unwrap_or(step.after);
    let Expr::Function(fn_id, args) = ctx.get(before) else {
        return Vec::new();
    };
    if ctx.sym_name(*fn_id) != "integrate" || args.len() != 2 {
        return Vec::new();
    }
    if let Expr::Function(after_fn_id, _) = ctx.get(after) {
        if ctx.sym_name(*after_fn_id) == "integrate" {
            return Vec::new();
        }
    }
    let Expr::Variable(var_sym) = ctx.get(args[1]) else {
        return Vec::new();
    };
    let var_name = ctx.sym_name(*var_sym);
    let mut scratch = ctx.clone();
    let decomposition =
        cas_math::symbolic_integration_support::integrate_symbolic_rational_linear_partial_fraction_decomposition_expr(
            &mut scratch,
            args[0],
            var_name,
        )
        .or_else(|| {
            cas_math::symbolic_integration_support::integrate_symbolic_rational_linear_positive_quadratic_partial_fraction_decomposition_expr(
                &mut scratch,
                args[0],
                var_name,
            )
        })
        .or_else(|| {
            cas_math::symbolic_integration_support::integrate_symbolic_positive_quadratic_linear_numerator_decomposition_expr(
                &mut scratch,
                args[0],
                var_name,
            )
        });
    let Some(decomposition) = decomposition else {
        return Vec::new();
    };
    let decomposition_display = display_expr(&scratch, decomposition);
    let decomposition_latex = latex_expr(&scratch, decomposition);
    let integrand_display = display_expr(ctx, args[0]);

    let mut substeps = Vec::new();
    // The oracles return the identity when the integrand is ALREADY decomposed
    // (`1/x`), and announcing a decomposition that does not happen is exactly
    // the "generic template" the substep policy forbids. The honest table
    // statement survives as the remaining substep: `1/x -> ln|x|`.
    if decomposition_display != integrand_display {
        substeps.push(
            SubStep::keyed(
                "partial_fractions.decompose",
                vec![],
                integrand_display,
                decomposition_display.clone(),
            )
            .with_before_latex(latex_expr(ctx, args[0]))
            .with_after_latex(decomposition_latex.clone()),
        );
    }
    substeps.push(
        SubStep::keyed(
            "integral.integrate_simple_terms",
            vec![],
            decomposition_display,
            display_expr(ctx, after),
        )
        .with_before_latex(decomposition_latex)
        .with_after_latex(latex_expr(ctx, after)),
    );
    substeps
}

pub(super) fn quotient_scale_against_polynomial_trace(
    scratch: &mut Context,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
    target: &Polynomial,
    var_name: &str,
) -> Option<BigRational> {
    let numerator = polynomial_product_from_factors_trace(scratch, numerator_factors, var_name)?;
    let denominator =
        polynomial_product_from_factors_trace(scratch, denominator_factors, var_name)?;
    let expected = denominator.mul(target);
    constant_polynomial_ratio(&numerator, &expected)
}

pub(super) fn quotient_after_cancel_once(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
    common_factor: ExprId,
) -> Option<(String, String)> {
    let numerator_factors = cas_math::expr_nary::mul_factors(ctx, numerator);
    let denominator_factors = cas_math::expr_nary::mul_factors(ctx, denominator);

    let numerator_remaining = remove_first_factor(&numerator_factors, common_factor)?;
    let denominator_remaining = remove_first_factor(&denominator_factors, common_factor)?;

    let mut temp_ctx = ctx.clone();
    let quotient = build_quotient_from_factors(
        &mut temp_ctx,
        numerator_remaining.as_slice(),
        denominator_remaining.as_slice(),
    );
    Some(render_temp_expr(&temp_ctx, quotient))
}

pub(super) fn cube_identity_plan_for_fraction_cancel(
    ctx: &Context,
    numerator: ExprId,
    denominator: ExprId,
) -> Option<CubeIdentityPlan> {
    let (left_term, right_term, kind) = cube_identity_terms(ctx, numerator)?;

    let left_base = cube_base_from_term(ctx, left_term)?;
    let right_base = cube_base_from_term(ctx, right_term)?;
    if !linear_factor_matches(ctx, denominator, left_base, right_base, kind) {
        return None;
    }

    Some(CubeIdentityPlan {
        left_base,
        right_base,
        kind,
    })
}

fn reciprocal_exponent_plan(ctx: &Context, before: ExprId) -> Option<ReciprocalExponentPlan> {
    let Expr::Pow(base, exponent) = ctx.get(before) else {
        return None;
    };
    if !is_integer_literal(ctx, *exponent, 2) {
        return None;
    }

    match ctx.get(*base) {
        Expr::Function(fn_id, args)
            if *fn_id == ctx.builtin_id(BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            let _ = args[0];
            Some(ReciprocalExponentPlan)
        }
        Expr::Pow(radicand, inner_exponent) if is_one_half(ctx, *inner_exponent) => {
            let _ = *radicand;
            Some(ReciprocalExponentPlan)
        }
        _ => None,
    }
}
