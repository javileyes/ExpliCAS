//! `arithmetic`: familia `trig_angles`.
//!
//! Ver la cabecera de `arithmetic.rs` para el contexto.

use super::*;

pub(super) fn maybe_trig_sum_to_product_zero_candidate(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let term_count = AddView::from_expr(ctx, expr).terms.len();
    (3..=6).contains(&term_count)
        && top_level_terms_match_trig_family_or_number(
            ctx,
            expr,
            &[BuiltinFn::Sin, BuiltinFn::Cos],
            2,
        )
}

pub(super) fn matches_structural_trig_sum_to_product_zero_scope_family(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    for first_index in 0..view.terms.len() {
        for second_index in (first_index + 1)..view.terms.len() {
            let focus_terms = [view.terms[first_index], view.terms[second_index]];
            let focus_expr = build_signed_sum_expr(ctx, &focus_terms);
            if try_rewrite_trig_sum_to_product_for_cancellation(ctx, focus_expr).is_some() {
                return true;
            }
        }
    }

    false
}

fn extract_scaled_trig_square_for_double_angle_candidate(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BigRational, BuiltinFn, cas_ast::ExprId)> {
    let (base_expr, mut coeff) = if let Expr::Neg(inner) = ctx.get(expr) {
        (*inner, BigRational::from_integer(BigInt::from(-1_i32)))
    } else {
        (expr, BigRational::from_integer(BigInt::from(1_i32)))
    };

    let mut factors = Vec::new();
    let mut stack = vec![base_expr];
    while let Some(curr) = stack.pop() {
        if let Expr::Mul(lhs, rhs) = ctx.get(curr) {
            stack.push(*lhs);
            stack.push(*rhs);
        } else {
            factors.push(curr);
        }
    }

    let mut trig_match = None;
    for factor in &factors {
        if trig_match.is_none() {
            trig_match = extract_trig_square_for_cancellation(ctx, *factor);
            if trig_match.is_some() {
                continue;
            }
        }

        let Expr::Number(n) = ctx.get(*factor) else {
            return None;
        };
        coeff *= n.clone();
    }

    let (trig_fn, arg) = trig_match?;
    Some((coeff, trig_fn, arg))
}

fn numeric_trig_double_angle_cos_variant_zero_scope_candidate_for_focus(
    ctx: &mut cas_ast::Context,
    view: &AddView,
    focus_index: usize,
    focus_sign: Sign,
    scale: BigRational,
    base_arg: cas_ast::ExprId,
) -> bool {
    let signed_scale = scale * BigRational::from_integer(BigInt::from(sign_to_i64(focus_sign)));
    let mut numeric_total = BigRational::zero();
    let mut sin_sq_total = BigRational::zero();
    let mut cos_sq_total = BigRational::zero();

    for (index, (term_expr, term_sign)) in view.terms.iter().copied().enumerate() {
        if index == focus_index {
            continue;
        }

        let signed_term = BigRational::from_integer(BigInt::from(sign_to_i64(term_sign)));
        let positive_expr = match ctx.get(term_expr) {
            Expr::Neg(inner) => *inner,
            _ => term_expr,
        };

        if let Expr::Number(n) = ctx.get(positive_expr) {
            numeric_total += signed_term * n.clone();
            continue;
        }

        let Some((coeff, trig_fn, arg)) =
            extract_scaled_trig_square_for_double_angle_candidate(ctx, term_expr)
        else {
            return false;
        };
        if compare_expr(ctx, arg, base_arg) != Ordering::Equal {
            return false;
        }

        let signed_coeff = signed_term * coeff;
        match trig_fn {
            BuiltinFn::Sin => sin_sq_total += signed_coeff,
            BuiltinFn::Cos => cos_sq_total += signed_coeff,
            _ => return false,
        }
    }

    (sin_sq_total == signed_scale.clone() * BigRational::from_integer(BigInt::from(2_i32))
        && numeric_total == -signed_scale.clone())
        || (cos_sq_total == -signed_scale.clone() * BigRational::from_integer(BigInt::from(2_i32))
            && numeric_total == signed_scale)
}

pub(super) fn maybe_trig_double_angle_cos_variant_zero_scope_candidate(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if !(3..=4).contains(&view.terms.len()) || expr_contains_division_node(ctx, expr) {
        return false;
    }

    let mut fallback_broad_match = false;

    for (focus_index, (term_expr, term_sign)) in view.terms.iter().copied().enumerate() {
        let Some((scale_expr, base_arg)) =
            extract_scaled_double_angle_cosine_for_cancellation(ctx, term_expr)
        else {
            continue;
        };

        let Some(scale) = (match ctx.get(scale_expr) {
            Expr::Number(n) => Some(n.clone()),
            _ => None,
        }) else {
            fallback_broad_match = true;
            continue;
        };

        if numeric_trig_double_angle_cos_variant_zero_scope_candidate_for_focus(
            ctx,
            &view,
            focus_index,
            term_sign,
            scale,
            base_arg,
        ) {
            return true;
        }
    }

    fallback_broad_match
}

pub(super) fn maybe_trig_embedded_double_angle_factor_zero_scope_candidate(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if !(2..=4).contains(&view.terms.len()) || expr_contains_division_node(ctx, expr) {
        return false;
    }

    let mut product_term_count = 0usize;
    let mut rewritable_term_count = 0usize;
    for (term_expr, _) in view.terms {
        let positive_expr = match ctx.get(term_expr) {
            Expr::Neg(inner) => *inner,
            _ => term_expr,
        };
        if matches!(ctx.get(positive_expr), Expr::Number(_)) {
            return false;
        }
        if matches!(ctx.get(positive_expr), Expr::Mul(_, _)) {
            product_term_count += 1;
        }
        if try_rewrite_trig_embedded_double_angle_factor_for_cancellation(ctx, term_expr).is_some()
        {
            rewritable_term_count += 1;
        }
    }

    rewritable_term_count >= 1 && product_term_count >= 2
}

pub(super) fn maybe_two_term_trig_sum_to_product_equivalence_candidate(
    ctx: &cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    let lhs_contains_trig = expr_contains_any_builtin(
        ctx,
        lhs_core,
        &[
            BuiltinFn::Sin,
            BuiltinFn::Cos,
            BuiltinFn::Tan,
            BuiltinFn::Cot,
            BuiltinFn::Sec,
            BuiltinFn::Csc,
        ],
    );
    let lhs_contains_log = expr_contains_any_builtin(
        ctx,
        lhs_core,
        &[BuiltinFn::Ln, BuiltinFn::Log, BuiltinFn::Log10],
    );

    let rhs_contains_trig = expr_contains_any_builtin(
        ctx,
        rhs_core,
        &[
            BuiltinFn::Sin,
            BuiltinFn::Cos,
            BuiltinFn::Tan,
            BuiltinFn::Cot,
            BuiltinFn::Sec,
            BuiltinFn::Csc,
        ],
    );
    let rhs_contains_log = expr_contains_any_builtin(
        ctx,
        rhs_core,
        &[BuiltinFn::Ln, BuiltinFn::Log, BuiltinFn::Log10],
    );

    if !matches!(ctx.get(lhs_core), Expr::Add(_, _) | Expr::Sub(_, _))
        || !lhs_contains_trig
        || lhs_contains_log
        || !rhs_contains_trig
        || rhs_contains_log
    {
        return false;
    }

    let terms = AddView::from_expr(ctx, lhs_core).terms;
    terms.len() == 2
        && terms.iter().all(|(term_expr, _)| {
            expr_contains_any_builtin(
                ctx,
                *term_expr,
                &[
                    BuiltinFn::Sin,
                    BuiltinFn::Cos,
                    BuiltinFn::Tan,
                    BuiltinFn::Cot,
                    BuiltinFn::Sec,
                    BuiltinFn::Csc,
                ],
            )
        })
}

fn is_trig_product_to_sum_product_candidate(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let (_coeff, base) = extract_coef_and_base(ctx, expr);
    if !matches!(ctx.get(base), Expr::Mul(_, _)) {
        return false;
    }

    let mut trig_factor_count = 0usize;
    for factor in flatten_mul_chain(ctx, base) {
        if matches!(ctx.get(factor), Expr::Number(_)) {
            continue;
        }
        if extract_plain_sin_or_cos_arg_for_product_sum_candidate(ctx, factor).is_some() {
            trig_factor_count += 1;
            continue;
        }
        return false;
    }

    trig_factor_count >= 2
}

pub(super) fn maybe_two_term_trig_product_to_sum_equivalence_candidate(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    (is_plain_two_term_sin_cos_sum_or_diff_product_sum_candidate(ctx, lhs_core)
        && is_trig_product_to_sum_product_candidate(ctx, rhs_core))
        || (is_trig_product_to_sum_product_candidate(ctx, lhs_core)
            && is_plain_two_term_sin_cos_sum_or_diff_product_sum_candidate(ctx, rhs_core))
}

fn extract_embedded_double_angle_factor_base_arg(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for factor in factors {
        let factor = match ctx.get(factor) {
            Expr::Neg(inner) => *inner,
            _ => factor,
        };
        let Expr::Function(fn_id, args) = ctx.get(factor) else {
            continue;
        };
        if args.len() != 1
            || (!ctx.is_builtin(*fn_id, BuiltinFn::Sin) && !ctx.is_builtin(*fn_id, BuiltinFn::Cos))
        {
            continue;
        }

        if let Some(base_arg) = extract_double_angle_arg_relaxed(ctx, args[0]) {
            return Some(base_arg);
        }
    }

    None
}

pub(super) fn maybe_two_term_embedded_double_angle_expansion_candidate(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> bool {
    [(lhs_core, rhs_core), (rhs_core, lhs_core)]
        .into_iter()
        .any(|(source, target)| {
            let Some(base_arg) = extract_embedded_double_angle_factor_base_arg(ctx, source) else {
                return false;
            };

            !contains_division_like_term(ctx, target)
                && expr_contains_direct_sin_or_cos_with_arg(ctx, target, base_arg)
        })
}

pub(super) fn try_rewrite_signed_double_angle_contraction_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(rewrite) = try_rewrite_double_angle_contraction_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }

    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        if let Some(rewrite) = try_rewrite_double_angle_contraction_expr(ctx, inner) {
            return Some(ctx.add(Expr::Neg(rewrite.rewritten)));
        }
    }

    let normalized_expr = normalize_additive_scope_expr(ctx, expr);
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, normalized_expr)?;

    let direct_sub = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if let Some(rewrite) = try_rewrite_double_angle_contraction_expr(ctx, direct_sub) {
        return Some(rewrite.rewritten);
    }

    let flipped_sub = ctx.add(Expr::Sub(rhs_core, lhs_core));
    try_rewrite_double_angle_contraction_expr(ctx, flipped_sub)
        .map(|rewrite| ctx.add(Expr::Neg(rewrite.rewritten)))
}

pub(super) fn try_rewrite_trig_sum_to_product_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, &'static str)> {
    if let Some(rewrite) = try_rewrite_sum_to_product_contraction_expr(ctx, expr) {
        let description = match rewrite.kind {
            TrigSumToProductContractionRewriteKind::SinSum => "Expand sine sum to product",
            TrigSumToProductContractionRewriteKind::SinDiff => "Expand sine difference to product",
            TrigSumToProductContractionRewriteKind::CosSum => "Expand cosine sum to product",
            TrigSumToProductContractionRewriteKind::CosDiff => {
                "Expand cosine difference to product"
            }
        };
        return Some((rewrite.rewritten, description));
    }

    if let Some(rewritten) = try_rewrite_signed_double_angle_contraction_for_cancellation(ctx, expr)
    {
        return Some((rewritten, "Double Angle Contraction"));
    }

    let two = ctx.num(2);

    if let Some((arg_a, arg_b)) = extract_trig_two_term_sum(ctx, expr, "sin") {
        let avg_arg = build_avg_with_simplifier(ctx, arg_a, arg_b, crate::collect::collect);
        let half_diff_arg =
            build_half_diff_with_simplifier(ctx, arg_a, arg_b, false, crate::collect::collect);
        let sin_avg = ctx.call_builtin(BuiltinFn::Sin, vec![avg_arg]);
        let cos_half_diff = ctx.call_builtin(BuiltinFn::Cos, vec![half_diff_arg]);
        let product = smart_mul(ctx, sin_avg, cos_half_diff);
        return Some((smart_mul(ctx, two, product), "Expand sine sum to product"));
    }

    if let Some((arg_a, arg_b)) = extract_trig_two_term_diff(ctx, expr, "sin") {
        let avg_arg = build_avg_with_simplifier(ctx, arg_a, arg_b, crate::collect::collect);
        let half_diff_arg =
            build_half_diff_with_simplifier(ctx, arg_a, arg_b, false, crate::collect::collect);
        let cos_avg = ctx.call_builtin(BuiltinFn::Cos, vec![avg_arg]);
        let sin_half_diff = ctx.call_builtin(BuiltinFn::Sin, vec![half_diff_arg]);
        let product = smart_mul(ctx, cos_avg, sin_half_diff);
        return Some((
            smart_mul(ctx, two, product),
            "Expand sine difference to product",
        ));
    }

    if let Some((arg_a, arg_b)) = extract_trig_two_term_sum(ctx, expr, "cos") {
        let avg_arg = build_avg_with_simplifier(ctx, arg_a, arg_b, crate::collect::collect);
        let half_diff_arg =
            build_half_diff_with_simplifier(ctx, arg_a, arg_b, false, crate::collect::collect);
        let half_diff_arg = normalize_for_even_fn(ctx, half_diff_arg);
        let cos_avg = ctx.call_builtin(BuiltinFn::Cos, vec![avg_arg]);
        let cos_half_diff = ctx.call_builtin(BuiltinFn::Cos, vec![half_diff_arg]);
        let product = smart_mul(ctx, cos_avg, cos_half_diff);
        return Some((smart_mul(ctx, two, product), "Expand cosine sum to product"));
    }

    if let Some((arg_a, arg_b)) = extract_trig_two_term_diff(ctx, expr, "cos") {
        let avg_arg = build_avg_with_simplifier(ctx, arg_a, arg_b, crate::collect::collect);
        let half_diff_arg =
            build_half_diff_with_simplifier(ctx, arg_a, arg_b, false, crate::collect::collect);
        let sin_avg = ctx.call_builtin(BuiltinFn::Sin, vec![avg_arg]);
        let sin_half_diff = ctx.call_builtin(BuiltinFn::Sin, vec![half_diff_arg]);
        let product = smart_mul(ctx, sin_avg, sin_half_diff);
        let two_product = smart_mul(ctx, two, product);
        return Some((
            ctx.add(Expr::Neg(two_product)),
            "Expand cosine difference to product",
        ));
    }

    None
}

pub(super) fn try_rewrite_trig_double_angle_cos_one_minus_two_sin_sq_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 || !ctx.is_builtin(fn_id, BuiltinFn::Cos) {
        return None;
    }

    let base_arg = extract_double_angle_arg_relaxed(ctx, args[0])?;
    let one = ctx.num(1);
    let two = ctx.num(2);
    let sin_x = ctx.call_builtin(BuiltinFn::Sin, vec![base_arg]);
    let sin_sq = ctx.add(Expr::Pow(sin_x, two));
    let two_sin_sq = smart_mul(ctx, two, sin_sq);
    Some(ctx.add(Expr::Sub(one, two_sin_sq)))
}

fn extract_plain_trig_square_for_double_angle_half_identity(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(BuiltinFn, cas_ast::ExprId)> {
    fn extract_plain_trig_call(
        ctx: &cas_ast::Context,
        expr: cas_ast::ExprId,
    ) -> Option<(BuiltinFn, cas_ast::ExprId)> {
        let Expr::Function(fn_id, args) = ctx.get(expr) else {
            return None;
        };
        if args.len() != 1 {
            return None;
        }

        if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
            Some((BuiltinFn::Sin, args[0]))
        } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
            Some((BuiltinFn::Cos, args[0]))
        } else {
            None
        }
    }

    match ctx.get(expr) {
        Expr::Pow(base, exp) if extract_i64_integer(ctx, *exp) == Some(2) => {
            extract_plain_trig_call(ctx, *base)
        }
        Expr::Mul(lhs, rhs) => {
            let (lhs_fn, lhs_arg) = extract_plain_trig_call(ctx, *lhs)?;
            let (rhs_fn, rhs_arg) = extract_plain_trig_call(ctx, *rhs)?;
            if lhs_fn == rhs_fn && compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal {
                Some((lhs_fn, lhs_arg))
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) fn try_rewrite_trig_square_double_angle_half_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let (trig_fn, arg) = extract_plain_trig_square_for_double_angle_half_identity(ctx, expr)?;
    let one = ctx.num(1);
    let two = ctx.num(2);
    let double_arg = smart_mul(ctx, two, arg);
    let cos_double_arg = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);

    let numerator = match trig_fn {
        BuiltinFn::Sin => ctx.add(Expr::Sub(one, cos_double_arg)),
        BuiltinFn::Cos => ctx.add(Expr::Add(one, cos_double_arg)),
        _ => return None,
    };

    Some(ctx.add(Expr::Div(numerator, two)))
}

pub(super) fn try_build_fast_recursive_trig_angle_sum_diff_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return None;
    }

    let normalized_terms: Vec<_> = view
        .terms
        .iter()
        .copied()
        .map(|(term_expr, term_sign)| {
            normalize_signed_add_term_for_fast_match(ctx, term_expr, term_sign)
        })
        .collect();

    for focus_index in 0..normalized_terms.len() {
        let (focus_expr, focus_sign) = normalized_terms[focus_index];
        let Some((trig_fn, multiple, base)) =
            extract_trig_linear_multiple_term_for_fast_recursive_identity(ctx, focus_expr)
        else {
            continue;
        };
        if multiple < 2 {
            continue;
        }

        let previous_arg = if multiple == 2 {
            base
        } else {
            let previous_multiple = ctx.num(multiple - 1);
            build_scaled_expr(ctx, previous_multiple, base)
        };
        let prev_sin = ctx.call_builtin(BuiltinFn::Sin, vec![previous_arg]);
        let prev_cos = ctx.call_builtin(BuiltinFn::Cos, vec![previous_arg]);
        let base_sin = ctx.call_builtin(BuiltinFn::Sin, vec![base]);
        let base_cos = ctx.call_builtin(BuiltinFn::Cos, vec![base]);

        let (expanded_positive, expected_remaining) = match trig_fn {
            BuiltinFn::Sin => {
                let term1 = smart_mul(ctx, prev_sin, base_cos);
                let term2 = smart_mul(ctx, prev_cos, base_sin);
                (
                    ctx.add(Expr::Add(term1, term2)),
                    vec![(term1, focus_sign.negate()), (term2, focus_sign.negate())],
                )
            }
            BuiltinFn::Cos => {
                let term1 = smart_mul(ctx, prev_cos, base_cos);
                let term2 = smart_mul(ctx, prev_sin, base_sin);
                (
                    ctx.add(Expr::Sub(term1, term2)),
                    vec![(term1, focus_sign.negate()), (term2, focus_sign)],
                )
            }
            _ => continue,
        };

        let remaining_terms: Vec<_> = normalized_terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != focus_index).then_some(term))
            .collect();

        if !signed_terms_match_multiset(ctx, &remaining_terms, &expected_remaining) {
            continue;
        }

        return Some(Rewrite::with_local(
            ctx.num(0),
            "Angle Sum/Diff Identity",
            focus_expr,
            apply_sign_to_expr(ctx, i64::from(focus_sign.to_i32()), expanded_positive),
        ));
    }

    None
}

pub(super) fn extract_relaxed_double_angle_arg(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    match ctx.get(expr) {
        Expr::Mul(lhs, rhs) => {
            if extract_i64_integer(ctx, *lhs) == Some(2) {
                Some(*rhs)
            } else if extract_i64_integer(ctx, *rhs) == Some(2) {
                Some(*lhs)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn build_trig_sine_product_to_sum_intermediate(
    ctx: &mut cas_ast::Context,
    scale: cas_ast::ExprId,
    arg: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let three = ctx.num(3);
    let triple_arg = smart_mul(ctx, three, arg);
    let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let cos_triple = ctx.call_builtin(BuiltinFn::Cos, vec![triple_arg]);
    let base = ctx.add(Expr::Sub(cos_arg, cos_triple));
    build_scaled_expr(ctx, scale, base)
}

fn build_trig_sine_product_triple_angle_zero_rewrite(
    ctx: &mut cas_ast::Context,
    scope_expr: cas_ast::ExprId,
    target_expr: cas_ast::ExprId,
    scale: cas_ast::ExprId,
    arg: cas_ast::ExprId,
) -> Rewrite {
    let product_sum = build_trig_sine_product_to_sum_intermediate(ctx, scale, arg);
    let cubic_polynomial = build_trig_sine_product_cubic_polynomial(ctx, scale, arg);
    let product_sum_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: product_sum
        }
    );
    let cubic_polynomial_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: cubic_polynomial
        }
    );
    let target_display = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: ctx,
            id: target_expr
        }
    );

    Rewrite::with_local(
        ctx.num(0),
        "Product-to-Sum and Triple-Angle Identity",
        scope_expr,
        ctx.num(0),
    )
    .substep(
        "Convertir producto de senos en diferencia de cosenos",
        vec![format!("Se obtiene {product_sum_display}.")],
    )
    .substep(
        "Usar cos(3u) = 4·cos(u)^3 - 3·cos(u)",
        vec![format!(
            "La expresión se reescribe como {cubic_polynomial_display}."
        )],
    )
    .substep(
        "Usar 1 - cos(u)^2 = sin(u)^2",
        vec![format!("Así se obtiene {target_display}.")],
    )
    .substep(
        "Cancelar términos iguales",
        vec![
            "Tras reconocer la misma forma en el otro lado, toda la expresión se anula."
                .to_string(),
        ],
    )
}

pub(super) fn try_build_exact_trig_sine_product_triple_angle_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for index in 0..view.terms.len() {
        let (term_expr, term_sign) =
            normalize_signed_add_term(ctx, view.terms[index].0, view.terms[index].1);
        let Some((scale, arg)) =
            extract_scaled_double_sine_product_for_cancellation(ctx, term_expr)
        else {
            continue;
        };
        let target_expr = build_trig_sine_product_cosine_cubic_target(ctx, scale, arg);
        let remaining_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(other_index, term)| (other_index != index).then_some(term))
            .collect();
        if remaining_terms.is_empty() {
            continue;
        }
        let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);

        let matches = match term_sign {
            Sign::Pos => {
                expr_matches_negation_after_default_simplify(ctx, remaining_expr, target_expr)
            }
            Sign::Neg => exprs_match_after_default_simplify(ctx, remaining_expr, target_expr),
        };
        if matches {
            return Some(build_trig_sine_product_triple_angle_zero_rewrite(
                ctx,
                expr,
                target_expr,
                scale,
                arg,
            ));
        }
    }

    None
}

pub(super) fn build_trig_square_double_angle_term(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let two = ctx.num(2);
    let doubled_arg = smart_mul(ctx, two, arg);
    ctx.call_builtin(BuiltinFn::Sin, vec![doubled_arg])
}

pub(super) fn matches_direct_half_angle_square_zero_identity(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    try_build_two_term_direct_half_angle_square_rewrite(ctx, expr).is_some()
}

pub(super) fn try_build_exact_trig_sum_to_product_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return None;
    }

    for first_index in 0..view.terms.len() {
        for second_index in (first_index + 1)..view.terms.len() {
            let focus_terms = [view.terms[first_index], view.terms[second_index]];
            let focus_expr = build_signed_sum_expr(ctx, &focus_terms);
            let Some((rewritten, description)) =
                try_rewrite_trig_sum_to_product_for_cancellation(ctx, focus_expr)
            else {
                continue;
            };

            let Some(remaining_index) =
                (0..view.terms.len()).find(|index| *index != first_index && *index != second_index)
            else {
                continue;
            };
            let remaining_expr = signed_term_expr(
                ctx,
                view.terms[remaining_index].0,
                view.terms[remaining_index].1,
            );

            if expr_matches_negation_for_cancellation(ctx, rewritten, remaining_expr)
                || expr_matches_negation_after_default_simplify(ctx, rewritten, remaining_expr)
            {
                return Some(
                    Rewrite::with_local(ctx.num(0), description, focus_expr, rewritten).substep(
                        "Cancelar términos iguales",
                        vec![
                            "Tras aplicar la identidad, el término restante es el opuesto y toda la expresión se anula."
                                .to_string(),
                        ],
                    ),
                );
            }
        }
    }

    None
}

pub(super) fn try_build_exact_trig_product_to_sum_sin_sin_three_term_zero_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return None;
    }

    for (focus_index, (focus_term_expr, focus_term_sign)) in view.terms.iter().copied().enumerate()
    {
        let Some((lhs_arg, rhs_arg)) = extract_exact_double_sine_product_args(ctx, focus_term_expr)
        else {
            continue;
        };

        let mut saw_diff = false;
        let mut saw_sum = false;
        for (index, (term_expr, term_sign)) in view.terms.iter().copied().enumerate() {
            if index == focus_index {
                continue;
            }

            let Some(kind) = classify_exact_cos_sum_or_diff_term_for_sin_sin_zero_scope(
                ctx, term_expr, lhs_arg, rhs_arg,
            ) else {
                saw_diff = false;
                saw_sum = false;
                break;
            };

            match kind {
                ExactSinSinProductToSumCosKind::Diff
                    if term_sign == focus_term_sign.negate() && !saw_diff =>
                {
                    saw_diff = true;
                }
                ExactSinSinProductToSumCosKind::Sum if term_sign == focus_term_sign && !saw_sum => {
                    saw_sum = true;
                }
                _ => {
                    saw_diff = false;
                    saw_sum = false;
                    break;
                }
            }
        }

        if !saw_diff || !saw_sum {
            continue;
        }

        let focus_expr = signed_term_expr(ctx, focus_term_expr, focus_term_sign);
        let diff_arg = ctx.add(Expr::Sub(lhs_arg, rhs_arg));
        let sum_arg = ctx.add(Expr::Add(lhs_arg, rhs_arg));
        let cos_diff = ctx.call_builtin(BuiltinFn::Cos, vec![diff_arg]);
        let cos_sum = ctx.call_builtin(BuiltinFn::Cos, vec![sum_arg]);
        let rewritten = ctx.add(Expr::Sub(cos_diff, cos_sum));
        let adjusted_rewritten = apply_sign_to_expr(ctx, sign_to_i64(focus_term_sign), rewritten);
        return Some(
            Rewrite::with_local(
                ctx.num(0),
                "Product-to-Sum Identity",
                focus_expr,
                adjusted_rewritten,
            )
            .substep(
                "Cancelar términos iguales",
                vec![
                    "La identidad 2·sin(a)·sin(b) = cos(a-b) - cos(a+b) cancela exactamente el residual restante."
                        .to_string(),
                ],
            ),
        );
    }

    None
}

pub(super) fn matches_structural_trig_product_to_sum_sin_sin_three_term_family(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    for (focus_index, (focus_term_expr, _)) in view.terms.iter().copied().enumerate() {
        let Some((lhs_arg, rhs_arg)) = extract_exact_double_sine_product_args(ctx, focus_term_expr)
        else {
            continue;
        };

        let mut saw_diff = false;
        let mut saw_sum = false;
        let mut valid = true;
        for (index, (term_expr, _)) in view.terms.iter().copied().enumerate() {
            if index == focus_index {
                continue;
            }

            match classify_exact_cos_sum_or_diff_term_for_sin_sin_zero_scope(
                ctx, term_expr, lhs_arg, rhs_arg,
            ) {
                Some(ExactSinSinProductToSumCosKind::Diff) if !saw_diff => saw_diff = true,
                Some(ExactSinSinProductToSumCosKind::Sum) if !saw_sum => saw_sum = true,
                _ => {
                    valid = false;
                    break;
                }
            }
        }

        if valid && saw_diff && saw_sum {
            return true;
        }
    }

    false
}

pub(super) fn try_build_exact_trig_product_to_sum_sin_sin_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return None;
    }

    for (focus_index, (focus_term_expr, focus_term_sign)) in view.terms.iter().copied().enumerate()
    {
        let Some(rewrite) = try_rewrite_product_to_sum_expr(ctx, focus_term_expr) else {
            continue;
        };
        if rewrite.kind != TrigProductToSumRewriteKind::SinSin {
            continue;
        }

        let remaining_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != focus_index).then_some(term))
            .collect();
        let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);
        let focus_expr = signed_term_expr(ctx, focus_term_expr, focus_term_sign);
        let adjusted_rewritten = apply_sign_to_expr(
            ctx,
            if focus_term_sign == Sign::Pos { 1 } else { -1 },
            rewrite.rewritten,
        );
        let neg_adjusted_rewritten = ctx.add(Expr::Neg(adjusted_rewritten));
        let distributed_neg_adjusted_rewritten =
            negate_additive_scope_expr(ctx, adjusted_rewritten);

        if exprs_match_for_cancellation(ctx, neg_adjusted_rewritten, remaining_expr)
            || exprs_match_for_cancellation(ctx, distributed_neg_adjusted_rewritten, remaining_expr)
        {
            return Some(
                Rewrite::with_local(
                    ctx.num(0),
                    "Product-to-Sum Identity",
                    focus_expr,
                    adjusted_rewritten,
                )
                .substep(
                    "Cancelar términos iguales",
                    vec![
                        "Tras aplicar producto a suma, el resto de la expresión es el opuesto y toda la expresión se anula."
                            .to_string(),
                    ],
                ),
            );
        }
    }

    None
}

fn try_rewrite_exact_trig_pythagorean_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    if let Some(rewrite) = try_rewrite_pythagorean_identity_add_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }

    if let Some(rewrite) = try_rewrite_pythagorean_factor_form_add_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }

    if let Some(rewrite) = try_rewrite_sec_tan_pythagorean_identity_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }

    if let Some(rewrite) = try_rewrite_csc_cot_pythagorean_identity_expr(ctx, expr) {
        return Some(rewrite.rewritten);
    }

    None
}

pub(super) fn try_build_exact_trig_pythagorean_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if !(2..=3).contains(&view.terms.len()) {
        return None;
    }

    for subset_len in 1..=2 {
        for first_index in 0..view.terms.len() {
            let second_index_options: Vec<Option<usize>> = if subset_len == 1 {
                vec![None]
            } else {
                ((first_index + 1)..view.terms.len()).map(Some).collect()
            };

            for second_index in second_index_options {
                let focus_terms: Vec<_> = view
                    .terms
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, term)| {
                        (index == first_index || second_index == Some(index)).then_some(term)
                    })
                    .collect();
                if focus_terms.len() != subset_len {
                    continue;
                }
                let focus_expr = build_signed_sum_expr(ctx, &focus_terms);

                let remaining_terms: Vec<_> = view
                    .terms
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(index, term)| {
                        (index != first_index && second_index != Some(index)).then_some(term)
                    })
                    .collect();
                if remaining_terms.is_empty() {
                    continue;
                }
                let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);

                let focus_variants = [
                    (focus_expr, 1_i64),
                    (ctx.add(Expr::Neg(focus_expr)), -1_i64),
                ];

                for (candidate_focus, focus_sign) in focus_variants {
                    let Some(rewritten) =
                        try_rewrite_exact_trig_pythagorean_for_cancellation(ctx, candidate_focus)
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
                                "Pythagorean Identity",
                                focus_expr,
                                adjusted_rewritten,
                            )
                            .substep(
                                "Cancelar términos iguales",
                                vec![
                                    "Tras aplicar la identidad pitagórica, el resto de la expresión es el opuesto y toda la expresión se anula."
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

fn is_plain_trig_angle_identity_term(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return false;
    };
    if args.len() != 1 {
        return false;
    }

    matches!(
        ctx.builtin_of(*fn_id),
        Some(BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Tan)
    ) && matches!(ctx.get(args[0]), Expr::Add(_, _) | Expr::Sub(_, _))
}

pub(super) fn expr_contains_plain_trig_angle_identity_term(
    ctx: &cas_ast::Context,
    root: cas_ast::ExprId,
) -> bool {
    let mut stack = vec![root];
    while let Some(expr) = stack.pop() {
        if is_plain_trig_angle_identity_term(ctx, expr) {
            return true;
        }

        match ctx.get(expr) {
            Expr::Function(_, args) => stack.extend(args.iter().copied()),
            Expr::Add(lhs, rhs)
            | Expr::Sub(lhs, rhs)
            | Expr::Mul(lhs, rhs)
            | Expr::Div(lhs, rhs)
            | Expr::Pow(lhs, rhs) => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push(*inner),
            Expr::Matrix { data, .. } => stack.extend(data.iter().copied()),
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }

    false
}

pub(super) fn is_risky_plain_trig_angle_pair_for_common_scale(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    view.terms.len() == 2
        && view
            .terms
            .iter()
            .all(|(term_expr, _)| is_plain_trig_angle_identity_term(ctx, *term_expr))
}

pub(super) fn try_build_two_term_direct_half_angle_square_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, expr)?;

    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewritten) = try_rewrite_trig_square_double_angle_half_expr(ctx, source) else {
            continue;
        };
        if exprs_match_for_cancellation(ctx, rewritten, target) {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Half-Angle Square Identity",
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

pub(super) fn try_build_direct_trig_product_to_sum_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    let rewrite = try_rewrite_product_to_sum_expr(ctx, lhs_core)?;

    if exprs_match_for_cancellation(ctx, rewrite.rewritten, rhs_core)
        || exprs_match_after_default_simplify(ctx, rewrite.rewritten, rhs_core)
    {
        return Some(Rewrite::with_local(
            ctx.num(0),
            "Product-to-Sum Identity",
            lhs_core,
            rhs_core,
        ));
    }

    None
}

pub(super) fn try_build_direct_trig_sum_to_product_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    let (rewritten, description) = try_rewrite_trig_sum_to_product_for_cancellation(ctx, lhs_core)?;

    if exprs_match_for_cancellation(ctx, rewritten, rhs_core)
        || exprs_match_after_default_simplify(ctx, rewritten, rhs_core)
    {
        return Some(Rewrite::with_local(
            ctx.num(0),
            description,
            lhs_core,
            rhs_core,
        ));
    }

    None
}

pub(crate) fn try_build_direct_trig_double_angle_cos_variant_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((scale, base_arg)) =
            extract_scaled_double_angle_cosine_for_cancellation(ctx, source)
        else {
            continue;
        };

        let one_minus_two_sin_sq =
            build_scaled_double_angle_cos_one_minus_two_sin_sq_expr(ctx, scale, base_arg);
        if exprs_match_for_cancellation(ctx, one_minus_two_sin_sq, target) {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Double Angle Expansion",
                lhs_core,
                rhs_core,
            ));
        }

        let two_cos_sq_minus_one =
            build_scaled_double_angle_cos_two_cos_sq_minus_one_expr(ctx, scale, base_arg);
        if exprs_match_for_cancellation(ctx, two_cos_sq_minus_one, target) {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Double Angle Expansion",
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

fn build_scaled_double_angle_cos_one_minus_two_sin_sq_expr(
    ctx: &mut cas_ast::Context,
    scale: cas_ast::ExprId,
    base_arg: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let two = ctx.num(2);
    let sin_x = ctx.call_builtin(BuiltinFn::Sin, vec![base_arg]);
    let sin_sq = ctx.add(Expr::Pow(sin_x, two));
    let two_scale = smart_mul(ctx, two, scale);
    let two_scale = run_default_simplify(ctx, two_scale);
    let two_scale_sin_sq = smart_mul(ctx, two_scale, sin_sq);
    ctx.add(Expr::Sub(scale, two_scale_sin_sq))
}

fn build_scaled_double_angle_cos_two_cos_sq_minus_one_expr(
    ctx: &mut cas_ast::Context,
    scale: cas_ast::ExprId,
    base_arg: cas_ast::ExprId,
) -> cas_ast::ExprId {
    let two = ctx.num(2);
    let cos_x = ctx.call_builtin(BuiltinFn::Cos, vec![base_arg]);
    let cos_sq = ctx.add(Expr::Pow(cos_x, two));
    let two_scale = smart_mul(ctx, two, scale);
    let two_scale = run_default_simplify(ctx, two_scale);
    let two_scale_cos_sq = smart_mul(ctx, two_scale, cos_sq);
    ctx.add(Expr::Sub(two_scale_cos_sq, scale))
}

fn extract_scaled_double_angle_cosine_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 && ctx.is_builtin(fn_id, BuiltinFn::Cos) => {
            let base_arg = extract_double_angle_arg_relaxed(ctx, args[0])?;
            Some((ctx.num(1), base_arg))
        }
        Expr::Mul(_, _) => {
            let factors = flatten_mul_chain(ctx, expr);
            let mut residual_factors = Vec::new();
            let mut base_arg = None;

            for factor in factors {
                if base_arg.is_none() {
                    if let Expr::Function(fn_id, args) = ctx.get(factor).clone() {
                        if args.len() == 1 && ctx.is_builtin(fn_id, BuiltinFn::Cos) {
                            if let Some(inner) = extract_double_angle_arg_relaxed(ctx, args[0]) {
                                base_arg = Some(inner);
                                continue;
                            }
                        }
                    }
                }

                residual_factors.push(factor);
            }

            let base_arg = base_arg?;
            if residual_factors.is_empty() {
                return None;
            }

            let scale = build_mul_expr_from_factors(ctx, &residual_factors);
            Some((run_default_simplify(ctx, scale), base_arg))
        }
        _ => None,
    }
}

fn try_rewrite_trig_embedded_double_angle_factor_for_cancellation(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (index, factor) in factors.iter().copied().enumerate() {
        let rewritten_factor =
            if let Some(rewrite) = try_rewrite_double_angle_function_expr(ctx, factor) {
                if rewrite.kind != TrigMultiAngleRewriteKind::DoubleSin {
                    continue;
                }
                rewrite.rewritten
            } else if let Expr::Neg(inner) = ctx.get(factor).clone() {
                let rewrite = try_rewrite_double_angle_function_expr(ctx, inner)?;
                if rewrite.kind != TrigMultiAngleRewriteKind::DoubleSin {
                    continue;
                }
                ctx.add(Expr::Neg(rewrite.rewritten))
            } else {
                continue;
            };

        let mut rewritten_factors = factors.clone();
        rewritten_factors[index] = rewritten_factor;
        return Some(build_mul_expr_from_factors(ctx, &rewritten_factors));
    }

    None
}

pub(crate) fn try_build_exact_zero_trig_double_angle_cos_variant_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for (focus_index, (focus_term_expr, focus_term_sign)) in view.terms.iter().copied().enumerate()
    {
        let mut rewritten_candidates = Vec::new();

        if let Some((scale, base_arg)) =
            extract_scaled_double_angle_cosine_for_cancellation(ctx, focus_term_expr)
        {
            rewritten_candidates.push(build_scaled_double_angle_cos_one_minus_two_sin_sq_expr(
                ctx, scale, base_arg,
            ));
            rewritten_candidates.push(build_scaled_double_angle_cos_two_cos_sq_minus_one_expr(
                ctx, scale, base_arg,
            ));
        }

        if let Some(rewritten) =
            try_rewrite_signed_double_angle_contraction_for_cancellation(ctx, focus_term_expr)
        {
            rewritten_candidates.push(rewritten);
        }

        if rewritten_candidates.is_empty() {
            continue;
        }

        let remaining_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != focus_index).then_some(term))
            .collect();
        if remaining_terms.is_empty() {
            continue;
        }

        let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);
        let normalized_remaining_expr =
            combine_additive_numeric_constants_for_cancellation(ctx, remaining_expr)
                .unwrap_or(remaining_expr);
        for rewritten in rewritten_candidates {
            let adjusted_rewritten =
                apply_sign_to_expr(ctx, sign_to_i64(focus_term_sign), rewritten);
            let neg_adjusted_rewritten = ctx.add(Expr::Neg(adjusted_rewritten));
            let distributed_neg_adjusted_rewritten =
                negate_additive_scope_expr(ctx, adjusted_rewritten);

            // Cheap syntactic probes stay unconditional; the
            // default-simplify probes are gated like the two-term
            // exact-equivalence path (each one launches a FULL
            // simplifier pipeline - unguarded they explode in breadth
            // and the double-angle/power-reduction pair regenerates
            // its own input one level deeper every round).
            let nested_default_simplify = default_simplify_nesting_depth() > 0;
            if expr_matches_negation_for_cancellation(ctx, adjusted_rewritten, remaining_expr)
                || expr_matches_negation_for_cancellation(
                    ctx,
                    adjusted_rewritten,
                    normalized_remaining_expr,
                )
                || (!nested_default_simplify
                    && (expr_matches_negation_after_default_simplify(
                        ctx,
                        adjusted_rewritten,
                        remaining_expr,
                    ) || expr_matches_negation_after_default_simplify(
                        ctx,
                        adjusted_rewritten,
                        normalized_remaining_expr,
                    ) || exprs_match_after_default_simplify(
                        ctx,
                        neg_adjusted_rewritten,
                        remaining_expr,
                    ) || exprs_match_after_default_simplify(
                        ctx,
                        neg_adjusted_rewritten,
                        normalized_remaining_expr,
                    ) || exprs_match_after_default_simplify(
                        ctx,
                        distributed_neg_adjusted_rewritten,
                        remaining_expr,
                    ) || exprs_match_after_default_simplify(
                        ctx,
                        distributed_neg_adjusted_rewritten,
                        normalized_remaining_expr,
                    ) || additive_scopes_match_after_default_simplify(
                        ctx,
                        distributed_neg_adjusted_rewritten,
                        remaining_expr,
                    ) || additive_scopes_match_after_default_simplify(
                        ctx,
                        distributed_neg_adjusted_rewritten,
                        normalized_remaining_expr,
                    )))
            {
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    "Double Angle Expansion",
                    focus_term_expr,
                    rewritten,
                ));
            }
        }
    }

    None
}

pub(super) fn try_build_direct_trig_embedded_double_angle_expansion_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewritten) =
            try_rewrite_trig_embedded_double_angle_factor_for_cancellation(ctx, source)
        else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, rewritten, target)
            || exprs_match_after_default_simplify(ctx, rewritten, target)
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Double Angle Expansion",
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

fn try_rewrite_trig_cos_double_angle_times_cos_polynomial_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 3 {
        return None;
    }

    let mut saw_two = false;
    let mut cos_args = Vec::with_capacity(2);
    for factor in factors {
        if !saw_two && extract_i64_integer(ctx, factor) == Some(2) {
            saw_two = true;
            continue;
        }

        let Expr::Function(fn_id, args) = ctx.get(factor) else {
            return None;
        };
        if !ctx.is_builtin(*fn_id, BuiltinFn::Cos) || args.len() != 1 {
            return None;
        }
        cos_args.push(args[0]);
    }

    if !saw_two || cos_args.len() != 2 {
        return None;
    }

    let base_arg = if extract_double_angle_arg_relaxed(ctx, cos_args[0])
        .is_some_and(|inner| compare_expr(ctx, inner, cos_args[1]) == Ordering::Equal)
    {
        cos_args[1]
    } else if extract_double_angle_arg_relaxed(ctx, cos_args[1])
        .is_some_and(|inner| compare_expr(ctx, inner, cos_args[0]) == Ordering::Equal)
    {
        cos_args[0]
    } else {
        return None;
    };

    let cos_base = ctx.call_builtin(BuiltinFn::Cos, vec![base_arg]);
    let three = ctx.num(3);
    let cos_cube = ctx.add(Expr::Pow(cos_base, three));
    let four = ctx.num(4);
    let four_cos_cube = smart_mul(ctx, four, cos_cube);
    let two = ctx.num(2);
    let two_cos = smart_mul(ctx, two, cos_base);
    Some(ctx.add(Expr::Sub(four_cos_cube, two_cos)))
}

fn try_rewrite_trig_cos_double_angle_times_sin_polynomial_expr(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 3 {
        return None;
    }

    let mut saw_two = false;
    let mut cos_double_arg = None;
    let mut sin_arg = None;
    for factor in factors {
        if !saw_two && extract_i64_integer(ctx, factor) == Some(2) {
            saw_two = true;
            continue;
        }

        let Expr::Function(fn_id, args) = ctx.get(factor) else {
            return None;
        };
        if args.len() != 1 {
            return None;
        }

        if ctx.is_builtin(*fn_id, BuiltinFn::Cos) && cos_double_arg.is_none() {
            cos_double_arg = Some(args[0]);
        } else if ctx.is_builtin(*fn_id, BuiltinFn::Sin) && sin_arg.is_none() {
            sin_arg = Some(args[0]);
        } else {
            return None;
        }
    }

    let (Some(cos_double_arg), Some(base_arg)) = (cos_double_arg, sin_arg) else {
        return None;
    };
    let inner = extract_double_angle_arg_relaxed(ctx, cos_double_arg)?;
    if compare_expr(ctx, inner, base_arg) != Ordering::Equal {
        return None;
    }

    let cos_base = ctx.call_builtin(BuiltinFn::Cos, vec![base_arg]);
    let two = ctx.num(2);
    let cos_sq = ctx.add(Expr::Pow(cos_base, two));
    let sin_base = ctx.call_builtin(BuiltinFn::Sin, vec![base_arg]);
    let cos_sq_sin = smart_mul(ctx, cos_sq, sin_base);
    let four = ctx.num(4);
    let four_cos_sq_sin = smart_mul(ctx, four, cos_sq_sin);
    let two_sin = smart_mul(ctx, two, sin_base);
    Some(ctx.add(Expr::Sub(four_cos_sq_sin, two_sin)))
}

pub(crate) fn try_build_direct_trig_cos_double_angle_polynomial_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewritten) =
            try_rewrite_trig_cos_double_angle_times_cos_polynomial_expr(ctx, source)
        else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, rewritten, target)
            || exprs_match_after_default_simplify(ctx, rewritten, target)
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Double Angle Expansion",
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

pub(super) fn try_build_direct_trig_mixed_double_angle_polynomial_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewritten) =
            try_rewrite_trig_cos_double_angle_times_sin_polynomial_expr(ctx, source)
        else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, rewritten, target)
            || exprs_match_after_default_simplify(ctx, rewritten, target)
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Double Angle Expansion",
                lhs_core,
                rhs_core,
            ));
        }
    }

    None
}

pub(super) fn try_build_small_sec_tan_pythagorean_zero_core_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let zero = ctx.num(0);
    let one = ctx.num(1);
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, expr)?;

    for (identity_side, one_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if extract_i64_integer(ctx, one_side) != Some(1) {
            continue;
        }
        let Some(rewrite) = try_rewrite_sec_tan_pythagorean_identity_expr(ctx, identity_side)
        else {
            continue;
        };
        if !(exprs_match_for_cancellation(ctx, rewrite.rewritten, one)
            || exprs_match_after_default_simplify(ctx, rewrite.rewritten, one))
        {
            continue;
        }

        return Some(
            Rewrite::with_local(zero, "Pythagorean Identity", identity_side, one_side).substep(
                "Usar sec(u)^2 - tan(u)^2 = 1",
                vec![
                    "La identidad pitagórica deja 1, y la resta restante se anula exactamente."
                        .to_string(),
                ],
            ),
        );
    }

    None
}

pub(super) fn try_build_small_csc_cot_pythagorean_zero_core_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let zero = ctx.num(0);
    let one = ctx.num(1);
    let (lhs_core, rhs_core) = extract_two_term_core_difference(ctx, expr)?;

    for (identity_side, one_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if extract_i64_integer(ctx, one_side) != Some(1) {
            continue;
        }
        let Some(rewrite) = try_rewrite_csc_cot_pythagorean_identity_expr(ctx, identity_side)
        else {
            continue;
        };
        if !(exprs_match_for_cancellation(ctx, rewrite.rewritten, one)
            || exprs_match_after_default_simplify(ctx, rewrite.rewritten, one))
        {
            continue;
        }

        return Some(
            Rewrite::with_local(zero, "Pythagorean Identity", identity_side, one_side).substep(
                "Usar csc(u)^2 - cot(u)^2 = 1",
                vec![
                    "La identidad pitagórica deja 1, y la resta restante se anula exactamente."
                        .to_string(),
                ],
            ),
        );
    }

    None
}

pub(super) fn try_build_exact_zero_trig_cos_double_angle_polynomial_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for (focus_index, (focus_term_expr, focus_term_sign)) in view.terms.iter().copied().enumerate()
    {
        let Some(rewritten) =
            try_rewrite_trig_cos_double_angle_times_cos_polynomial_expr(ctx, focus_term_expr)
        else {
            continue;
        };

        let remaining_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != focus_index).then_some(term))
            .collect();
        if remaining_terms.is_empty() {
            continue;
        }
        let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);
        let adjusted_rewritten = apply_sign_to_expr(ctx, sign_to_i64(focus_term_sign), rewritten);
        let neg_adjusted_rewritten = ctx.add(Expr::Neg(adjusted_rewritten));
        let distributed_neg_adjusted_rewritten =
            negate_additive_scope_expr(ctx, adjusted_rewritten);

        if expr_matches_negation_for_cancellation(ctx, adjusted_rewritten, remaining_expr) {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Double Angle Expansion",
                focus_term_expr,
                rewritten,
            ));
        }

        let remaining_term_count = AddView::from_expr(ctx, remaining_expr).terms.len();
        let adjusted_term_count = AddView::from_expr(ctx, adjusted_rewritten).terms.len();
        let distributed_term_count = AddView::from_expr(ctx, distributed_neg_adjusted_rewritten)
            .terms
            .len();
        if adjusted_term_count != remaining_term_count
            && distributed_term_count != remaining_term_count
            && !product_has_top_level_additive_factor(ctx, remaining_expr)
        {
            continue;
        }

        if expr_matches_negation_after_default_simplify(ctx, adjusted_rewritten, remaining_expr)
            || exprs_match_after_default_simplify(ctx, neg_adjusted_rewritten, remaining_expr)
            || exprs_match_after_default_simplify(
                ctx,
                distributed_neg_adjusted_rewritten,
                remaining_expr,
            )
            || additive_scopes_match_after_default_simplify(
                ctx,
                distributed_neg_adjusted_rewritten,
                remaining_expr,
            )
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Double Angle Expansion",
                focus_term_expr,
                rewritten,
            ));
        }
    }

    None
}

pub(super) fn try_build_exact_zero_trig_mixed_double_angle_polynomial_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for (focus_index, (focus_term_expr, focus_term_sign)) in view.terms.iter().copied().enumerate()
    {
        let Some(rewritten) =
            try_rewrite_trig_cos_double_angle_times_sin_polynomial_expr(ctx, focus_term_expr)
        else {
            continue;
        };

        let remaining_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != focus_index).then_some(term))
            .collect();
        if remaining_terms.is_empty() {
            continue;
        }
        let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);
        let adjusted_rewritten = apply_sign_to_expr(ctx, sign_to_i64(focus_term_sign), rewritten);
        let neg_adjusted_rewritten = ctx.add(Expr::Neg(adjusted_rewritten));
        let distributed_neg_adjusted_rewritten =
            negate_additive_scope_expr(ctx, adjusted_rewritten);

        if expr_matches_negation_for_cancellation(ctx, adjusted_rewritten, remaining_expr)
            || expr_matches_negation_after_default_simplify(ctx, adjusted_rewritten, remaining_expr)
            || exprs_match_after_default_simplify(ctx, neg_adjusted_rewritten, remaining_expr)
            || exprs_match_after_default_simplify(
                ctx,
                distributed_neg_adjusted_rewritten,
                remaining_expr,
            )
            || additive_scopes_match_after_default_simplify(
                ctx,
                distributed_neg_adjusted_rewritten,
                remaining_expr,
            )
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Double Angle Expansion",
                focus_term_expr,
                rewritten,
            ));
        }
    }

    None
}

pub(super) fn try_build_exact_zero_trig_embedded_double_angle_factor_zero_scope_rewrite(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for (focus_index, (focus_term_expr, focus_term_sign)) in view.terms.iter().copied().enumerate()
    {
        let Some(rewritten) =
            try_rewrite_trig_embedded_double_angle_factor_for_cancellation(ctx, focus_term_expr)
        else {
            continue;
        };

        let remaining_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != focus_index).then_some(term))
            .collect();
        if remaining_terms.is_empty() {
            continue;
        }
        let remaining_expr = build_signed_sum_expr(ctx, &remaining_terms);
        let adjusted_rewritten = apply_sign_to_expr(ctx, sign_to_i64(focus_term_sign), rewritten);
        let neg_adjusted_rewritten = ctx.add(Expr::Neg(adjusted_rewritten));
        let distributed_neg_adjusted_rewritten =
            negate_additive_scope_expr(ctx, adjusted_rewritten);

        if expr_matches_negation_for_cancellation(ctx, adjusted_rewritten, remaining_expr)
            || expr_matches_negation_after_default_simplify(ctx, adjusted_rewritten, remaining_expr)
            || exprs_match_after_default_simplify(ctx, neg_adjusted_rewritten, remaining_expr)
            || exprs_match_after_default_simplify(
                ctx,
                distributed_neg_adjusted_rewritten,
                remaining_expr,
            )
            || additive_scopes_match_after_default_simplify(
                ctx,
                distributed_neg_adjusted_rewritten,
                remaining_expr,
            )
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Double Angle Expansion",
                focus_term_expr,
                rewritten,
            ));
        }
    }

    None
}

pub(super) fn match_tan_triple_angle_contraction_arg(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
    let (_, denom) = as_div(ctx, expr)?;
    let tan_arg = find_first_tan_arg(ctx, expr)?;

    let tan_x = ctx.call_builtin(BuiltinFn::Tan, vec![tan_arg]);
    let three = ctx.num(3);
    let one = ctx.num(1);
    let three_pow = ctx.num(3);
    let tan_cubed = ctx.add(Expr::Pow(tan_x, three_pow));
    let three_tan = smart_mul(ctx, three, tan_x);
    let numer = ctx.add(Expr::Sub(three_tan, tan_cubed));
    let two_pow = ctx.num(2);
    let tan_squared = ctx.add(Expr::Pow(tan_x, two_pow));
    let three_tan_squared = smart_mul(ctx, three, tan_squared);
    let denom_expected = ctx.add(Expr::Sub(one, three_tan_squared));
    let expected = ctx.add(Expr::Div(numer, denom_expected));

    let expected_normalized = cas_math::canonical_forms::normalize_core(ctx, expected);
    let expr_normalized = cas_math::canonical_forms::normalize_core(ctx, expr);
    (compare_expr(ctx, expected_normalized, expr_normalized) == Ordering::Equal)
        .then_some((tan_arg, denom))
}

pub(super) fn try_build_direct_multi_angle_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if let Some((tan_arg, denom)) = match_tan_triple_angle_contraction_arg(ctx, source) {
            let three = ctx.num(3);
            let triple_arg = smart_mul(ctx, three, tan_arg);
            let tan_triple = ctx.call_builtin(BuiltinFn::Tan, vec![triple_arg]);
            let tan_triple_normalized = cas_math::canonical_forms::normalize_core(ctx, tan_triple);
            let target_normalized = cas_math::canonical_forms::normalize_core(ctx, target);
            if compare_expr(ctx, tan_triple, target) == Ordering::Equal
                || compare_expr(ctx, tan_triple_normalized, target_normalized) == Ordering::Equal
            {
                return Some(
                    Rewrite::with_local(ctx.num(0), "Triple Angle Identity", lhs_core, rhs_core)
                        .requires(crate::ImplicitCondition::NonZero(denom)),
                );
            }
        }

        if let Some(rewrite) = try_rewrite_triple_angle_expr(ctx, source) {
            if exprs_match_for_cancellation(ctx, rewrite.rewritten, target)
                || exprs_match_after_default_simplify(ctx, rewrite.rewritten, target)
            {
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    "Triple Angle Identity",
                    lhs_core,
                    rhs_core,
                ));
            }
        }

        if let Some(rewrite) = try_rewrite_quintuple_angle_expr(ctx, source) {
            if exprs_match_for_cancellation(ctx, rewrite.rewritten, target)
                || exprs_match_after_default_simplify(ctx, rewrite.rewritten, target)
            {
                return Some(Rewrite::with_local(
                    ctx.num(0),
                    "Quintuple Angle Identity",
                    lhs_core,
                    rhs_core,
                ));
            }
        }
    }

    None
}

pub(super) fn try_match_half_angle_tan_equivalence(
    ctx: &mut cas_ast::Context,
    source: cas_ast::ExprId,
    target: cas_ast::ExprId,
) -> Option<(&'static str, cas_ast::ExprId, &'static str, &'static str)> {
    let (numerator, denominator) = as_div(ctx, source)?;

    let sin_arg = extract_unary_builtin_arg(ctx, denominator, BuiltinFn::Sin)?;
    let base_arg = extract_double_angle_arg_relaxed(ctx, sin_arg)?;
    let target_arg = extract_unary_builtin_arg(ctx, target, BuiltinFn::Tan)?;
    if !exprs_match_with_local_default_simplify(ctx, base_arg, target_arg) {
        return None;
    }

    let one = ctx.num(1);
    let two = ctx.num(2);
    let canonical_double_arg_raw = smart_mul(ctx, two, base_arg);
    let canonical_double_arg = run_default_simplify(ctx, canonical_double_arg_raw);
    let cos_double = ctx.call_builtin(BuiltinFn::Cos, vec![canonical_double_arg]);
    let expected_numerator = ctx.add(Expr::Sub(one, cos_double));
    if !exprs_match_with_local_default_simplify(ctx, numerator, expected_numerator) {
        return None;
    }

    Some((
        "Half Angle Identity",
        denominator,
        "Usar (1 - cos(2u))/sin(2u) = tan(u)",
        "La forma de semiaángulo se reconoce localmente y evita el fallback de simplificación general.",
    ))
}
