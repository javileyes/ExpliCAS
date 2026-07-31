//! `arithmetic`: familia `logarithms`.
//!
//! Ver la cabecera de `arithmetic.rs` para el contexto.

use super::*;

fn extract_scaled_log_abs_mul_div(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId)> {
    let extract_direct = |ctx: &mut cas_ast::Context,
                          expr: cas_ast::ExprId|
     -> Option<(cas_ast::ExprId, cas_ast::ExprId)> {
        let (base, arg) = try_extract_log_parts(ctx, expr)?;
        let inner = abs_argument(ctx, arg)?;
        Some((base, inner))
    };

    if let Some((base, inner)) = extract_direct(ctx, expr) {
        let one = ctx.num(1);
        return Some((one, base, inner));
    }

    match ctx.get(expr).clone() {
        Expr::Mul(lhs, rhs) => {
            if let Some((base, inner)) = extract_direct(ctx, lhs) {
                Some((rhs, base, inner))
            } else if let Some((base, inner)) = extract_direct(ctx, rhs) {
                Some((lhs, base, inner))
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) fn try_match_log_abs_mul_div_cancellation_side(
    ctx: &mut cas_ast::Context,
    focus_expr: cas_ast::ExprId,
) -> Option<LogAbsMulDivCancellationMatch> {
    let (scale, log_base, inner) = extract_scaled_log_abs_mul_div(ctx, focus_expr)?;
    if let Some((lhs, rhs)) = as_mul(ctx, inner) {
        let lhs_abs = ctx.call_builtin(BuiltinFn::Abs, vec![lhs]);
        let rhs_abs = ctx.call_builtin(BuiltinFn::Abs, vec![rhs]);
        let lhs_log = make_log_expr(ctx, log_base, lhs_abs);
        let rhs_log = make_log_expr(ctx, log_base, rhs_abs);
        let expanded = ctx.add(Expr::Add(lhs_log, rhs_log));
        let focus_after = build_scaled_expr(ctx, scale, expanded);
        return Some(LogAbsMulDivCancellationMatch {
            focus_after,
            components: [
                (build_scaled_expr(ctx, scale, lhs_log), Sign::Pos),
                (build_scaled_expr(ctx, scale, rhs_log), Sign::Pos),
            ],
        });
    }

    if let Some((num, den)) = as_div(ctx, inner) {
        let num_abs = ctx.call_builtin(BuiltinFn::Abs, vec![num]);
        let den_abs = ctx.call_builtin(BuiltinFn::Abs, vec![den]);
        let num_log = make_log_expr(ctx, log_base, num_abs);
        let den_log = make_log_expr(ctx, log_base, den_abs);
        let expanded = ctx.add(Expr::Sub(num_log, den_log));
        let focus_after = build_scaled_expr(ctx, scale, expanded);
        return Some(LogAbsMulDivCancellationMatch {
            focus_after,
            components: [
                (build_scaled_expr(ctx, scale, num_log), Sign::Pos),
                (build_scaled_expr(ctx, scale, den_log), Sign::Neg),
            ],
        });
    }

    None
}

fn normalize_log_argument_for_cancellation(
    ctx: &mut cas_ast::Context,
    arg: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, bool)> {
    if let Some((pow_base, pow_exp)) = as_pow(ctx, arg) {
        let power = small_positive_integer_value(ctx, pow_exp)?;
        let normalized_arg = if power % 2 == 0 {
            ctx.call_builtin(BuiltinFn::Abs, vec![pow_base])
        } else {
            pow_base
        };
        let one = ctx.num(1);
        return Some((
            normalized_arg,
            pow_exp,
            compare_expr(ctx, pow_exp, one) != Ordering::Equal,
        ));
    }

    let one = ctx.num(1);
    Some((arg, one, false))
}

fn collect_log_arg_terms_for_cancellation(
    ctx: &mut cas_ast::Context,
    log_base: cas_ast::ExprId,
    arg: cas_ast::ExprId,
    scale: cas_ast::ExprId,
    sign: Sign,
    out: &mut Vec<(cas_ast::ExprId, Sign)>,
    changed_by_power: &mut bool,
) -> Option<()> {
    if let Some((num, den)) = as_div(ctx, arg) {
        collect_log_arg_terms_for_cancellation(
            ctx,
            log_base,
            num,
            scale,
            sign,
            out,
            changed_by_power,
        )?;
        collect_log_arg_terms_for_cancellation(
            ctx,
            log_base,
            den,
            scale,
            sign.negate(),
            out,
            changed_by_power,
        )?;
        return Some(());
    }

    if as_mul(ctx, arg).is_some() {
        for factor in flatten_mul_chain(ctx, arg) {
            collect_log_arg_terms_for_cancellation(
                ctx,
                log_base,
                factor,
                scale,
                sign,
                out,
                changed_by_power,
            )?;
        }
        return Some(());
    }

    if matches!(ctx.get(arg), Expr::Pow(_, _)) {
        let (normalized_arg, power_scale, did_change_by_power) =
            normalize_log_argument_for_cancellation(ctx, arg)?;
        *changed_by_power |= did_change_by_power;
        let one = ctx.num(1);
        let total_scale = if compare_expr(ctx, power_scale, one) == Ordering::Equal {
            scale
        } else if compare_expr(ctx, scale, one) == Ordering::Equal {
            power_scale
        } else {
            smart_mul(ctx, scale, power_scale)
        };
        return collect_log_arg_terms_for_cancellation(
            ctx,
            log_base,
            normalized_arg,
            total_scale,
            sign,
            out,
            changed_by_power,
        );
    }

    if let Some(inner) = abs_argument(ctx, arg) {
        if let Some((num, den)) = as_div(ctx, inner) {
            let num_abs = ctx.call_builtin(BuiltinFn::Abs, vec![num]);
            let den_abs = ctx.call_builtin(BuiltinFn::Abs, vec![den]);
            collect_log_arg_terms_for_cancellation(
                ctx,
                log_base,
                num_abs,
                scale,
                sign,
                out,
                changed_by_power,
            )?;
            collect_log_arg_terms_for_cancellation(
                ctx,
                log_base,
                den_abs,
                scale,
                sign.negate(),
                out,
                changed_by_power,
            )?;
            return Some(());
        }

        if as_mul(ctx, inner).is_some() {
            for factor in flatten_mul_chain(ctx, inner) {
                let abs_factor = ctx.call_builtin(BuiltinFn::Abs, vec![factor]);
                collect_log_arg_terms_for_cancellation(
                    ctx,
                    log_base,
                    abs_factor,
                    scale,
                    sign,
                    out,
                    changed_by_power,
                )?;
            }
            return Some(());
        }
    }

    let log_expr = make_log_expr(ctx, log_base, arg);
    out.push((build_scaled_expr(ctx, scale, log_expr), sign));
    Some(())
}

pub(super) fn try_normalize_log_term_for_fast_match(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<cas_ast::ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut log_factor = None;
    let mut scale_factors = Vec::new();
    for factor in factors {
        if try_extract_log_parts(ctx, factor).is_some() {
            if log_factor.is_some() {
                return None;
            }
            log_factor = Some(factor);
        } else {
            scale_factors.push(factor);
        }
    }

    let log_factor = log_factor?;
    let (log_base, log_arg) = try_extract_log_parts(ctx, log_factor)?;
    let scale = match scale_factors.len() {
        0 => ctx.num(1),
        1 => scale_factors[0],
        _ => build_balanced_mul(ctx, &scale_factors),
    };

    let mut components = Vec::new();
    let mut changed_by_power = false;
    collect_log_arg_terms_for_cancellation(
        ctx,
        log_base,
        log_arg,
        scale,
        Sign::Pos,
        &mut components,
        &mut changed_by_power,
    )?;

    (components.len() == 1 && components[0].1 == Sign::Pos).then_some(components[0].0)
}

fn try_extract_scaled_log_term_parts(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> Option<(cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut log_factor = None;
    let mut scale_factors = Vec::new();
    for factor in factors {
        if try_extract_log_parts(ctx, factor).is_some() {
            if log_factor.is_some() {
                return None;
            }
            log_factor = Some(factor);
        } else {
            scale_factors.push(factor);
        }
    }

    let log_factor = log_factor?;
    let (log_base, log_arg) = try_extract_log_parts(ctx, log_factor)?;
    let scale = match scale_factors.len() {
        0 => ctx.num(1),
        1 => scale_factors[0],
        _ => build_balanced_mul(ctx, &scale_factors),
    };

    Some((scale, log_base, log_arg))
}

fn log_arg_supports_product_power_cancellation_expansion(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Mul(_, _) | Expr::Div(_, _) | Expr::Pow(_, _) => true,
        Expr::Function(fn_id, args) if ctx.is_builtin(fn_id, BuiltinFn::Abs) => args
            .first()
            .copied()
            .is_some_and(|inner| log_arg_supports_product_power_cancellation_expansion(ctx, inner)),
        _ => false,
    }
}

pub(super) fn maybe_log_product_power_zero_candidate(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 3 {
        return false;
    }

    let mut top_level_log_terms = 0;
    let mut top_level_nonlog_terms = 0;
    let mut has_plausible_expandable_focus = false;
    for (term_expr, term_sign) in view.terms.iter().copied() {
        let (normalized_term_expr, _normalized_term_sign) =
            normalize_signed_add_term(ctx, term_expr, term_sign);
        let Some((_scale, _log_base, log_arg)) =
            try_extract_scaled_log_term_parts(ctx, normalized_term_expr)
        else {
            top_level_nonlog_terms += 1;
            continue;
        };
        top_level_log_terms += 1;
        if !has_plausible_expandable_focus
            && log_arg_supports_product_power_cancellation_expansion(ctx, log_arg)
        {
            has_plausible_expandable_focus =
                try_match_log_product_power_cancellation_side(ctx, normalized_term_expr)
                    .is_some_and(|matched| {
                        matched.components.len() <= view.terms.len().saturating_sub(1)
                    });
        }
    }

    top_level_nonlog_terms == 0 && top_level_log_terms >= 2 && has_plausible_expandable_focus
}

pub(super) fn maybe_log_abs_mul_div_zero_candidate(
    ctx: &mut cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 3 || !expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Abs]) {
        return false;
    }

    let mut top_level_log_terms = 0;
    let mut top_level_nonlog_terms = 0;
    let mut expandable_abs_log_term = false;
    for (term_expr, term_sign) in view.terms.iter().copied() {
        let (normalized_term_expr, _normalized_term_sign) =
            normalize_signed_add_term(ctx, term_expr, term_sign);
        if try_extract_scaled_log_term_parts(ctx, normalized_term_expr).is_some() {
            top_level_log_terms += 1;
        } else {
            top_level_nonlog_terms += 1;
        }
        if extract_scaled_log_abs_mul_div(ctx, normalized_term_expr).is_some() {
            expandable_abs_log_term = true;
        }
    }

    top_level_nonlog_terms == 0 && top_level_log_terms >= 2 && expandable_abs_log_term
}

pub(super) fn log_terms_match_up_to_abs_subject_for_cancellation(
    ctx: &mut cas_ast::Context,
    lhs: cas_ast::ExprId,
    rhs: cas_ast::ExprId,
) -> bool {
    let Some((lhs_scale, lhs_base, lhs_arg)) = try_extract_scaled_log_term_parts(ctx, lhs) else {
        return false;
    };
    let Some((rhs_scale, rhs_base, rhs_arg)) = try_extract_scaled_log_term_parts(ctx, rhs) else {
        return false;
    };

    // Compare bases with the log-base-aware comparator: `try_extract_*_log_term_parts`
    // returns `ln_base_sentinel()` (a non-arena ExprId) for natural logs, which would
    // panic `compare_expr`/`exprs_match_for_cancellation` (an out-of-bounds `Context::get`)
    // when compared against a real base such as the `2` of `log(2, x)`. `bases_equal_for_logs`
    // resolves the sentinel (it matches `e` and itself) without dereferencing it.
    if !exprs_match_for_cancellation(ctx, lhs_scale, rhs_scale)
        || !cas_math::logarithm_inverse_support::bases_equal_for_logs(ctx, lhs_base, rhs_base)
    {
        return false;
    }

    exprs_match_for_cancellation(ctx, lhs_arg, rhs_arg)
        || abs_argument(ctx, lhs_arg)
            .is_some_and(|inner| exprs_match_for_cancellation(ctx, inner, rhs_arg))
        || abs_argument(ctx, rhs_arg)
            .is_some_and(|inner| exprs_match_for_cancellation(ctx, lhs_arg, inner))
}

pub(super) fn log_cancellation_component_matches(
    ctx: &mut cas_ast::Context,
    normalized_term_expr: cas_ast::ExprId,
    fast_log_term: Option<cas_ast::ExprId>,
    component_expr: cas_ast::ExprId,
) -> bool {
    fast_log_term
        .is_some_and(|fast_term| compare_expr(ctx, fast_term, component_expr) == Ordering::Equal)
        || log_terms_match_up_to_abs_subject_for_cancellation(
            ctx,
            normalized_term_expr,
            component_expr,
        )
        || exprs_match_after_default_simplify(ctx, normalized_term_expr, component_expr)
}

pub(super) fn try_match_log_product_power_cancellation_components_side(
    ctx: &mut cas_ast::Context,
    focus_expr: cas_ast::ExprId,
) -> Option<LogPowerProductCancellationComponentsMatch> {
    let factors = flatten_mul_chain(ctx, focus_expr);
    let mut log_factor = None;
    let mut scale_factors = Vec::new();
    for factor in factors {
        if try_extract_log_parts(ctx, factor).is_some() {
            if log_factor.is_some() {
                return None;
            }
            log_factor = Some(factor);
        } else {
            scale_factors.push(factor);
        }
    }

    let log_factor = log_factor?;
    let (log_base, log_arg) = try_extract_log_parts(ctx, log_factor)?;

    let scale = match scale_factors.len() {
        0 => ctx.num(1),
        1 => scale_factors[0],
        _ => build_balanced_mul(ctx, &scale_factors),
    };

    let mut normalized_terms = Vec::new();
    let mut changed_by_power = false;
    collect_log_arg_terms_for_cancellation(
        ctx,
        log_base,
        log_arg,
        scale,
        Sign::Pos,
        &mut normalized_terms,
        &mut changed_by_power,
    )?;

    Some(LogPowerProductCancellationComponentsMatch {
        focus_expr,
        components: normalized_terms,
        changed_by_power,
    })
}

fn try_match_log_product_power_cancellation_side(
    ctx: &mut cas_ast::Context,
    focus_expr: cas_ast::ExprId,
) -> Option<LogPowerProductCancellationMatch> {
    let matched = try_match_log_product_power_cancellation_components_side(ctx, focus_expr)?;
    let (scale, _log_base, log_arg) = try_extract_scaled_log_term_parts(ctx, matched.focus_expr)?;
    let factors = flatten_mul_chain(ctx, matched.focus_expr);
    let log_factor = factors
        .into_iter()
        .find(|factor| try_extract_log_parts(ctx, *factor).is_some())?;
    let raw_expansion = cas_math::logarithm_inverse_support::try_expand_log_auto_rule_expr(
        ctx, log_factor, log_arg,
    )?;
    let raw_focus_after = build_scaled_expr(ctx, scale, raw_expansion.rewritten);
    let focus_after = build_signed_add_expr(ctx, &matched.components);
    let needs_power_split = matched.changed_by_power
        || !exprs_match_for_cancellation(ctx, raw_focus_after, focus_after);

    Some(LogPowerProductCancellationMatch {
        raw_focus_after,
        focus_after,
        components: matched.components,
        needs_power_split,
    })
}

pub(super) fn try_build_direct_log_expansion_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    let uses_only_natural_logs = |ctx: &cas_ast::Context, expr: cas_ast::ExprId| {
        expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Ln])
            && !expr_contains_any_builtin(ctx, expr, &[BuiltinFn::Log, BuiltinFn::Log10])
    };
    let matches_target = |ctx: &mut cas_ast::Context,
                          rewritten: cas_ast::ExprId,
                          target: cas_ast::ExprId,
                          allow_ln_default_simplify: bool| {
        exprs_match_for_cancellation(ctx, rewritten, target)
            || (allow_ln_default_simplify
                && exprs_match_after_default_simplify(ctx, rewritten, target))
    };

    if !expr_contains_log_builtin_for_profile(ctx, lhs_core)
        && !expr_contains_log_builtin_for_profile(ctx, rhs_core)
    {
        return None;
    }

    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if let Some(matched) = try_match_log_product_power_cancellation_side(ctx, source) {
            let allow_ln_default_simplify =
                uses_only_natural_logs(ctx, source) && uses_only_natural_logs(ctx, target);
            if matches_target(ctx, matched.focus_after, target, allow_ln_default_simplify)
                || matches_target(
                    ctx,
                    matched.raw_focus_after,
                    target,
                    allow_ln_default_simplify,
                )
            {
                let mut rewrite =
                    Rewrite::with_local(ctx.num(0), "Log Expansion Identity", source, target)
                        .substep(
                            "Expandir el logaritmo del producto, cociente o potencia",
                            vec![
                                "Tras expandir el logaritmo, ambos lados quedan en la misma forma."
                                    .to_string(),
                            ],
                        );
                if matched.needs_power_split
                    && compare_expr(ctx, matched.raw_focus_after, matched.focus_after)
                        != Ordering::Equal
                {
                    rewrite = rewrite.substep(
                        "Sacar exponentes fuera del logaritmo cuando sea necesario",
                        vec![
                            "La potencia interna se reescribe como un factor exterior equivalente."
                                .to_string(),
                        ],
                    );
                }
                return Some(rewrite);
            }
        }

        if let Some(matched) = try_match_log_abs_mul_div_cancellation_side(ctx, source) {
            let distributed_focus_after = build_signed_add_expr(ctx, &matched.components);
            if exprs_match_for_cancellation(ctx, distributed_focus_after, target)
                || exprs_match_for_cancellation(ctx, matched.focus_after, target)
            {
                return Some(
                    Rewrite::with_local(
                        ctx.num(0),
                        "Log Expansion Identity",
                        source,
                        target,
                    )
                    .substep(
                        "Expandir el logaritmo del producto o del cociente",
                        vec![
                            "Tras expandir el logaritmo con valor absoluto, ambos lados quedan en la misma forma."
                                .to_string(),
                        ],
                    ),
                );
            }
        }
    }

    None
}

pub(super) fn try_build_direct_log_chain_product_equivalence_rewrite(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<Rewrite> {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewritten) = try_rewrite_log_chain_product_expr(ctx, source) else {
            continue;
        };

        if exprs_match_for_cancellation(ctx, rewritten.rewritten, target)
            || exprs_match_after_default_simplify(ctx, rewritten.rewritten, target)
        {
            return Some(Rewrite::with_local(
                ctx.num(0),
                "Log Chain Identity",
                source,
                rewritten.rewritten,
            ));
        }
    }

    None
}

fn log_arg_contains_negative_power(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(base, exponent) => {
            matches!(ctx.get(*exponent), Expr::Number(n) if n < &BigRational::zero())
                || log_arg_contains_negative_power(ctx, *base)
                || log_arg_contains_negative_power(ctx, *exponent)
        }
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) | Expr::Div(lhs, rhs) => {
            log_arg_contains_negative_power(ctx, *lhs) || log_arg_contains_negative_power(ctx, *rhs)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => log_arg_contains_negative_power(ctx, *inner),
        Expr::Function(_, args) => args
            .iter()
            .copied()
            .any(|arg| log_arg_contains_negative_power(ctx, arg)),
        Expr::Matrix { data, .. } => data
            .iter()
            .copied()
            .any(|entry| log_arg_contains_negative_power(ctx, entry)),
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => false,
    }
}

fn log_arg_is_constant_one(ctx: &cas_ast::Context, expr: cas_ast::ExprId) -> bool {
    matches!(ctx.get(expr), Expr::Number(n) if n.is_one())
}

fn log_arg_has_reciprocal_shape_for_default_simplify_reject(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    contains_division_like_term(ctx, expr) || log_arg_contains_negative_power(ctx, expr)
}

fn log_arg_is_bounded_nonreciprocal_shape_for_default_simplify_reject(
    ctx: &cas_ast::Context,
    expr: cas_ast::ExprId,
) -> bool {
    if log_arg_is_constant_one(ctx, expr)
        || log_arg_has_reciprocal_shape_for_default_simplify_reject(ctx, expr)
    {
        return false;
    }

    match ctx.get(expr) {
        Expr::Variable(_) | Expr::SessionRef(_) | Expr::Number(_) | Expr::Constant(_) => true,
        Expr::Mul(lhs, rhs) => {
            log_arg_is_bounded_nonreciprocal_shape_for_default_simplify_reject(ctx, *lhs)
                && log_arg_is_bounded_nonreciprocal_shape_for_default_simplify_reject(ctx, *rhs)
        }
        Expr::Pow(base, exponent) => {
            log_arg_is_bounded_nonreciprocal_shape_for_default_simplify_reject(ctx, *base)
                && matches!(ctx.get(*exponent), Expr::Number(n) if n >= &BigRational::zero())
        }
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, BuiltinFn::Abs) && args.len() == 1 =>
        {
            log_arg_is_bounded_nonreciprocal_shape_for_default_simplify_reject(ctx, args[0])
        }
        _ => false,
    }
}

pub(super) fn reject_negated_log_pair_without_reciprocal_shape_before_default_simplify(
    ctx: &mut cas_ast::Context,
    lhs_core: cas_ast::ExprId,
    rhs_core: cas_ast::ExprId,
) -> Option<bool> {
    let (lhs_expr, lhs_sign) = normalize_signed_add_term(ctx, lhs_core, Sign::Pos);
    let (rhs_expr, rhs_sign) = normalize_signed_add_term(ctx, rhs_core, Sign::Pos);
    if lhs_sign == rhs_sign {
        return None;
    }

    let (lhs_scale, lhs_base, lhs_arg) = try_extract_scaled_log_term_parts(ctx, lhs_expr)?;
    let (rhs_scale, rhs_base, rhs_arg) = try_extract_scaled_log_term_parts(ctx, rhs_expr)?;
    if !exprs_match_for_cancellation(ctx, lhs_scale, rhs_scale)
        || !exprs_match_for_cancellation(ctx, lhs_base, rhs_base)
    {
        return None;
    }

    if log_arg_is_bounded_nonreciprocal_shape_for_default_simplify_reject(ctx, lhs_arg)
        && log_arg_is_bounded_nonreciprocal_shape_for_default_simplify_reject(ctx, rhs_arg)
    {
        Some(false)
    } else {
        None
    }
}
