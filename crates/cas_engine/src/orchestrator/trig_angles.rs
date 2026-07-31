//! Orquestador: familia `trig_angles` (troceo P1).
//!
//! Ver la cabecera de `orchestrator.rs` para el contexto.

use super::*;

fn build_trig_square_double_angle_term_root(ctx: &mut Context, arg: ExprId) -> ExprId {
    let two = ctx.num(2);
    let doubled_arg = smart_mul(ctx, two, arg);
    ctx.call_builtin(BuiltinFn::Sin, vec![doubled_arg])
}

pub(super) fn matches_trig_square_double_angle_term_root(
    ctx: &mut Context,
    expr: ExprId,
    arg: ExprId,
) -> bool {
    let target = build_trig_square_double_angle_term_root(ctx, arg);
    compare_expr(ctx, expr, target) == Ordering::Equal
}

fn build_half_angle_square_target_root(
    ctx: &mut Context,
    trig_fn: BuiltinFn,
    arg: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let double_arg = smart_mul(ctx, two, arg);
    let cos_double_arg = ctx.call_builtin(BuiltinFn::Cos, vec![double_arg]);
    let numerator = match trig_fn {
        BuiltinFn::Sin => ctx.add(Expr::Sub(one, cos_double_arg)),
        BuiltinFn::Cos => ctx.add(Expr::Add(one, cos_double_arg)),
        _ => return ctx.num(0),
    };
    ctx.add(Expr::Div(numerator, two))
}

fn extract_direct_half_angle_square_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *denominator)? != 2 {
        return None;
    }

    let view = AddView::from_expr(ctx, *numerator);
    if view.terms.len() != 2 {
        return None;
    }

    let mut saw_positive_one = false;
    let mut cos_term = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Pos || saw_positive_one {
                return None;
            }
            saw_positive_one = true;
            continue;
        }

        let arg = extract_positive_cos_double_angle_arg_root(ctx, term_expr)?;
        if cos_term.is_some() {
            return None;
        }
        cos_term = Some((arg, term_sign));
    }

    let (arg, cos_sign) = cos_term?;
    if !saw_positive_one {
        return None;
    }

    let trig_fn = match cos_sign {
        Sign::Pos => BuiltinFn::Cos,
        Sign::Neg => BuiltinFn::Sin,
    };
    Some((trig_fn, arg))
}

pub(super) fn matches_direct_half_angle_square_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (half_angle_expr, trig_square_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((trig_fn, half_angle_arg)) =
            extract_direct_half_angle_square_target_root(ctx, half_angle_expr)
        else {
            continue;
        };

        let Some((coeff, trig_name, trig_arg, effective_sign)) =
            extract_signed_numeric_trig_pow2(ctx, trig_square_expr, Sign::Pos)
        else {
            continue;
        };
        if effective_sign != Sign::Pos || coeff != BigRational::one() {
            continue;
        }

        let expected_trig_name = match trig_fn {
            BuiltinFn::Sin => "sin",
            BuiltinFn::Cos => "cos",
            _ => continue,
        };
        if trig_name == expected_trig_name
            && compare_expr(ctx, half_angle_arg, trig_arg) == Ordering::Equal
        {
            return true;
        }
    }

    false
}

pub(super) fn extract_direct_scaled_half_angle_square_target_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut saw_positive_one = false;
    let mut cos_arg = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Pos || saw_positive_one {
                return None;
            }
            saw_positive_one = true;
            continue;
        }

        let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, term_expr) else {
            return None;
        };
        if cos_arg.is_some() {
            return None;
        }
        cos_arg = Some((arg, term_sign));
    }

    let (arg, cos_sign) = cos_arg?;
    if !saw_positive_one {
        return None;
    }

    let trig_fn = match cos_sign {
        Sign::Pos => BuiltinFn::Cos,
        Sign::Neg => BuiltinFn::Sin,
    };
    Some((trig_fn, arg))
}

pub(super) fn matches_direct_scaled_half_angle_square_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (scaled_target_expr, trig_square_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((trig_fn, full_arg)) =
            extract_direct_scaled_half_angle_square_target_root(ctx, scaled_target_expr)
        else {
            continue;
        };

        let Some((coeff, trig_name, trig_arg, effective_sign)) =
            extract_signed_numeric_trig_pow2(ctx, trig_square_expr, Sign::Pos)
        else {
            continue;
        };
        if effective_sign != Sign::Pos || coeff != BigRational::from_integer(2.into()) {
            continue;
        }

        let expected_trig_name = match trig_fn {
            BuiltinFn::Sin => "sin",
            BuiltinFn::Cos => "cos",
            _ => continue,
        };
        if trig_name != expected_trig_name {
            continue;
        }

        let Some(base_arg) = extract_half_scaled_base_root(ctx, trig_arg) else {
            continue;
        };
        if compare_expr(ctx, base_arg, full_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn extract_direct_abs_trig_half_angle_target_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let abs_inner = try_unwrap_abs_arg(ctx, expr)?;
    let (full_angle, is_sin) = extract_trig_half_angle(ctx, abs_inner)?;
    let trig_fn = if is_sin {
        BuiltinFn::Sin
    } else {
        BuiltinFn::Cos
    };
    Some((trig_fn, full_angle))
}

fn extract_direct_sqrt_abs_trig_half_angle_target_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let radicand = extract_unary_builtin_arg_root(ctx, expr, BuiltinFn::Sqrt)?;
    let Expr::Div(numerator, denominator) = ctx.get(radicand) else {
        return None;
    };
    if extract_i64_integer(ctx, *denominator)? != 2 {
        return None;
    }

    let view = AddView::from_expr(ctx, *numerator);
    if view.terms.len() != 2 {
        return None;
    }

    let mut saw_positive_one = false;
    let mut cos_arg = None;
    let mut cos_sign = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Pos || saw_positive_one {
                return None;
            }
            saw_positive_one = true;
            continue;
        }

        let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, term_expr) else {
            return None;
        };
        if cos_arg.is_some() {
            return None;
        }
        cos_arg = Some(arg);
        cos_sign = Some(term_sign);
    }

    let trig_fn = match cos_sign? {
        Sign::Neg => BuiltinFn::Sin,
        Sign::Pos => BuiltinFn::Cos,
    };

    saw_positive_one.then_some((trig_fn, cos_arg?))
}

pub(super) fn build_direct_sqrt_abs_trig_half_angle_target_root(
    ctx: &mut Context,
    trig_fn: BuiltinFn,
    full_arg: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let cos_expr = ctx.call_builtin(BuiltinFn::Cos, vec![full_arg]);
    let numerator = match trig_fn {
        BuiltinFn::Sin => ctx.add(Expr::Sub(one, cos_expr)),
        BuiltinFn::Cos => ctx.add(Expr::Add(one, cos_expr)),
        _ => unreachable!("only sin/cos half-angle absolutes are supported"),
    };
    let radicand = ctx.add(Expr::Div(numerator, two));
    ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand])
}

pub(super) fn matches_direct_abs_trig_half_angle_pair_root(
    ctx: &Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (abs_expr, sqrt_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((abs_fn, abs_arg)) = extract_direct_abs_trig_half_angle_target_root(ctx, abs_expr)
        else {
            continue;
        };
        let Some((sqrt_fn, sqrt_arg)) =
            extract_direct_sqrt_abs_trig_half_angle_target_root(ctx, sqrt_expr)
        else {
            continue;
        };
        if abs_fn == sqrt_fn && compare_expr(ctx, abs_arg, sqrt_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn extract_positive_cos_quadruple_angle_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, expr) else {
        return None;
    };
    let double_angle_arg = extract_double_angle_arg_relaxed(ctx, arg)?;
    extract_double_angle_arg_relaxed(ctx, double_angle_arg)
}

pub(super) fn build_scaled_double_angle_sin_square_root(ctx: &mut Context, arg: ExprId) -> ExprId {
    let two = ctx.num(2);
    let four = ctx.num(4);
    let doubled_arg = smart_mul(ctx, two, arg);
    let sin_double = ctx.call_builtin(BuiltinFn::Sin, vec![doubled_arg]);
    let sin_sq = ctx.add(Expr::Pow(sin_double, two));
    ctx.add(Expr::Div(sin_sq, four))
}

pub(super) fn matches_direct_half_angle_square_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    for index in 0..view.terms.len() {
        let (term_expr, term_sign) = view.terms[index];
        let Some((coeff, trig_name, arg, effective_sign)) =
            extract_signed_numeric_trig_pow2(ctx, term_expr, term_sign)
        else {
            continue;
        };
        if !coeff.is_one() {
            continue;
        }
        let trig_fn = match trig_name {
            "sin" => BuiltinFn::Sin,
            "cos" => BuiltinFn::Cos,
            _ => continue,
        };
        let other_terms: Vec<_> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(other_index, term)| (other_index != index).then_some(term))
            .collect();
        if (other_terms.len() != 1 || other_terms[0].1 != Sign::Neg)
            && (other_terms.len() != 1 || other_terms[0].1 != Sign::Pos)
        {
            continue;
        }
        let target = build_half_angle_square_target_root(ctx, trig_fn, arg);
        let matches_target = compare_expr(ctx, other_terms[0].0, target) == Ordering::Equal;
        if matches_target
            && ((effective_sign == Sign::Pos && other_terms[0].1 == Sign::Neg)
                || (effective_sign == Sign::Neg && other_terms[0].1 == Sign::Pos))
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_half_angle_binomial_square_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    (matches_direct_half_angle_square_zero_identity_root(ctx, lhs_core)
        && matches_direct_trig_binomial_square_zero_identity_root(ctx, rhs_core))
        || (matches_direct_half_angle_square_zero_identity_root(ctx, rhs_core)
            && matches_direct_trig_binomial_square_zero_identity_root(ctx, lhs_core))
}

pub(super) fn matches_direct_trig_product_to_sum_sin_sin_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let lhs_rewrite = try_rewrite_product_to_sum_expr(ctx, lhs_core);
    if lhs_rewrite.is_some_and(|rewrite| {
        rewrite.kind == cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::SinSin
            && render_expr(ctx, rewrite.rewritten) == render_expr(ctx, rhs_core)
    }) {
        return true;
    }

    let rhs_rewrite = try_rewrite_product_to_sum_expr(ctx, rhs_core);
    rhs_rewrite.is_some_and(|rewrite| {
        rewrite.kind == cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::SinSin
            && render_expr(ctx, rewrite.rewritten) == render_expr(ctx, lhs_core)
    })
}

pub(super) fn matches_direct_trig_product_to_sum_sin_sin_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut product_args = None;
    let mut cos_sum_arg = None;
    let mut cos_diff_arg = None;

    for (term_expr, term_sign) in view.terms {
        if term_sign == Sign::Pos {
            if let Some(args) = extract_scaled_trig_sin_sin_product_args_root(ctx, term_expr) {
                if product_args.is_some() {
                    return false;
                }
                product_args = Some(args);
                continue;
            }

            let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, term_expr)
            else {
                return false;
            };
            if cos_sum_arg.is_some() {
                return false;
            }
            cos_sum_arg = Some(arg);
            continue;
        }

        let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, term_expr) else {
            return false;
        };
        if cos_diff_arg.is_some() {
            return false;
        }
        cos_diff_arg = Some(arg);
    }

    let Some((lhs_arg, rhs_arg)) = product_args else {
        return false;
    };
    let Some(cos_sum_arg) = cos_sum_arg else {
        return false;
    };
    let Some(cos_diff_arg) = cos_diff_arg else {
        return false;
    };

    matches_angle_sum_or_diff_arg_root(ctx, cos_sum_arg, lhs_arg, rhs_arg, true)
        && matches_angle_sum_or_diff_arg_root(ctx, cos_diff_arg, lhs_arg, rhs_arg, false)
}

pub(super) fn matches_direct_trig_product_to_sum_cos_cos_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let lhs_rewrite = try_rewrite_product_to_sum_expr(ctx, lhs_core);
    if lhs_rewrite.is_some_and(|rewrite| {
        rewrite.kind == cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::CosCos
            && render_expr(ctx, rewrite.rewritten) == render_expr(ctx, rhs_core)
    }) {
        return true;
    }

    let rhs_rewrite = try_rewrite_product_to_sum_expr(ctx, rhs_core);
    rhs_rewrite.is_some_and(|rewrite| {
        rewrite.kind == cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::CosCos
            && render_expr(ctx, rewrite.rewritten) == render_expr(ctx, lhs_core)
    })
}

pub(super) fn matches_direct_trig_product_to_sum_sin_cos_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    fn matches_sum_target(ctx: &mut Context, product_expr: ExprId, sum_expr: ExprId) -> bool {
        let Some((sin_arg, cos_arg)) =
            extract_scaled_trig_sin_cos_product_args_root(ctx, product_expr)
        else {
            return false;
        };
        let view = AddView::from_expr(ctx, sum_expr);
        if view.terms.len() != 2 {
            return false;
        }

        let mut saw_sum = false;
        let mut saw_diff = false;
        for (term_expr, term_sign) in view.terms {
            let Some((BuiltinFn::Sin, arg)) = extract_plain_sin_or_cos_arg_root(ctx, term_expr)
            else {
                return false;
            };
            match term_sign {
                Sign::Pos
                    if matches_angle_sum_or_diff_arg_root(ctx, arg, sin_arg, cos_arg, true) =>
                {
                    if saw_sum {
                        return false;
                    }
                    saw_sum = true;
                }
                Sign::Pos
                    if matches_angle_sum_or_diff_arg_root(ctx, arg, sin_arg, cos_arg, false) =>
                {
                    if saw_diff {
                        return false;
                    }
                    saw_diff = true;
                }
                Sign::Neg
                    if matches_angle_sum_or_diff_arg_root(ctx, arg, cos_arg, sin_arg, false) =>
                {
                    if saw_diff {
                        return false;
                    }
                    saw_diff = true;
                }
                _ => return false,
            }
        }

        saw_sum && saw_diff
    }

    matches_sum_target(ctx, lhs_core, rhs_core) || matches_sum_target(ctx, rhs_core, lhs_core)
}

pub(super) fn matches_direct_normalized_trig_product_to_sum_sin_cos_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (product_expr, averaged_sum_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(sum_numerator) = extract_div_by_two_numerator_root(ctx, averaged_sum_expr) else {
            continue;
        };
        let two = ctx.num(2);
        let doubled_product = smart_mul(ctx, two, product_expr);
        if matches_direct_trig_product_to_sum_sin_cos_pair_root(ctx, doubled_product, sum_numerator)
        {
            return true;
        }
    }

    false
}

pub(super) fn extract_scaled_double_angle_sin_square_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if let Expr::Div(num, den) = ctx.get(expr) {
        if matches!(ctx.get(*den), Expr::Number(n) if *n == BigRational::new(4.into(), 1.into())) {
            let (coeff, trig_name, arg, effective_sign) =
                extract_signed_numeric_trig_pow2(ctx, *num, Sign::Pos)?;
            if effective_sign == Sign::Pos && coeff == BigRational::one() && trig_name == "sin" {
                return extract_double_angle_arg_relaxed(ctx, arg);
            }
        }
    }

    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let mut saw_quarter = false;
    let mut sin_sq_arg = None;
    for factor in factors {
        match ctx.get(factor) {
            Expr::Number(n) if *n == BigRational::new(1.into(), 4.into()) => {
                if saw_quarter {
                    return None;
                }
                saw_quarter = true;
            }
            _ => {
                let (coeff, trig_name, arg, effective_sign) =
                    extract_signed_numeric_trig_pow2(ctx, factor, Sign::Pos)?;
                if effective_sign != Sign::Pos
                    || coeff != BigRational::one()
                    || trig_name != "sin"
                    || sin_sq_arg.is_some()
                {
                    return None;
                }
                let double_angle_arg = extract_double_angle_arg_relaxed(ctx, arg)?;
                sin_sq_arg = Some(double_angle_arg);
            }
        }
    }

    saw_quarter.then_some(sin_sq_arg?).or(None)
}

pub(super) fn matches_direct_sum_to_product_contraction_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (sum_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_sum_to_product_contraction_expr(ctx, sum_expr) else {
            continue;
        };
        match rewrite.kind {
            cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinSum
            | cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinDiff =>
            {
                let Some((rewritten_sin_arg, rewritten_cos_arg)) =
                    extract_scaled_trig_sin_cos_product_args_root(ctx, rewrite.rewritten)
                else {
                    continue;
                };
                let Some((product_sin_arg, product_cos_arg)) =
                    extract_scaled_trig_sin_cos_product_args_root(ctx, product_expr)
                else {
                    continue;
                };
                if compare_expr(ctx, rewritten_sin_arg, product_sin_arg) == Ordering::Equal
                    && matches_expr_or_negation_root(ctx, rewritten_cos_arg, product_cos_arg)
                {
                    return true;
                }
            }
            cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::CosSum => {
                let Some((rewritten_arg_a, rewritten_arg_b)) =
                    extract_scaled_trig_cos_cos_product_args_root(ctx, rewrite.rewritten)
                else {
                    continue;
                };
                let Some((product_arg_a, product_arg_b)) =
                    extract_scaled_trig_cos_cos_product_args_root(ctx, product_expr)
                else {
                    continue;
                };
                if matches_unordered_cos_arg_pair_up_to_sign_root(
                    ctx,
                    rewritten_arg_a,
                    rewritten_arg_b,
                    product_arg_a,
                    product_arg_b,
                ) {
                    return true;
                }
            }
            _ => {
                if compare_expr(ctx, rewrite.rewritten, product_expr) == Ordering::Equal {
                    return true;
                }
            }
        }
    }

    false
}

fn extract_plain_trig_double_angle_arg_root(
    ctx: &mut Context,
    expr: ExprId,
    trig_fn: BuiltinFn,
) -> Option<ExprId> {
    let (found_fn, arg) = extract_plain_sin_or_cos_arg_root(ctx, expr)?;
    (found_fn == trig_fn)
        .then(|| extract_double_angle_arg_relaxed(ctx, arg))
        .flatten()
}

fn build_trig_product_to_sum_double_angle_difference_target_root(
    ctx: &mut Context,
    arg: ExprId,
) -> ExprId {
    let three = ctx.num(3);
    let triple_arg = smart_mul(ctx, three, arg);
    let sin_triple = ctx.call_builtin(BuiltinFn::Sin, vec![triple_arg]);
    let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    ctx.add(Expr::Sub(sin_triple, sin_arg))
}

fn build_trig_product_to_sum_double_angle_sum_target_root(
    ctx: &mut Context,
    arg: ExprId,
) -> ExprId {
    let three = ctx.num(3);
    let triple_arg = smart_mul(ctx, three, arg);
    let sin_triple = ctx.call_builtin(BuiltinFn::Sin, vec![triple_arg]);
    let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    ctx.add(Expr::Add(sin_triple, sin_arg))
}

pub(super) fn rewrite_direct_trig_product_to_sum_double_angle_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (sin_arg, cos_arg) = extract_scaled_trig_sin_cos_product_args_root(ctx, expr)?;
    let canonical_cos_arg = canonicalize_even_cos_arg_root(ctx, cos_arg);

    if let Some(base_arg) = extract_double_angle_arg_relaxed(ctx, canonical_cos_arg) {
        if compare_expr(ctx, sin_arg, base_arg) == Ordering::Equal {
            return Some(
                build_trig_product_to_sum_double_angle_difference_target_root(ctx, base_arg),
            );
        }
    }

    if let Some(base_arg) = extract_double_angle_arg_relaxed(ctx, sin_arg) {
        if compare_expr(ctx, canonical_cos_arg, base_arg) == Ordering::Equal {
            return Some(build_trig_product_to_sum_double_angle_sum_target_root(
                ctx, base_arg,
            ));
        }
    }

    None
}

pub(super) fn matches_direct_trig_product_to_sum_cos_cos_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut product_args = None;
    let mut plain_cos_args: smallvec::SmallVec<[ExprId; 2]> = smallvec::SmallVec::new();

    for (term_expr, term_sign) in view.terms {
        if term_sign == Sign::Pos {
            let Some(args) = extract_scaled_trig_cos_cos_product_args_root(ctx, term_expr) else {
                return false;
            };
            if product_args.is_some() {
                return false;
            }
            product_args = Some(args);
            continue;
        }

        let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, term_expr) else {
            return false;
        };
        plain_cos_args.push(arg);
    }

    let Some((lhs_arg, rhs_arg)) = product_args else {
        return false;
    };
    if plain_cos_args.len() != 2 {
        return false;
    }

    (matches_angle_sum_or_diff_arg_root(ctx, plain_cos_args[0], lhs_arg, rhs_arg, true)
        && matches_angle_sum_or_diff_arg_root(ctx, plain_cos_args[1], lhs_arg, rhs_arg, false))
        || (matches_angle_sum_or_diff_arg_root(ctx, plain_cos_args[1], lhs_arg, rhs_arg, true)
            && matches_angle_sum_or_diff_arg_root(ctx, plain_cos_args[0], lhs_arg, rhs_arg, false))
}

pub(super) fn matches_direct_trig_product_to_sum_sin_cos_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut product_args = None;
    let mut plain_sin_args: smallvec::SmallVec<[ExprId; 2]> = smallvec::SmallVec::new();

    for (term_expr, term_sign) in view.terms {
        if term_sign == Sign::Pos {
            let Some(args) = extract_scaled_trig_sin_cos_product_args_root(ctx, term_expr) else {
                return false;
            };
            if product_args.is_some() {
                return false;
            }
            product_args = Some(args);
            continue;
        }

        let Some((BuiltinFn::Sin, arg)) = extract_plain_sin_or_cos_arg_root(ctx, term_expr) else {
            return false;
        };
        plain_sin_args.push(arg);
    }

    let Some((sin_arg, cos_arg)) = product_args else {
        return false;
    };
    if plain_sin_args.len() != 2 {
        return false;
    }

    let expected_sum = ctx.add(Expr::Add(sin_arg, cos_arg));
    let expected_diff = ctx.add(Expr::Sub(sin_arg, cos_arg));
    (compare_expr(ctx, plain_sin_args[0], expected_sum) == Ordering::Equal
        && compare_expr(ctx, plain_sin_args[1], expected_diff) == Ordering::Equal)
        || (compare_expr(ctx, plain_sin_args[1], expected_sum) == Ordering::Equal
            && compare_expr(ctx, plain_sin_args[0], expected_diff) == Ordering::Equal)
}

fn extract_unit_pythagorean_complement_pow2_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(&'static str, ExprId)> {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 2 {
        return None;
    }

    let mut saw_one = false;
    let mut trig_term = None;
    for (term_expr, term_sign) in terms {
        if term_sign == Sign::Pos
            && matches!(
                ctx.get(term_expr),
                Expr::Number(n) if *n == BigRational::from_integer(1.into())
            )
        {
            saw_one = true;
            continue;
        }
        if term_sign == Sign::Neg {
            trig_term = extract_unit_trig_pow2_root(ctx, term_expr);
            continue;
        }
        return None;
    }

    saw_one.then_some(())?;
    trig_term
}

pub(super) fn matches_direct_pythagorean_factor_form_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (pow_side, complement_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((pow_name, pow_arg)) = extract_unit_trig_pow2_root(ctx, pow_side) else {
            continue;
        };
        let Some((complement_name, complement_arg)) =
            extract_unit_pythagorean_complement_pow2_root(ctx, complement_side)
        else {
            continue;
        };
        let expected_complement = match pow_name {
            "sin" => "cos",
            "cos" => "sin",
            _ => continue,
        };
        if complement_name == expected_complement
            && compare_expr(ctx, pow_arg, complement_arg) == Ordering::Equal
        {
            return true;
        }
    }
    false
}

pub(super) fn matches_direct_pythagorean_identity_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (sum_side, one_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Expr::Number(n) = ctx.get(one_side) else {
            continue;
        };
        if !n.is_one() {
            continue;
        }

        let terms = AddView::from_expr(ctx, sum_side).terms;
        if terms.len() != 2 {
            continue;
        }

        let mut sin_arg = None;
        let mut cos_arg = None;
        let mut valid = true;
        for (term_expr, term_sign) in terms {
            if term_sign != Sign::Pos {
                valid = false;
                break;
            }

            let Some((name, arg)) = extract_unit_trig_pow2_root(ctx, term_expr) else {
                valid = false;
                break;
            };

            match name {
                "sin" if sin_arg.is_none() => sin_arg = Some(arg),
                "cos" if cos_arg.is_none() => cos_arg = Some(arg),
                _ => {
                    valid = false;
                    break;
                }
            }
        }

        if valid {
            if let (Some(sin_arg), Some(cos_arg)) = (sin_arg, cos_arg) {
                if compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal {
                    return true;
                }
            }
        }
    }

    false
}

fn extract_direct_pythagorean_extended_lhs_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 || !view.terms.iter().all(|(_, sign)| *sign == Sign::Pos) {
        return None;
    }

    let mut sin_arg = None;
    let mut cos_arg = None;
    for (term_expr, _) in view.terms {
        if let Some(arg) = extract_unit_trig_pow_root(ctx, term_expr, BuiltinFn::Sin, 4) {
            if sin_arg.is_some() {
                return None;
            }
            sin_arg = Some(arg);
            continue;
        }
        if let Some(arg) = extract_unit_trig_pow_root(ctx, term_expr, BuiltinFn::Cos, 4) {
            if cos_arg.is_some() {
                return None;
            }
            cos_arg = Some(arg);
            continue;
        }
        return None;
    }

    let (Some(sin_arg), Some(cos_arg)) = (sin_arg, cos_arg) else {
        return None;
    };
    (compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal).then_some(sin_arg)
}

fn extract_direct_pythagorean_extended_rhs_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut saw_one = false;
    let mut product_arg = None;
    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Pos || saw_one {
                return None;
            }
            saw_one = true;
            continue;
        }

        if term_sign != Sign::Neg || product_arg.is_some() {
            return None;
        }

        let factors = flatten_mul_chain(ctx, term_expr);
        let mut numeric_coeff = BigRational::one();
        let mut sin_arg = None;
        let mut cos_arg = None;
        for factor in factors {
            if let Expr::Number(n) = ctx.get(factor) {
                numeric_coeff *= n.clone();
                continue;
            }
            if let Some(arg) = extract_unit_trig_pow_root(ctx, factor, BuiltinFn::Sin, 2) {
                if sin_arg.is_some() {
                    return None;
                }
                sin_arg = Some(arg);
                continue;
            }
            if let Some(arg) = extract_unit_trig_pow_root(ctx, factor, BuiltinFn::Cos, 2) {
                if cos_arg.is_some() {
                    return None;
                }
                cos_arg = Some(arg);
                continue;
            }
            return None;
        }

        let (Some(sin_arg), Some(cos_arg)) = (sin_arg, cos_arg) else {
            return None;
        };
        if numeric_coeff != BigRational::from_integer(2.into())
            || compare_expr(ctx, sin_arg, cos_arg) != Ordering::Equal
        {
            return None;
        }
        product_arg = Some(sin_arg);
    }

    saw_one.then_some(product_arg?).or(None)
}

pub(super) fn matches_direct_pythagorean_extended_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (quartic_side, reduced_side) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(quartic_arg) = extract_direct_pythagorean_extended_lhs_arg_root(ctx, quartic_side)
        else {
            continue;
        };
        let Some(reduced_arg) = extract_direct_pythagorean_extended_rhs_arg_root(ctx, reduced_side)
        else {
            continue;
        };
        if compare_expr(ctx, quartic_arg, reduced_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn extract_positive_cos_double_angle_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, expr) else {
        return None;
    };
    extract_double_angle_arg_relaxed(ctx, arg)
}

pub(super) fn build_positive_cos_double_angle_expr_root(ctx: &mut Context, arg: ExprId) -> ExprId {
    let two = ctx.num(2);
    let doubled_arg = smart_mul(ctx, two, arg);
    ctx.call_builtin(BuiltinFn::Cos, vec![doubled_arg])
}

pub(super) fn matches_direct_quintuple_angle_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let lhs_rewrite = try_rewrite_quintuple_angle_expr(ctx, lhs_core);
    if lhs_rewrite
        .is_some_and(|rewrite| render_expr(ctx, rewrite.rewritten) == render_expr(ctx, rhs_core))
    {
        return true;
    }

    let rhs_rewrite = try_rewrite_quintuple_angle_expr(ctx, rhs_core);
    rhs_rewrite
        .is_some_and(|rewrite| render_expr(ctx, rewrite.rewritten) == render_expr(ctx, lhs_core))
}

pub(super) fn matches_direct_trig_mixed_double_angle_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let lhs_minus_rhs = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if matches_direct_trig_mixed_double_angle_zero_identity_root(ctx, lhs_minus_rhs) {
        return true;
    }

    let rhs_minus_lhs = ctx.add(Expr::Sub(rhs_core, lhs_core));
    matches_direct_trig_mixed_double_angle_zero_identity_root(ctx, rhs_minus_lhs)
}

fn extract_signed_plain_trig_double_angle_arg_root(
    ctx: &mut Context,
    expr: ExprId,
    term_sign: Sign,
    trig_fn: BuiltinFn,
) -> Option<(ExprId, Sign)> {
    let mut numeric_coeff = if term_sign == Sign::Neg {
        -BigRational::one()
    } else {
        BigRational::one()
    };
    let mut trig_arg = None;

    for factor in flatten_mul_chain(ctx, expr) {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }
        let arg = extract_plain_trig_double_angle_arg_root(ctx, factor, trig_fn)?;
        if trig_arg.is_some() {
            return None;
        }
        trig_arg = Some(arg);
    }

    if numeric_coeff.abs() != BigRational::one() {
        return None;
    }

    let effective_sign = if numeric_coeff.is_negative() {
        Sign::Neg
    } else {
        Sign::Pos
    };
    Some((trig_arg?, effective_sign))
}

fn extract_signed_trig_sine_double_angle_product_args_root(
    ctx: &mut Context,
    expr: ExprId,
    term_sign: Sign,
) -> Option<(ExprId, ExprId, Sign)> {
    let mut numeric_coeff = if term_sign == Sign::Neg {
        -BigRational::one()
    } else {
        BigRational::one()
    };
    let mut sin_arg = None;
    let mut cos_arg = None;

    for factor in flatten_mul_chain(ctx, expr) {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }

        match extract_plain_sin_or_cos_arg_root(ctx, factor) {
            Some((BuiltinFn::Sin, arg)) => {
                if sin_arg.is_some() {
                    return None;
                }
                sin_arg = Some(arg);
            }
            Some((BuiltinFn::Cos, arg)) => {
                if cos_arg.is_some() {
                    return None;
                }
                cos_arg = Some(arg);
            }
            _ => return None,
        }
    }

    if numeric_coeff.abs() != BigRational::from_integer(2.into()) {
        return None;
    }

    let (sin_arg, cos_arg) = (sin_arg?, cos_arg?);
    let effective_sign = if numeric_coeff.is_negative() {
        Sign::Neg
    } else {
        Sign::Pos
    };
    Some((
        sin_arg,
        canonicalize_even_cos_arg_root(ctx, cos_arg),
        effective_sign,
    ))
}

pub(super) fn matches_direct_trig_sine_double_angle_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let mut double_angle = None;
    let mut product = None;

    for (term_expr, term_sign) in view.terms {
        if let Some((arg, effective_sign)) = extract_signed_plain_trig_double_angle_arg_root(
            ctx,
            term_expr,
            term_sign,
            BuiltinFn::Sin,
        ) {
            if double_angle.is_some() {
                return false;
            }
            double_angle = Some((arg, effective_sign));
            continue;
        }

        if let Some((sin_arg, cos_arg, effective_sign)) =
            extract_signed_trig_sine_double_angle_product_args_root(ctx, term_expr, term_sign)
        {
            if product.is_some() {
                return false;
            }
            product = Some((sin_arg, cos_arg, effective_sign));
            continue;
        }

        return false;
    }

    let (Some((double_angle_arg, double_angle_sign)), Some((sin_arg, cos_arg, product_sign))) =
        (double_angle, product)
    else {
        return false;
    };

    compare_expr(ctx, double_angle_arg, sin_arg) == Ordering::Equal
        && compare_expr(ctx, double_angle_arg, cos_arg) == Ordering::Equal
        && double_angle_sign == product_sign.negate()
}

pub(super) fn matches_direct_trig_sine_double_angle_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let lhs_minus_rhs = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if matches_direct_trig_sine_double_angle_zero_identity_root(ctx, lhs_minus_rhs) {
        return true;
    }

    let rhs_minus_lhs = ctx.add(Expr::Sub(rhs_core, lhs_core));
    matches_direct_trig_sine_double_angle_zero_identity_root(ctx, rhs_minus_lhs)
}

pub(super) fn matches_direct_sec_tan_pythagorean_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_sec_tan_pythagorean_identity_expr(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, rewrite.rewritten) <= 24
            && cas_ast::count_nodes(ctx, target_expr) <= 24
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                rewrite.rewritten,
                target_expr,
            )
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_tan_to_sec_pythagorean_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_tan_to_sec_pythagorean_identity_expr(ctx, source_expr)
        else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, rewrite.rewritten) <= 24
            && cas_ast::count_nodes(ctx, target_expr) <= 24
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                rewrite.rewritten,
                target_expr,
            )
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_csc_cot_pythagorean_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_csc_cot_pythagorean_identity_expr(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, rewrite.rewritten) <= 24
            && cas_ast::count_nodes(ctx, target_expr) <= 24
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                rewrite.rewritten,
                target_expr,
            )
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_cot_to_csc_pythagorean_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_cot_to_csc_pythagorean_identity_expr(ctx, source_expr)
        else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, rewrite.rewritten) <= 24
            && cas_ast::count_nodes(ctx, target_expr) <= 24
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                rewrite.rewritten,
                target_expr,
            )
        {
            return true;
        }
    }

    false
}

fn extract_numeric_general_phase_shift_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(NumericGeneralPhaseShiftTargetRoot, BigRational, BigRational)> {
    let (coeff, base_expr) = extract_coef_and_base(ctx, expr);
    if coeff.is_zero() {
        return None;
    }
    let (global_sign, coeff_abs) = if coeff < BigRational::zero() {
        (-1_i8, -coeff)
    } else {
        (1_i8, coeff)
    };

    let Expr::Function(fn_id, args) = ctx.get(base_expr) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let trig_fn = if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
        BuiltinFn::Sin
    } else if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
        BuiltinFn::Cos
    } else {
        return None;
    };

    let shifted_arg = args[0];
    let (base_arg, ratio, subtract_shift) = match ctx.get(shifted_arg).clone() {
        Expr::Add(lhs, rhs) => {
            if let Some(ratio) = extract_numeric_atan_ratio_arg_root(ctx, lhs) {
                (rhs, ratio, false)
            } else if let Some(ratio) = extract_numeric_atan_ratio_arg_root(ctx, rhs) {
                (lhs, ratio, false)
            } else {
                return None;
            }
        }
        Expr::Sub(lhs, rhs) => (lhs, extract_numeric_atan_ratio_arg_root(ctx, rhs)?, true),
        _ => return None,
    };

    Some((
        NumericGeneralPhaseShiftTargetRoot {
            trig_fn,
            base_arg,
            subtract_shift,
            global_sign,
        },
        coeff_abs,
        ratio,
    ))
}

fn extract_numeric_general_phase_shift_linear_signature_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational, BigRational, i8, i8)> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut sin_term = None;
    let mut cos_term = None;
    for (term_expr, term_sign) in view.terms {
        let (mut coeff, base_expr) = extract_coef_and_base(ctx, term_expr);
        if term_sign == Sign::Neg {
            coeff = -coeff;
        }
        if coeff.is_zero() {
            return None;
        }

        if let Some(arg) = extract_unary_builtin_arg_root(ctx, base_expr, BuiltinFn::Sin) {
            if sin_term.is_some() {
                return None;
            }
            sin_term = Some((arg, coeff));
            continue;
        } else if let Some(arg) = extract_unary_builtin_arg_root(ctx, base_expr, BuiltinFn::Cos) {
            if cos_term.is_some() {
                return None;
            }
            cos_term = Some((arg, coeff));
            continue;
        } else {
            return None;
        }
    }

    let (sin_arg, sin_coeff_signed) = sin_term?;
    let (cos_arg, cos_coeff_signed) = cos_term?;
    if compare_expr(ctx, sin_arg, cos_arg) != Ordering::Equal {
        return None;
    }

    let (sin_sign, sin_coeff) = if sin_coeff_signed < BigRational::zero() {
        (-1_i8, -sin_coeff_signed)
    } else {
        (1_i8, sin_coeff_signed)
    };
    let (cos_sign, cos_coeff) = if cos_coeff_signed < BigRational::zero() {
        (-1_i8, -cos_coeff_signed)
    } else {
        (1_i8, cos_coeff_signed)
    };

    Some((sin_arg, sin_coeff, cos_coeff, sin_sign, cos_sign))
}

pub(super) fn matches_direct_numeric_general_phase_shift_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (linear_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((arg, sin_coeff, cos_coeff, sin_sign, cos_sign)) =
            extract_numeric_general_phase_shift_linear_signature_root(ctx, linear_expr)
        else {
            continue;
        };
        let Some((target, target_coeff, target_ratio)) =
            extract_numeric_general_phase_shift_target_root(ctx, target_expr)
        else {
            continue;
        };
        if compare_expr(ctx, target.base_arg, arg) != Ordering::Equal {
            continue;
        }

        let (expected_ratio, expected_subtract_shift, expected_global_sign) = match target.trig_fn {
            BuiltinFn::Sin if !sin_coeff.is_zero() => (
                cos_coeff.clone() / sin_coeff.clone(),
                sin_sign != cos_sign,
                sin_sign,
            ),
            BuiltinFn::Cos if !cos_coeff.is_zero() => (
                sin_coeff.clone() / cos_coeff.clone(),
                sin_sign == cos_sign,
                cos_sign,
            ),
            _ => continue,
        };
        if target.subtract_shift != expected_subtract_shift
            || target.global_sign != expected_global_sign
            || target_ratio != expected_ratio
        {
            continue;
        }

        let amplitude_sq = sin_coeff.clone() * sin_coeff + cos_coeff.clone() * cos_coeff;
        let Some(amplitude) = rational_sqrt(&amplitude_sq) else {
            continue;
        };
        if target_coeff == amplitude {
            return true;
        }
    }

    false
}

pub(super) fn extract_special_angle_exact_value_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (builtin_name, arg) = {
        let Expr::Function(fn_id, args) = ctx.get(expr) else {
            return None;
        };
        if args.len() != 1 {
            return None;
        }
        let builtin = ctx.builtin_of(*fn_id)?;
        (builtin.name(), args[0])
    };
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    if profiling {
        crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
            "root.special_angle.lookup_trig_or_inverse",
            render_expr_for_orchestrator_profile(ctx, expr),
        );
        if let Some(hit) =
            run_profiled_root_shortcut("root.special_angle.lookup_trig_or_inverse", || {
                lookup_trig_or_inverse(ctx, builtin_name, arg).map(|hit| hit.value.to_expr(ctx))
            })
        {
            return Some(hit);
        }

        let detected_angle = run_profiled_orchestrator_section(
            "root.special_angle.detect_special_angle",
            None,
            || detect_special_angle(ctx, arg),
        );
        if let Some(angle) = detected_angle {
            if let Some(value) =
                run_profiled_root_shortcut("root.special_angle.lookup_trig_value", || {
                    lookup_trig_value(builtin_name, angle).map(|value| value.to_expr(ctx))
                })
            {
                return Some(value);
            }
        }

        crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
            "root.special_angle.legacy_eval",
            render_expr_for_orchestrator_profile(ctx, expr),
        );
        return run_profiled_root_shortcut("root.special_angle.legacy_eval", || {
            try_rewrite_legacy_evaluate_trig_expr(ctx, expr).map(|rewrite| rewrite.rewritten)
        });
    }

    if let Some(hit) = lookup_trig_or_inverse(ctx, builtin_name, arg) {
        return Some(hit.value.to_expr(ctx));
    }
    if let Some(angle) = detect_special_angle(ctx, arg) {
        if let Some(value) = lookup_trig_value(builtin_name, angle) {
            return Some(value.to_expr(ctx));
        }
    }
    try_rewrite_legacy_evaluate_trig_expr(ctx, expr).map(|rewrite| rewrite.rewritten)
}

fn is_potential_special_angle_exact_factor_source_root(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Neg(inner) => is_potential_special_angle_exact_factor_source_root(ctx, *inner),
        Expr::Function(_, args) if args.len() == 1 => {
            cas_ast::collect_variables(ctx, args[0]).is_empty()
                && cas_ast::count_nodes(ctx, args[0]) <= 16
        }
        _ => false,
    }
}

fn is_definitely_non_special_angle_exact_value_probe_root(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Neg(inner) => is_definitely_non_special_angle_exact_value_probe_root(ctx, *inner),
        Expr::Function(fn_id, args) => {
            args.len() == 1
                && ctx.is_builtin(*fn_id, BuiltinFn::Sqrt)
                && cas_ast::collect_variables(ctx, args[0]).is_empty()
        }
        _ => false,
    }
}

pub(super) fn matches_direct_special_angle_exact_value_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(expected) = extract_special_angle_exact_value_root(ctx, source_expr) else {
            continue;
        };
        let normalized_expected = if cas_ast::count_nodes(ctx, expected) <= 20 {
            isolated_simplify_expr_if_changed(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                expected,
            )
            .unwrap_or(expected)
        } else {
            expected
        };
        let normalized_target = if cas_ast::count_nodes(ctx, target_expr) <= 20 {
            isolated_simplify_expr_if_changed(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                target_expr,
            )
            .unwrap_or(target_expr)
        } else {
            target_expr
        };

        if compare_expr(ctx, normalized_expected, normalized_target) == Ordering::Equal {
            return true;
        }
        if ground_exact_constant_key_root(ctx, normalized_expected).is_some()
            && ground_exact_constant_key_root(ctx, normalized_expected)
                == ground_exact_constant_key_root(ctx, normalized_target)
        {
            return true;
        }
        if cas_ast::count_nodes(ctx, normalized_expected) <= 20
            && cas_ast::count_nodes(ctx, normalized_target) <= 20
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                normalized_expected,
                normalized_target,
            )
        {
            return true;
        }
    }

    false
}

pub(super) fn rewrite_direct_double_angle_inverse_trig_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Some((BuiltinFn::Sin, doubled_arg)) = extract_plain_sin_or_cos_arg_root(ctx, expr) else {
        return None;
    };
    let inner_arg = extract_double_angle_arg_relaxed(ctx, doubled_arg)?;

    if let Expr::Function(fn_id, args) = ctx.get(inner_arg) {
        if args.len() == 1
            && (ctx.is_builtin(*fn_id, BuiltinFn::Arcsin)
                || ctx.is_builtin(*fn_id, BuiltinFn::Arccos))
        {
            let inner = args[0];
            let one = ctx.num(1);
            let two = ctx.num(2);
            let inner_sq = ctx.add(Expr::Pow(inner, two));
            let radicand = ctx.add(Expr::Sub(one, inner_sq));
            let sqrt_factor = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
            let two = ctx.num(2);
            return Some(build_mul_expr_from_factors_root(
                ctx,
                &[two, inner, sqrt_factor],
            ));
        }
    }

    let sin_inner = ctx.call_builtin(BuiltinFn::Sin, vec![inner_arg]);
    let cos_inner = ctx.call_builtin(BuiltinFn::Cos, vec![inner_arg]);

    let rewritten_sin = if let Some(plan) =
        cas_math::inverse_trig_composition_support::try_plan_inverse_trig_composition_expr(
            ctx, sin_inner, false, false,
        ) {
        strip_multiplicative_one_root(ctx, plan.rewritten)
    } else if let Some(rewrite) = try_rewrite_trig_inverse_composition_expr(ctx, sin_inner) {
        strip_multiplicative_one_root(ctx, rewrite.rewritten)
    } else {
        return None;
    };

    let rewritten_cos = if let Some(plan) =
        cas_math::inverse_trig_composition_support::try_plan_inverse_trig_composition_expr(
            ctx, cos_inner, false, false,
        ) {
        strip_multiplicative_one_root(ctx, plan.rewritten)
    } else if let Some(rewrite) = try_rewrite_trig_inverse_composition_expr(ctx, cos_inner) {
        strip_multiplicative_one_root(ctx, rewrite.rewritten)
    } else {
        return None;
    };

    let two = ctx.num(2);
    Some(build_mul_expr_from_factors_root(
        ctx,
        &[two, rewritten_sin, rewritten_cos],
    ))
}

pub(super) fn matches_direct_double_angle_inverse_trig_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewritten) =
            rewrite_direct_double_angle_inverse_trig_target_root(ctx, source_expr)
        else {
            continue;
        };
        let rewritten = strip_multiplicative_one_root(ctx, rewritten);
        let target = strip_multiplicative_one_root(ctx, target_expr);
        if compare_expr(ctx, rewritten, target) == Ordering::Equal {
            return true;
        }
        if cas_ast::count_nodes(ctx, rewritten) <= 32
            && cas_ast::count_nodes(ctx, target) <= 32
            && isolated_simplify_rewrites_to_target(
                &crate::phase::SimplifyOptions::default(),
                ctx,
                rewritten,
                target,
            )
        {
            return true;
        }
        let difference = ctx.add(Expr::Sub(rewritten, target));
        if cas_ast::count_nodes(ctx, difference) <= 48
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

pub(super) fn matches_direct_pure_double_angle_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (double_angle_expr, product_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((BuiltinFn::Sin, doubled_arg)) =
            extract_plain_sin_or_cos_arg_root(ctx, double_angle_expr)
        else {
            continue;
        };
        let Some(base_arg) = extract_double_angle_arg_relaxed(ctx, doubled_arg) else {
            continue;
        };
        let Some((sin_arg, cos_arg)) =
            extract_scaled_trig_sin_cos_product_args_root(ctx, product_expr)
        else {
            continue;
        };
        let canonical_cos_arg = canonicalize_even_cos_arg_root(ctx, cos_arg);
        if compare_expr(ctx, base_arg, sin_arg) == Ordering::Equal
            && compare_expr(ctx, base_arg, canonical_cos_arg) == Ordering::Equal
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_trig_phase_shift_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    fn matches_cos_even_target(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> bool {
        let Some((BuiltinFn::Cos, lhs_arg)) = extract_plain_sin_or_cos_arg_root(ctx, lhs) else {
            return false;
        };
        let Some((BuiltinFn::Cos, rhs_arg)) = extract_plain_sin_or_cos_arg_root(ctx, rhs) else {
            return false;
        };
        let neg_lhs_arg = ctx.add(Expr::Neg(lhs_arg));
        let neg_rhs_arg = ctx.add(Expr::Neg(rhs_arg));

        compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal
            || compare_expr(ctx, neg_lhs_arg, rhs_arg) == Ordering::Equal
            || compare_expr(ctx, lhs_arg, neg_rhs_arg) == Ordering::Equal
    }

    fn matches_negated_cos_even_target(ctx: &mut Context, lhs: ExprId, rhs: ExprId) -> bool {
        let (Expr::Neg(lhs_inner), Expr::Neg(rhs_inner)) = (ctx.get(lhs), ctx.get(rhs)) else {
            return false;
        };
        matches_cos_even_target(ctx, *lhs_inner, *rhs_inner)
    }

    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_trig_phase_shift_function_expr(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal
            || matches_cos_even_target(ctx, rewrite.rewritten, target_expr)
            || matches_negated_cos_even_target(ctx, rewrite.rewritten, target_expr)
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_trig_triple_angle_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_triple_angle_expr(ctx, source_expr) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target_expr) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_angle_sum_or_diff_arg_root(
    ctx: &mut Context,
    angle_arg: ExprId,
    lhs_arg: ExprId,
    rhs_arg: ExprId,
    is_sum: bool,
) -> bool {
    let direct_candidate = if is_sum {
        ctx.add(Expr::Add(lhs_arg, rhs_arg))
    } else {
        ctx.add(Expr::Sub(lhs_arg, rhs_arg))
    };
    if compare_expr(ctx, angle_arg, direct_candidate) == Ordering::Equal {
        return true;
    }

    if is_sum {
        let reversed_candidate = ctx.add(Expr::Add(rhs_arg, lhs_arg));
        if compare_expr(ctx, angle_arg, reversed_candidate) == Ordering::Equal {
            return true;
        }
    } else {
        let reversed_candidate = ctx.add(Expr::Sub(rhs_arg, lhs_arg));
        if compare_expr(ctx, angle_arg, reversed_candidate) == Ordering::Equal {
            return true;
        }
    }

    let Some((base, lhs_coeff, rhs_coeff)) = extract_linear_coefficients(ctx, lhs_arg, rhs_arg)
    else {
        return false;
    };
    let expected_coeff = if is_sum {
        lhs_coeff.clone() + rhs_coeff.clone()
    } else {
        lhs_coeff.clone() - rhs_coeff.clone()
    };
    let expected_arg = build_coef_times_base(ctx, &expected_coeff, base);
    if compare_expr(ctx, angle_arg, expected_arg) == Ordering::Equal {
        return true;
    }

    if is_sum {
        return false;
    }

    let reversed_coeff = rhs_coeff - lhs_coeff;
    let reversed_arg = build_coef_times_base(ctx, &reversed_coeff, base);
    compare_expr(ctx, angle_arg, reversed_arg) == Ordering::Equal
}

pub(super) fn extract_direct_tan_angle_sum_target_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let sum_arg = extract_unary_builtin_arg_root(ctx, expr, BuiltinFn::Tan)?;
    let Expr::Add(lhs_arg, rhs_arg) = ctx.get(sum_arg) else {
        return None;
    };
    Some((*lhs_arg, *rhs_arg))
}

pub(super) fn build_tan_angle_sum_fraction_root(
    ctx: &mut Context,
    lhs_arg: ExprId,
    rhs_arg: ExprId,
) -> ExprId {
    let tan_lhs = ctx.call_builtin(BuiltinFn::Tan, vec![lhs_arg]);
    let tan_rhs = ctx.call_builtin(BuiltinFn::Tan, vec![rhs_arg]);
    let numerator = ctx.add(Expr::Add(tan_lhs, tan_rhs));
    let one = ctx.num(1);
    let tan_product = build_mul_expr_from_factors_root(ctx, &[tan_lhs, tan_rhs]);
    let denominator = ctx.add(Expr::Sub(one, tan_product));
    ctx.add(Expr::Div(numerator, denominator))
}

fn extract_direct_tan_angle_sum_fraction_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    let numerator = *numerator;
    let denominator = *denominator;
    let (num_lhs, num_rhs) = extract_direct_tangent_addition_target_root(ctx, numerator)?;

    let view = AddView::from_expr(ctx, denominator);
    if view.terms.len() != 2 {
        return None;
    }

    let mut saw_positive_one = false;
    let mut product_args = None;
    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Pos || saw_positive_one {
                return None;
            }
            saw_positive_one = true;
            continue;
        }
        if term_sign != Sign::Neg || product_args.is_some() {
            return None;
        }
        product_args = extract_plain_trig_product_pair_args_root(ctx, term_expr, BuiltinFn::Tan);
        product_args?;
    }

    let (prod_lhs, prod_rhs) = product_args?;
    if !saw_positive_one
        || !matches_unordered_expr_pair_root(ctx, num_lhs, num_rhs, prod_lhs, prod_rhs)
    {
        return None;
    }

    Some((num_lhs, num_rhs))
}

pub(super) fn matches_direct_tan_angle_sum_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (tan_expr, fraction_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some((tan_lhs, tan_rhs)) = extract_direct_tan_angle_sum_target_root(ctx, tan_expr)
        else {
            continue;
        };
        let Some((fraction_lhs, fraction_rhs)) =
            extract_direct_tan_angle_sum_fraction_target_root(ctx, fraction_expr)
        else {
            continue;
        };
        if matches_unordered_expr_pair_root(ctx, tan_lhs, tan_rhs, fraction_lhs, fraction_rhs) {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_mixed_pythagorean_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 4 {
        return false;
    }

    let mut sin_term = None;
    let mut cos_term = None;
    let mut cosh_term = None;
    let mut sinh_term = None;

    for (term_expr, term_sign) in view.terms {
        if let Some((name, arg)) = extract_unit_trig_pow2_root(ctx, term_expr) {
            match name {
                "sin" if sin_term.is_none() => sin_term = Some((arg, term_sign)),
                "cos" if cos_term.is_none() => cos_term = Some((arg, term_sign)),
                _ => return false,
            }
            continue;
        }

        match extract_plain_sinh_or_cosh_pow2_arg_root(ctx, term_expr) {
            Some((BuiltinFn::Cosh, arg)) if cosh_term.is_none() => {
                cosh_term = Some((arg, term_sign))
            }
            Some((BuiltinFn::Sinh, arg)) if sinh_term.is_none() => {
                sinh_term = Some((arg, term_sign))
            }
            _ => return false,
        }
    }

    let (
        Some((sin_arg, sin_sign)),
        Some((cos_arg, cos_sign)),
        Some((cosh_arg, cosh_sign)),
        Some((sinh_arg, sinh_sign)),
    ) = (sin_term, cos_term, cosh_term, sinh_term)
    else {
        return false;
    };

    if compare_expr(ctx, sin_arg, cos_arg) != Ordering::Equal
        || compare_expr(ctx, cosh_arg, sinh_arg) != Ordering::Equal
        || sin_sign != cos_sign
        || cosh_sign == sinh_sign
    {
        return false;
    }

    matches!(
        (sin_sign, cosh_sign, sinh_sign),
        (Sign::Neg, Sign::Pos, Sign::Neg) | (Sign::Pos, Sign::Neg, Sign::Pos)
    )
}

fn extract_scaled_cos_double_angle_sine_term_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut sin_arg = None;
    let mut cos_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }
        match extract_plain_sin_or_cos_arg_root(ctx, factor) {
            Some((BuiltinFn::Sin, arg)) => sin_arg = Some(arg),
            Some((BuiltinFn::Cos, arg)) => cos_arg = Some(arg),
            _ => return None,
        }
    }

    if numeric_coeff != BigRational::from_integer(2.into()) {
        return None;
    }
    let sin_arg = sin_arg?;
    let cos_arg = cos_arg?;
    let two = ctx.num(2);
    let doubled_sin_arg = smart_mul(ctx, two, sin_arg);
    (compare_expr(ctx, cos_arg, doubled_sin_arg) == Ordering::Equal).then_some(sin_arg)
}

pub(super) fn extract_scaled_sin_double_angle_sine_term_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut sin_args: smallvec::SmallVec<[ExprId; 2]> = smallvec::SmallVec::new();

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }
        let Some((BuiltinFn::Sin, arg)) = extract_plain_sin_or_cos_arg_root(ctx, factor) else {
            return None;
        };
        sin_args.push(arg);
    }

    if numeric_coeff != BigRational::from_integer(2.into()) || sin_args.len() != 2 {
        return None;
    }

    for &candidate_u in &sin_args {
        let two = ctx.num(2);
        let doubled_candidate_u = smart_mul(ctx, two, candidate_u);
        if sin_args
            .iter()
            .any(|arg| compare_expr(ctx, *arg, doubled_candidate_u) == Ordering::Equal)
        {
            return Some(candidate_u);
        }
    }

    None
}

pub(super) fn matches_narrow_trig_mixed_double_angle_zero_candidate_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    view.terms.len() == 3
        && view.terms.iter().any(|(term_expr, _)| {
            extract_scaled_cos_double_angle_sine_term_arg_root(ctx, *term_expr).is_some()
        })
}

pub(super) fn matches_direct_trig_mixed_double_angle_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut double_angle = None;
    let mut linear_sine = None;
    let mut cos_square_sine = None;

    for (term_expr, term_sign) in view.terms {
        if let Some(arg) = extract_scaled_cos_double_angle_sine_term_arg_root(ctx, term_expr) {
            if double_angle.is_some() {
                return false;
            }
            double_angle = Some((arg, term_sign));
            continue;
        }

        if let Some(arg) = extract_scaled_plain_sine_term_arg_root(ctx, term_expr) {
            if linear_sine.is_some() {
                return false;
            }
            linear_sine = Some((arg, term_sign));
            continue;
        }

        if let Some(arg) = extract_scaled_cos_square_sine_term_arg_root(ctx, term_expr) {
            if cos_square_sine.is_some() {
                return false;
            }
            cos_square_sine = Some((arg, term_sign));
            continue;
        }

        return false;
    }

    let (
        Some((double_arg, double_sign)),
        Some((linear_arg, linear_sign)),
        Some((square_arg, square_sign)),
    ) = (double_angle, linear_sine, cos_square_sine)
    else {
        return false;
    };

    compare_expr(ctx, double_arg, linear_arg) == Ordering::Equal
        && compare_expr(ctx, double_arg, square_arg) == Ordering::Equal
        && double_sign == linear_sign
        && square_sign == double_sign.negate()
}

pub(super) fn matches_direct_nested_zero_pure_double_angle_residual_pair_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let matches_pair = |ctx: &mut Context, lhs: ExprId, rhs: ExprId| {
        matches_direct_pure_double_angle_pair_root(ctx, lhs, rhs)
    };

    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) if matches_pair(ctx, lhs, rhs) => return true,
        Expr::Add(lhs, rhs) => {
            if let Expr::Neg(inner) = ctx.get(rhs) {
                if matches_pair(ctx, lhs, *inner) {
                    return true;
                }
            }
        }
        _ => {}
    }

    extract_shared_additive_passthrough_sub_cores_root(ctx, expr)
        .is_some_and(|(lhs_core, rhs_core)| matches_pair(ctx, lhs_core, rhs_core))
}

fn matches_direct_signed_pure_double_angle_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (lhs_expr, rhs_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let (lhs_coeff, lhs_base) = extract_coef_and_base(ctx, lhs_expr);
        let (rhs_coeff, rhs_base) = extract_coef_and_base(ctx, rhs_expr);
        if lhs_coeff == rhs_coeff
            && matches_direct_pure_double_angle_pair_root(ctx, lhs_base, rhs_base)
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_nested_zero_signed_pure_double_angle_residual_pair_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let matches_pair = |ctx: &mut Context, lhs: ExprId, rhs: ExprId| {
        matches_direct_signed_pure_double_angle_pair_root(ctx, lhs, rhs)
    };

    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) if matches_pair(ctx, lhs, rhs) => return true,
        Expr::Add(lhs, rhs) => {
            if let Expr::Neg(inner) = ctx.get(rhs) {
                if matches_pair(ctx, lhs, *inner) {
                    return true;
                }
            }
        }
        _ => {}
    }

    extract_shared_additive_passthrough_sub_cores_root(ctx, expr)
        .is_some_and(|(lhs_core, rhs_core)| matches_pair(ctx, lhs_core, rhs_core))
}

fn extract_half_angle_tan_numerator_arg_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut has_one = false;
    let mut negative_cos_arg = None;
    for (term_expr, term_sign) in view.terms {
        match (ctx.get(term_expr), term_sign) {
            (Expr::Number(n), Sign::Pos) if n.is_one() => has_one = true,
            (_, Sign::Neg) => {
                negative_cos_arg = extract_positive_cos_double_angle_arg_root(ctx, term_expr);
            }
            _ => return None,
        }
    }

    (has_one).then_some(negative_cos_arg?).filter(|_| has_one)
}

pub(super) fn matches_direct_half_angle_tan_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Expr::Div(numerator, denominator) = ctx.get(source).clone() else {
            continue;
        };
        let Some(target_arg) = extract_unary_builtin_arg_root(ctx, target, BuiltinFn::Tan) else {
            continue;
        };
        let Some(num_arg) = extract_half_angle_tan_numerator_arg_root(ctx, numerator) else {
            continue;
        };
        let Some((BuiltinFn::Sin, den_arg)) = extract_plain_sin_or_cos_arg_root(ctx, denominator)
        else {
            continue;
        };
        let Some(den_base_arg) = extract_double_angle_arg_relaxed(ctx, den_arg) else {
            continue;
        };

        if compare_expr(ctx, num_arg, target_arg) == Ordering::Equal
            && compare_expr(ctx, den_base_arg, target_arg) == Ordering::Equal
        {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_nested_zero_half_angle_tan_residual_pair_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let matches_pair = |ctx: &mut Context, lhs: ExprId, rhs: ExprId| {
        matches_direct_half_angle_tan_pair_root(ctx, lhs, rhs)
    };

    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) if matches_pair(ctx, lhs, rhs) => return true,
        Expr::Add(lhs, rhs) => {
            if let Expr::Neg(inner) = ctx.get(rhs) {
                if matches_pair(ctx, lhs, *inner) {
                    return true;
                }
            }
        }
        _ => {}
    }

    extract_shared_additive_passthrough_sub_cores_root(ctx, expr)
        .is_some_and(|(lhs_core, rhs_core)| matches_pair(ctx, lhs_core, rhs_core))
}

pub(super) fn matches_direct_general_phase_shift_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 || !expr_contains_trig_builtin_local(ctx, expr) {
        return false;
    }

    let extract_shifted_base_arg = |ctx: &mut Context, expr: ExprId| {
        let (_trig_fn, arg) = extract_plain_sin_or_cos_arg_root(ctx, expr)?;
        let arg_terms = AddView::from_expr(ctx, arg).terms;
        if arg_terms.len() != 2 {
            return None;
        }
        let positive_terms: smallvec::SmallVec<[ExprId; 2]> = arg_terms
            .iter()
            .filter_map(|(term, sign)| (*sign == Sign::Pos).then_some(*term))
            .collect();
        (positive_terms.len() == 1).then_some(positive_terms[0])
    };

    let mut plain_sin_arg = None;
    let mut plain_cos_arg = None;
    let mut shifted_base_arg = None;

    for (term_expr, _term_sign) in view.terms {
        let (_coeff, base) = extract_coef_and_base(ctx, term_expr);
        if let Some((trig_fn, arg)) = extract_plain_sin_or_cos_arg_root(ctx, base) {
            if let Some(candidate_base_arg) = extract_shifted_base_arg(ctx, base) {
                shifted_base_arg = Some(candidate_base_arg);
            } else if trig_fn == BuiltinFn::Sin {
                plain_sin_arg = Some(arg);
            } else if trig_fn == BuiltinFn::Cos {
                plain_cos_arg = Some(arg);
            }
        }
    }

    let (Some(plain_sin_arg), Some(plain_cos_arg), Some(shifted_base_arg)) =
        (plain_sin_arg, plain_cos_arg, shifted_base_arg)
    else {
        return false;
    };
    if compare_expr(ctx, plain_sin_arg, plain_cos_arg) != Ordering::Equal
        || compare_expr(ctx, plain_sin_arg, shifted_base_arg) != Ordering::Equal
    {
        return false;
    }

    let parent_ctx = crate::ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let rule = crate::rules::arithmetic::ExpandTrigPhaseShiftToEnableCancellationRule;
    let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) else {
        return false;
    };
    let zero = ctx.num(0);
    compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal
}

pub(super) fn matches_direct_numeric_general_phase_shift_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 || !expr_contains_trig_builtin_local(ctx, expr) {
        return false;
    }

    for candidate_index in 0..view.terms.len() {
        let (candidate_expr, candidate_sign) = normalize_signed_add_term_root(
            ctx,
            view.terms[candidate_index].0,
            view.terms[candidate_index].1,
        );
        let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 3]> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != candidate_index).then_some(term))
            .collect();
        let normalized_remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 3]> =
            if candidate_sign == Sign::Neg {
                remaining_terms
            } else {
                remaining_terms
                    .iter()
                    .map(|(term_expr, term_sign)| (*term_expr, flip_add_sign_root(*term_sign)))
                    .collect()
            };
        let remaining_expr = build_signed_sum_expr_root(ctx, &normalized_remaining_terms);
        if matches_direct_numeric_general_phase_shift_pair_root(ctx, remaining_expr, candidate_expr)
        {
            return true;
        }
    }

    false
}

pub(super) fn is_potential_direct_three_term_phase_shift_zero_subset_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3
        || !expr_contains_trig_builtin_local(ctx, expr)
        || (!expr_contains_any_builtin_local(ctx, expr, &[BuiltinFn::Atan, BuiltinFn::Arctan])
            && !expr_contains_pi_constant_local(ctx, expr))
    {
        return false;
    }
    true
}

pub(super) fn matches_direct_three_term_phase_shift_pair_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 || !expr_contains_trig_builtin_local(ctx, expr) {
        return false;
    }

    for candidate_index in 0..view.terms.len() {
        let (candidate_expr, candidate_sign) = normalize_signed_add_term_root(
            ctx,
            view.terms[candidate_index].0,
            view.terms[candidate_index].1,
        );
        let candidate_has_phase_shift_signal = expr_contains_pi_constant_local(ctx, candidate_expr)
            || expr_contains_any_builtin_local(
                ctx,
                candidate_expr,
                &[BuiltinFn::Atan, BuiltinFn::Arctan],
            );
        if !expr_contains_trig_builtin_local(ctx, candidate_expr)
            || !candidate_has_phase_shift_signal
        {
            continue;
        }

        let remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 3]> = view
            .terms
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, term)| (index != candidate_index).then_some(term))
            .collect();
        let normalized_remaining_terms: smallvec::SmallVec<[(ExprId, Sign); 3]> =
            if candidate_sign == Sign::Neg {
                remaining_terms
            } else {
                remaining_terms
                    .iter()
                    .map(|(term_expr, term_sign)| (*term_expr, flip_add_sign_root(*term_sign)))
                    .collect()
            };
        let remaining_expr = build_signed_sum_expr_root(ctx, &normalized_remaining_terms);
        if crate::rules::arithmetic::matches_trig_phase_shift_cancellation_pair(
            ctx,
            remaining_expr,
            candidate_expr,
            false,
        ) {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_three_term_phase_shift_zero_subset_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    if !is_potential_direct_three_term_phase_shift_zero_subset_root(ctx, expr) {
        return false;
    }

    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    if matches_direct_three_term_phase_shift_pair_zero_identity_root(ctx, expr) {
        if profiling {
            run_profiled_root_shortcut("root.div.03g1a0.phase_shift_zero.fast_pair", || Some(expr));
        }
        return true;
    }

    let rewrite = if profiling {
        run_profiled_orchestrator_section(
            "root.div.03g1a.phase_shift_zero.exact_scope_rewrite",
            None,
            || {
                crate::rules::arithmetic::try_build_exact_trig_phase_shift_zero_scope_rewrite(
                    ctx, expr,
                )
            },
        )
    } else {
        crate::rules::arithmetic::try_build_exact_trig_phase_shift_zero_scope_rewrite(ctx, expr)
    };
    let Some(rewrite) = rewrite else {
        return false;
    };
    if profiling {
        let description_label = "root.div.03g1b.phase_shift_zero.phase_shift_identity_rewrite";
        run_profiled_root_shortcut(description_label, || Some(rewrite.new_expr));
        let mode_label = match rewrite
            .substeps
            .first()
            .map(|substep| substep.title.as_str())
        {
            Some("Phase Shift Identity") => "root.div.03g1c.phase_shift_zero.fast_structural",
            Some("Reescribir la combinación lineal") => {
                "root.div.03g1d.phase_shift_zero.linear_to_shifted"
            }
            Some("Expandir el término desplazado") => {
                "root.div.03g1e.phase_shift_zero.shifted_to_linear"
            }
            Some("Reescribir el término desplazado") => {
                "root.div.03g1f.phase_shift_zero.shifted_to_shifted"
            }
            _ => "root.div.03g1g.phase_shift_zero.unknown_mode",
        };
        run_profiled_root_shortcut(mode_label, || Some(rewrite.new_expr));
    }
    let zero = ctx.num(0);
    compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal
}

pub(super) fn matches_direct_sec_tan_pythagorean_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
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

            let focus_expr = build_signed_sum_expr_root(ctx, &focus_terms);
            let Some(rewrite) = try_rewrite_sec_tan_pythagorean_identity_expr(ctx, focus_expr)
            else {
                continue;
            };
            let one = ctx.num(1);
            if compare_expr(ctx, rewrite.rewritten, one) != Ordering::Equal {
                continue;
            }

            let (remaining_expr, remaining_sign) =
                normalize_signed_add_term_root(ctx, remaining_terms[0].0, remaining_terms[0].1);
            if remaining_sign == Sign::Neg && extract_i64_integer(ctx, remaining_expr) == Some(1) {
                return true;
            }
        }
    }

    false
}

pub(super) fn matches_direct_csc_cot_pythagorean_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
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

            let focus_expr = build_signed_sum_expr_root(ctx, &focus_terms);
            let Some(rewrite) = try_rewrite_csc_cot_pythagorean_identity_expr(ctx, focus_expr)
            else {
                continue;
            };
            let one = ctx.num(1);
            if compare_expr(ctx, rewrite.rewritten, one) != Ordering::Equal {
                continue;
            }

            let (remaining_expr, remaining_sign) =
                normalize_signed_add_term_root(ctx, remaining_terms[0].0, remaining_terms[0].1);
            if remaining_sign == Sign::Neg && extract_i64_integer(ctx, remaining_expr) == Some(1) {
                return true;
            }
        }
    }

    false
}

pub(super) fn matches_direct_symbolic_trig_sum_to_product_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 || !expr_contains_trig_builtin_local(ctx, expr) {
        return false;
    }

    let mut plain_term_count = 0usize;
    let mut product_term_count = 0usize;
    for (term_expr, _term_sign) in view.terms {
        let (_coeff, base) = extract_coef_and_base(ctx, term_expr);
        if extract_plain_sin_or_cos_arg_root(ctx, base).is_some() {
            plain_term_count += 1;
            continue;
        }

        let trig_factor_count = flatten_mul_chain(ctx, base)
            .into_iter()
            .filter(|factor| extract_plain_sin_or_cos_arg_root(ctx, *factor).is_some())
            .count();
        if trig_factor_count >= 2 {
            product_term_count += 1;
            continue;
        }

        return false;
    }

    if plain_term_count != 2 || product_term_count != 1 {
        return false;
    }

    let parent_ctx = crate::ParentContext::root().with_domain_mode(crate::DomainMode::Generic);
    let rule = crate::rules::arithmetic::ExpandTrigSumToProductToEnableCancellationRule;
    let Some(rewrite) = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx) else {
        return false;
    };
    let zero = ctx.num(0);
    compare_expr(ctx, rewrite.final_expr(), zero) == Ordering::Equal
}

pub(super) fn matches_direct_pythagorean_extended_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Sub(lhs, rhs) => matches_direct_pythagorean_extended_pair_root(ctx, lhs, rhs),
        Expr::Add(lhs, rhs) => {
            let Some((pos, neg)) = (match (ctx.get(lhs), ctx.get(rhs)) {
                (Expr::Neg(inner), _) => Some((rhs, *inner)),
                (_, Expr::Neg(inner)) => Some((lhs, *inner)),
                _ => None,
            }) else {
                return false;
            };
            matches_direct_pythagorean_extended_pair_root(ctx, pos, neg)
        }
        _ => false,
    }
}

fn matches_product_to_sum_sin_cos_factor_pair_direction_root(
    ctx: &mut Context,
    source_factors: &[ExprId],
    target_factors: &[ExprId],
) -> bool {
    if source_factors.len() < 4 || target_factors.is_empty() {
        return false;
    }

    for first in 0..source_factors.len().saturating_sub(2) {
        for second in (first + 1)..source_factors.len().saturating_sub(1) {
            for third in (second + 1)..source_factors.len() {
                let trig_subset = build_mul_expr_from_factors_root(
                    ctx,
                    &[
                        source_factors[first],
                        source_factors[second],
                        source_factors[third],
                    ],
                );

                for (target_index, target_factor) in target_factors.iter().copied().enumerate() {
                    if !matches_direct_trig_product_to_sum_sin_cos_pair_root(
                        ctx,
                        trig_subset,
                        target_factor,
                    ) {
                        continue;
                    }

                    let remaining_source: Vec<_> = source_factors
                        .iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(index, factor)| {
                            (index != first && index != second && index != third).then_some(factor)
                        })
                        .collect();
                    let remaining_target: Vec<_> = target_factors
                        .iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(index, factor)| (index != target_index).then_some(factor))
                        .collect();

                    if remaining_source.is_empty() || remaining_target.is_empty() {
                        continue;
                    }

                    let source_partner = build_mul_expr_from_factors_root(ctx, &remaining_source);
                    let target_partner = build_mul_expr_from_factors_root(ctx, &remaining_target);
                    if factors_match_by_equality_or_direct_pair_root(
                        ctx,
                        source_partner,
                        target_partner,
                    ) {
                        return true;
                    }
                }
            }
        }
    }

    false
}

pub(super) fn matches_direct_product_to_sum_sin_cos_factor_pair_zero_difference_root(
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

    let lhs_factors = flatten_mul_chain(ctx, lhs_term);
    let rhs_factors = flatten_mul_chain(ctx, rhs_term);
    matches_product_to_sum_sin_cos_factor_pair_direction_root(ctx, &lhs_factors, &rhs_factors)
        || matches_product_to_sum_sin_cos_factor_pair_direction_root(
            ctx,
            &rhs_factors,
            &lhs_factors,
        )
}

pub(super) fn try_standard_direct_half_angle_square_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (trig_fn, arg) = extract_direct_half_angle_square_target_root(ctx, expr)?;
    let rewritten = build_plain_trig_pow2_root(ctx, trig_fn, arg);
    let rule_name = match trig_fn {
        BuiltinFn::Sin => "sin²(x/2) = (1 - cos(x))/2",
        BuiltinFn::Cos => "cos²(x/2) = (1 + cos(x))/2",
        _ => unreachable!("only sin/cos half-angle squares are supported"),
    };

    Some(run_named_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        rewritten,
        "Trig Half-Angle Squares",
        rule_name,
        collect_steps,
    ))
}

pub(super) fn try_standard_direct_pythagorean_extended_zero_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches_direct_pythagorean_extended_zero_identity_root(ctx, expr) {
        return None;
    }

    let zero = ctx.num(0);
    let rewrite = crate::rule::Rewrite::with_local(zero, "Pythagorean Identity", expr, zero);
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Pythagorean Identity",
        collect_steps,
    ))
}

pub(super) fn try_standard_direct_sum_to_product_root_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let rewrite = try_rewrite_sum_to_product_contraction_expr(ctx, expr)?;
    let rewritten = match rewrite.kind {
        cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinSum
        | cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinDiff => {
            if let Some((sin_arg, cos_arg)) =
                extract_scaled_trig_sin_cos_product_args_root(ctx, rewrite.rewritten)
            {
                let canonical_cos_arg = canonicalize_even_cos_arg_root(ctx, cos_arg);
                build_scaled_trig_sin_cos_product_root(ctx, sin_arg, canonical_cos_arg)
            } else {
                rewrite.rewritten
            }
        }
        _ => rewrite.rewritten,
    };
    let mut shortcut_steps = Vec::new();
    if collect_steps {
        shortcut_steps.push(build_root_shortcut_compact_step(
            expr,
            rewritten,
            "Aplicar suma a producto",
            "Sum-to-Product Identity",
        ));
    }
    Some((rewritten, shortcut_steps))
}

fn canonicalize_sum_to_product_contraction_target_root(
    ctx: &mut Context,
    rewrite: cas_math::trig_sum_product_support::TrigSumToProductContractionRewrite,
) -> ExprId {
    match rewrite.kind {
        cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinSum
        | cas_math::trig_sum_product_support::TrigSumToProductContractionRewriteKind::SinDiff => {
            if let Some((sin_arg, cos_arg)) =
                extract_scaled_trig_sin_cos_product_args_root(ctx, rewrite.rewritten)
            {
                let canonical_cos_arg = canonicalize_even_cos_arg_root(ctx, cos_arg);
                build_scaled_trig_sin_cos_product_root(ctx, sin_arg, canonical_cos_arg)
            } else {
                rewrite.rewritten
            }
        }
        _ => rewrite.rewritten,
    }
}

pub(super) fn rewrites_sum_to_product_target_root(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    let Some(rewrite) = try_rewrite_sum_to_product_contraction_expr(ctx, source_expr) else {
        return false;
    };
    let rewritten = canonicalize_sum_to_product_contraction_target_root(ctx, rewrite);
    compare_expr(ctx, rewritten, target_expr) == Ordering::Equal
}

pub(super) fn rewrites_product_to_sum_target_root(
    ctx: &mut Context,
    source_expr: ExprId,
    target_expr: ExprId,
) -> bool {
    let Some(rewrite) = try_rewrite_product_to_sum_expr(ctx, source_expr) else {
        return false;
    };
    let rewritten = rewrite_direct_trig_product_to_sum_double_angle_target_root(ctx, source_expr)
        .unwrap_or(rewrite.rewritten);
    compare_expr(ctx, rewritten, target_expr) == Ordering::Equal
}

pub(super) fn try_standard_half_angle_square_factor_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let _ = options;
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (index, factor) in factors.iter().copied().enumerate() {
        let Some((trig_fn, arg)) = extract_direct_half_angle_square_target_root(ctx, factor) else {
            continue;
        };

        let mut rewritten_factors = factors.clone();
        rewritten_factors[index] = build_plain_trig_pow2_root(ctx, trig_fn, arg);
        let rewritten = build_mul_expr_from_factors_root(ctx, &rewritten_factors);
        let rule_name = match trig_fn {
            BuiltinFn::Sin => "sin²(x/2) = (1 - cos(x))/2",
            BuiltinFn::Cos => "cos²(x/2) = (1 + cos(x))/2",
            _ => unreachable!("only sin/cos half-angle squares are supported"),
        };

        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            rewritten,
            "Trig Half-Angle Squares",
            rule_name,
            collect_steps,
        ));
    }

    None
}

pub(super) fn try_standard_trig_product_to_sum_subset_factor_shortcut(
    options: &crate::phase::SimplifyOptions,
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
                let subset =
                    build_mul_expr_from_factors_root(ctx, &[factors[i], factors[j], factors[k]]);
                let Some(rewrite) = try_rewrite_product_to_sum_expr(ctx, subset) else {
                    continue;
                };
                if !matches!(
                    rewrite.kind,
                    cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::SinCos
                        | cas_math::trig_sum_product_support::TrigProductToSumRewriteKind::CosSin
                ) {
                    continue;
                }
                let rewritten_subset =
                    rewrite_direct_trig_product_to_sum_double_angle_target_root(ctx, subset)
                        .unwrap_or(rewrite.rewritten);

                let mut remaining_indices = Vec::with_capacity(factors.len() - 3);
                for index in 0..factors.len() {
                    if index != i && index != j && index != k {
                        remaining_indices.push(index);
                    }
                }

                let remaining_factors: Vec<_> = remaining_indices
                    .iter()
                    .map(|&index| factors[index])
                    .collect();
                let combined_partner = build_mul_expr_from_factors_root(ctx, &remaining_factors);

                if let Some(partner_canonical) =
                    canonicalize_direct_pair_factor_root(ctx, combined_partner)
                {
                    if compare_expr(ctx, partner_canonical, combined_partner) != Ordering::Equal {
                        let rewritten =
                            build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
                                ctx,
                                &[rewritten_subset, partner_canonical],
                            );
                        let rewrite = crate::rule::Rewrite::with_local(
                            rewritten,
                            "Product-to-Sum Combined Partner",
                            expr,
                            rewritten,
                        );
                        return Some(finish_standard_root_shortcut(
                            ctx,
                            expr,
                            rewrite,
                            "Product-to-Sum Combined Partner",
                            collect_steps,
                        ));
                    }
                }

                if remaining_indices.len() == 1 {
                    if let Some(partner_simplified) =
                        isolated_simplify_expr_if_changed(options, ctx, combined_partner)
                    {
                        let rewritten =
                            build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
                                ctx,
                                &[rewritten_subset, partner_simplified],
                            );
                        let rewrite = crate::rule::Rewrite::with_local(
                            rewritten,
                            "Product-to-Sum Simplified Partner",
                            expr,
                            rewritten,
                        );
                        return Some(finish_standard_root_shortcut(
                            ctx,
                            expr,
                            rewrite,
                            "Product-to-Sum Simplified Partner",
                            collect_steps,
                        ));
                    }
                }

                for partner_index in remaining_indices.iter().copied() {
                    let Some(partner_canonical) =
                        canonicalize_direct_pair_factor_root(ctx, factors[partner_index])
                    else {
                        continue;
                    };
                    if compare_expr(ctx, partner_canonical, factors[partner_index])
                        == Ordering::Equal
                    {
                        continue;
                    }

                    let mut rewritten_factors = Vec::with_capacity(factors.len() - 2);
                    for (index, factor) in factors.iter().copied().enumerate() {
                        if index == i {
                            rewritten_factors.push(rewritten_subset);
                        } else if index == j || index == k {
                            continue;
                        } else if index == partner_index {
                            rewritten_factors.push(partner_canonical);
                        } else {
                            rewritten_factors.push(factor);
                        }
                    }

                    let rewritten =
                        build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
                            ctx,
                            &rewritten_factors,
                        );
                    let rewrite = crate::rule::Rewrite::with_local(
                        rewritten,
                        "Product-to-Sum Direct Pair Factor",
                        expr,
                        rewritten,
                    );
                    return Some(finish_standard_root_shortcut(
                        ctx,
                        expr,
                        rewrite,
                        "Product-to-Sum Direct Pair Factor",
                        collect_steps,
                    ));
                }

                let mut rewritten_factors = Vec::with_capacity(factors.len() - 2);
                for (index, factor) in factors.iter().copied().enumerate() {
                    if index == i {
                        rewritten_factors.push(rewritten_subset);
                    } else if index == j || index == k {
                        continue;
                    } else {
                        rewritten_factors.push(factor);
                    }
                }

                let rewritten = build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
                    ctx,
                    &rewritten_factors,
                );
                let rewrite = crate::rule::Rewrite::with_local(
                    rewritten,
                    "Product-to-Sum Factor",
                    expr,
                    rewritten,
                );
                return Some(finish_standard_root_shortcut(
                    ctx,
                    expr,
                    rewrite,
                    "Product-to-Sum Factor",
                    collect_steps,
                ));
            }
        }
    }

    None
}

pub(super) fn try_standard_special_angle_exact_value_factor_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();

    fn canonicalize_special_angle_partner_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
        factor_sum_diff_cubes_partner_root(ctx, expr)
            .or_else(|| factor_higher_degree_difference_partner_root(ctx, expr))
            .or_else(|| factor_sophie_germain_partner_root(ctx, expr))
            .or_else(|| factor_known_small_polynomial_partner_root(ctx, expr))
            .or_else(|| canonicalize_direct_pair_factor_root(ctx, expr))
    }

    fn split_fractional_constant_factor_root(
        ctx: &mut Context,
        expr: ExprId,
    ) -> Option<Vec<ExprId>> {
        let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
            return None;
        };
        if !is_pure_arithmetic_constant_expr_root(ctx, denominator) {
            return None;
        }
        let one = ctx.num(1);
        let reciprocal = ctx.add(Expr::Div(one, denominator));
        Some(vec![reciprocal, numerator])
    }

    fn build_remaining_partner_root(
        ctx: &mut Context,
        factors: &[ExprId],
        excluded_index: usize,
    ) -> Option<ExprId> {
        let remaining_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, factor)| (index != excluded_index).then_some(factor))
            .collect();
        (!remaining_factors.is_empty())
            .then(|| build_mul_expr_from_factors_root(ctx, &remaining_factors))
    }

    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }
    let has_potential_special_angle_source = if profiling {
        run_profiled_root_shortcut("root.mul.19p.special_angle.potential_source_scan", || {
            Some(
                factors
                    .iter()
                    .copied()
                    .any(|factor| is_potential_special_angle_exact_factor_source_root(ctx, factor)),
            )
        })
        .unwrap_or(false)
    } else {
        factors
            .iter()
            .copied()
            .any(|factor| is_potential_special_angle_exact_factor_source_root(ctx, factor))
    };
    if !has_potential_special_angle_source {
        return None;
    }
    let special_values: Vec<Option<ExprId>> = if profiling {
        run_profiled_root_shortcut("root.mul.19q.special_angle.exact_value_probe", || {
            Some(
                factors
                    .iter()
                    .copied()
                    .map(|factor| {
                        if is_potential_special_angle_exact_factor_source_root(ctx, factor)
                            && !is_definitely_non_special_angle_exact_value_probe_root(ctx, factor)
                        {
                            crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                                "root.mul.19q.special_angle.exact_value_probe",
                                render_expr_for_orchestrator_profile(ctx, factor),
                            );
                            extract_special_angle_exact_value_root(ctx, factor)
                        } else {
                            None
                        }
                    })
                    .collect(),
            )
        })
        .unwrap_or_default()
    } else {
        factors
            .iter()
            .copied()
            .map(|factor| {
                if is_potential_special_angle_exact_factor_source_root(ctx, factor)
                    && !is_definitely_non_special_angle_exact_value_probe_root(ctx, factor)
                {
                    extract_special_angle_exact_value_root(ctx, factor)
                } else {
                    None
                }
            })
            .collect()
    };
    if special_values.iter().all(Option::is_none) {
        return None;
    }

    let mut direct_double_angle = || {
        for (special_index, special_value) in special_values.iter().copied().enumerate() {
            let Some(special_value) = special_value else {
                continue;
            };
            let Some(combined_partner) = build_remaining_partner_root(ctx, &factors, special_index)
            else {
                continue;
            };
            let Some(double_angle_arg) =
                extract_positive_cos_double_angle_arg_root(ctx, combined_partner)
            else {
                continue;
            };
            if cas_ast::collect_variables(ctx, double_angle_arg).is_empty() {
                continue;
            }

            let rewritten_factors = [special_value, combined_partner];
            let rewritten = build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
                ctx,
                &rewritten_factors,
            );
            let shortcut_steps = if collect_steps {
                vec![build_root_shortcut_compact_step(
                    expr,
                    rewritten,
                    "Canonizar producto con factor trigonométrico de ángulo especial y coseno de doble ángulo",
                    "Special Angle Direct Pair Product",
                )]
            } else {
                Vec::new()
            };
            return Some((rewritten, shortcut_steps));
        }
        None
    };
    if let Some(result) = if profiling {
        run_profiled_root_shortcut(
            "root.mul.19a.special_angle.direct_double_angle_partner",
            direct_double_angle,
        )
    } else {
        direct_double_angle()
    } {
        return Some(result);
    }

    let mut scaled_half_angle = || {
        for (special_index, special_value) in special_values.iter().copied().enumerate() {
            let Some(special_value) = special_value else {
                continue;
            };
            let Some(combined_partner) = build_remaining_partner_root(ctx, &factors, special_index)
            else {
                continue;
            };
            let Some((trig_fn, full_arg)) =
                extract_direct_scaled_half_angle_pow2_source_root(ctx, combined_partner)
            else {
                continue;
            };

            let partner_target = build_scaled_half_angle_target_root(ctx, trig_fn, full_arg);
            let rewritten_factors = [special_value, partner_target];
            let rewritten = build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
                ctx,
                &rewritten_factors,
            );
            let shortcut_steps = if collect_steps {
                vec![build_root_shortcut_compact_step(
                    expr,
                    rewritten,
                    "Canonizar producto con factor exacto de ángulo especial y partner de medio ángulo escalado",
                    "Special Angle Direct Pair Product",
                )]
            } else {
                Vec::new()
            };
            return Some((rewritten, shortcut_steps));
        }
        None
    };
    if let Some(result) = if profiling {
        run_profiled_root_shortcut(
            "root.mul.19b.special_angle.scaled_half_angle_partner",
            scaled_half_angle,
        )
    } else {
        scaled_half_angle()
    } {
        return Some(result);
    }

    let mut fractional_exact = || {
        for (special_index, special_value) in special_values.iter().copied().enumerate() {
            let Some(special_value) = special_value else {
                continue;
            };
            let Some(combined_partner) = build_remaining_partner_root(ctx, &factors, special_index)
            else {
                continue;
            };
            let partner_canonical = canonicalize_special_angle_partner_root(ctx, combined_partner)
                .unwrap_or(combined_partner);
            let partner_canonical = if cas_ast::count_nodes(ctx, partner_canonical) <= 20
                && extract_direct_scaled_half_angle_pow2_source_root(ctx, partner_canonical)
                    .is_none()
            {
                isolated_simplify_expr_if_changed(options, ctx, partner_canonical)
                    .unwrap_or(partner_canonical)
            } else {
                partner_canonical
            };
            let mut rewritten_factors = if matches!(
                ctx.get(partner_canonical),
                Expr::Add(_, _) | Expr::Sub(_, _)
            ) {
                vec![special_value]
            } else {
                let Some(split_factors) = split_fractional_constant_factor_root(ctx, special_value)
                else {
                    continue;
                };
                split_factors
            };
            rewritten_factors.push(partner_canonical);
            let rewritten_raw = build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
                ctx,
                &rewritten_factors,
            );
            let rewritten_raw = strip_multiplicative_one_root(ctx, rewritten_raw);
            let defer_nested_simplify = rewritten_factors.iter().copied().any(|factor| {
                extract_positive_cos_double_angle_arg_root(ctx, factor)
                    .is_some_and(|arg| !cas_ast::collect_variables(ctx, arg).is_empty())
                    || extract_direct_positive_double_cos_square_diff_target_root(ctx, factor)
                        .is_some_and(|arg| !cas_ast::collect_variables(ctx, arg).is_empty())
            });
            let rewritten = if defer_nested_simplify {
                rewritten_raw
            } else {
                isolated_simplify_expr_if_changed(options, ctx, rewritten_raw)
                    .unwrap_or(rewritten_raw)
            };
            let shortcut_steps = if collect_steps {
                vec![build_root_shortcut_compact_step(
                    expr,
                    rewritten,
                    "Canonizar producto con factor exacto fraccional de ángulo especial y partner equivalente",
                    "Special Angle Fractional Exact Product",
                )]
            } else {
                Vec::new()
            };
            return Some((rewritten, shortcut_steps));
        }
        None
    };
    if let Some(result) = if profiling {
        run_profiled_root_shortcut(
            "root.mul.19c.special_angle.fractional_exact_partner",
            fractional_exact,
        )
    } else {
        fractional_exact()
    } {
        return Some(result);
    }

    let mut small_polynomial_partner = || {
        for (special_index, special_value) in special_values.iter().copied().enumerate() {
            let Some(special_value) = special_value else {
                continue;
            };
            let Some(combined_partner) = build_remaining_partner_root(ctx, &factors, special_index)
            else {
                continue;
            };
            let Some(partner_factored) =
                factor_known_small_polynomial_partner_root(ctx, combined_partner)
            else {
                continue;
            };

            let rewritten =
                build_mul_expr_from_factors_root(ctx, &[special_value, partner_factored]);
            let shortcut_steps = if collect_steps {
                vec![build_root_shortcut_compact_step(
                    expr,
                    rewritten,
                    "Canonizar producto con factor trigonométrico de ángulo especial y partner polinómico pequeño",
                    "Special Angle Direct Pair Product",
                )]
            } else {
                Vec::new()
            };
            return Some((rewritten, shortcut_steps));
        }
        None
    };
    if let Some(result) = if profiling {
        run_profiled_root_shortcut(
            "root.mul.19d.special_angle.small_polynomial_partner",
            small_polynomial_partner,
        )
    } else {
        small_polynomial_partner()
    } {
        return Some(result);
    }

    let mut factorwise_fallback = || {
        let mut any_changed = false;
        let mut rewritten_factors = factors.clone();

        for (index, factor) in factors.iter().copied().enumerate() {
            let replacement = if let Some(special_value) = special_values[index] {
                Some(special_value)
            } else {
                canonicalize_direct_pair_factor_root(ctx, factor)
            };

            let Some(replacement) = replacement else {
                continue;
            };
            if compare_expr(ctx, replacement, factor) == Ordering::Equal {
                continue;
            }
            rewritten_factors[index] = replacement;
            any_changed = true;
        }

        if any_changed {
            let rewritten_raw = build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
                ctx,
                &rewritten_factors,
            );
            let defer_nested_simplify = rewritten_factors.iter().copied().any(|factor| {
                extract_positive_cos_double_angle_arg_root(ctx, factor)
                    .is_some_and(|arg| !cas_ast::collect_variables(ctx, arg).is_empty())
                    || extract_direct_positive_double_cos_square_diff_target_root(ctx, factor)
                        .is_some_and(|arg| !cas_ast::collect_variables(ctx, arg).is_empty())
                    || matches!(ctx.get(factor), Expr::Add(_, _) | Expr::Sub(_, _))
            });
            let rewritten = if defer_nested_simplify {
                rewritten_raw
            } else {
                isolated_simplify_expr_if_changed(options, ctx, rewritten_raw)
                    .unwrap_or(rewritten_raw)
            };
            let shortcut_steps = if collect_steps {
                vec![build_root_shortcut_compact_step(
                    expr,
                    rewritten,
                    "Canonizar producto con factor trigonométrico de ángulo especial y partner directo",
                    "Special Angle Direct Pair Product",
                )]
            } else {
                Vec::new()
            };
            return Some((rewritten, shortcut_steps));
        }

        None
    };
    if profiling {
        run_profiled_root_shortcut(
            "root.mul.19e.special_angle.factorwise_fallback",
            factorwise_fallback,
        )
    } else {
        factorwise_fallback()
    }
}

pub(super) fn build_scaled_half_angle_pow2_target_root(
    ctx: &mut Context,
    trig_fn: BuiltinFn,
    full_arg: ExprId,
) -> ExprId {
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let half_arg = smart_mul(ctx, half, full_arg);
    let trig_expr = ctx.call_builtin(trig_fn, vec![half_arg]);
    let two = ctx.num(2);
    let trig_sq = ctx.add(Expr::Pow(trig_expr, two));
    smart_mul(ctx, two, trig_sq)
}

fn build_scaled_half_angle_target_root(
    ctx: &mut Context,
    trig_fn: BuiltinFn,
    full_arg: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let cos_expr = ctx.call_builtin(BuiltinFn::Cos, vec![full_arg]);
    match trig_fn {
        BuiltinFn::Sin => ctx.add(Expr::Sub(one, cos_expr)),
        BuiltinFn::Cos => ctx.add(Expr::Add(one, cos_expr)),
        _ => ctx.num(0),
    }
}

fn extract_direct_scaled_half_angle_pow2_source_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    let (coeff, trig_name, trig_arg, effective_sign) =
        extract_signed_numeric_trig_pow2(ctx, expr, Sign::Pos)?;
    if effective_sign != Sign::Pos || coeff != BigRational::from_integer(2.into()) {
        return None;
    }

    let trig_fn = match trig_name {
        "sin" => BuiltinFn::Sin,
        "cos" => BuiltinFn::Cos,
        _ => return None,
    };
    let full_arg = extract_half_scaled_base_root(ctx, trig_arg)?;
    Some((trig_fn, full_arg))
}

fn matches_reciprocal_trig_half_angle_fraction_passthrough_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
        return false;
    };

    let Some((denominator_square_fn, full_arg)) =
        extract_direct_scaled_half_angle_square_target_root(ctx, denominator)
    else {
        return false;
    };
    let numerator_fn = match denominator_square_fn {
        BuiltinFn::Cos => BuiltinFn::Sin,
        BuiltinFn::Sin => BuiltinFn::Cos,
        _ => return false,
    };

    let Some(numerator_arg) =
        extract_scaled_plain_sin_or_cos_arg_root(ctx, numerator, numerator_fn)
    else {
        return false;
    };
    let Some(numerator_full_arg) = extract_half_scaled_base_root(ctx, numerator_arg) else {
        return false;
    };

    compare_expr(ctx, numerator_full_arg, full_arg) == Ordering::Equal
}

pub(super) fn try_standard_reciprocal_trig_half_angle_fraction_passthrough_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    _collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    matches_reciprocal_trig_half_angle_fraction_passthrough_root(ctx, expr)
        .then_some((expr, Vec::new()))
}

pub(super) fn try_standard_direct_scaled_half_angle_square_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (trig_fn, full_arg) = extract_direct_scaled_half_angle_pow2_source_root(ctx, expr)?;
    if !expr_contains_division_node_local(ctx, full_arg) {
        return None;
    }

    let canonical_arg = isolated_simplify_expr_if_changed(&isolated_probe_options(), ctx, full_arg)
        .unwrap_or(full_arg);
    let rewritten = build_scaled_half_angle_target_root(ctx, trig_fn, canonical_arg);
    let rule_name = match trig_fn {
        BuiltinFn::Sin => "2·sin²(x/2) = 1 - cos(x)",
        BuiltinFn::Cos => "2·cos²(x/2) = 1 + cos(x)",
        _ => unreachable!("only sin/cos scaled half-angle squares are supported"),
    };

    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::with_local(rewritten, rule_name, expr, rewritten),
        rule_name,
        collect_steps,
    ))
}

pub(super) fn try_standard_rational_half_angle_target_passthrough_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (trig_fn, full_arg) = extract_direct_scaled_half_angle_square_target_root(ctx, expr)?;
    if !expr_contains_division_node_local(ctx, full_arg) {
        return None;
    }

    let canonical_arg = isolated_simplify_expr_if_changed(&isolated_probe_options(), ctx, full_arg)
        .unwrap_or(full_arg);
    let rewritten = build_scaled_half_angle_target_root(ctx, trig_fn, canonical_arg);
    if compare_expr(ctx, rewritten, expr) == Ordering::Equal {
        return Some((expr, Vec::new()));
    }

    let rule_name = match trig_fn {
        BuiltinFn::Sin => "2·sin²(x/2) = 1 - cos(x)",
        BuiltinFn::Cos => "2·cos²(x/2) = 1 + cos(x)",
        _ => unreachable!("only sin/cos rational half-angle targets are supported"),
    };

    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::with_local(rewritten, rule_name, expr, rewritten),
        rule_name,
        collect_steps,
    ))
}

pub(super) fn try_standard_scaled_half_angle_anchor_direct_partner_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    fn canonicalize_scaled_half_angle_partner_root(
        ctx: &mut Context,
        expr: ExprId,
    ) -> Option<ExprId> {
        factor_sum_diff_cubes_partner_root(ctx, expr)
            .or_else(|| factor_higher_degree_difference_partner_root(ctx, expr))
            .or_else(|| factor_sophie_germain_partner_root(ctx, expr))
            .or_else(|| factor_known_small_polynomial_partner_root(ctx, expr))
            .or_else(|| canonicalize_direct_pair_factor_root(ctx, expr))
    }

    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 3 {
        return None;
    }

    for left_index in 0..factors.len() {
        for right_index in (left_index + 1)..factors.len() {
            let anchor_source =
                build_mul_expr_from_factors_root(ctx, &[factors[left_index], factors[right_index]]);
            let Some((trig_fn, full_arg)) =
                extract_direct_scaled_half_angle_pow2_source_root(ctx, anchor_source)
            else {
                continue;
            };

            let partner_factors: Vec<_> = factors
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, factor)| {
                    (index != left_index && index != right_index).then_some(factor)
                })
                .collect();
            if partner_factors.is_empty() {
                continue;
            }

            let partner_expr = build_mul_expr_from_factors_root(ctx, &partner_factors);
            if cas_ast::collect_variables(ctx, partner_expr).is_empty() {
                continue;
            }
            if is_safe_direct_pair_anchor_target_root(ctx, partner_expr) {
                continue;
            }
            let partner_canonical = canonicalize_scaled_half_angle_partner_root(ctx, partner_expr)
                .unwrap_or(partner_expr);
            if is_safe_direct_pair_anchor_target_root(ctx, partner_canonical) {
                continue;
            }
            let canonical_arg =
                isolated_simplify_expr_if_changed(options, ctx, full_arg).unwrap_or(full_arg);
            let anchor_target = build_scaled_half_angle_target_root(ctx, trig_fn, canonical_arg);
            let rewritten =
                build_mul_expr_from_factors_root(ctx, &[anchor_target, partner_canonical]);
            if compare_expr(ctx, rewritten, expr) == Ordering::Equal
                && compare_expr(ctx, partner_canonical, partner_expr) == Ordering::Equal
            {
                continue;
            }

            return Some(run_named_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                rewritten,
                "Scaled Half-Angle Anchor Direct Partner",
                "Scaled Half-Angle Anchor Direct Partner",
                collect_steps,
            ));
        }
    }

    None
}

fn matches_compact_tan_cot_half_angle_zero_pair_root(
    ctx: &mut Context,
    lhs: ExprId,
    rhs: ExprId,
) -> bool {
    (matches_direct_tan_cot_product_zero_identity_root(ctx, lhs)
        && matches_direct_half_angle_square_zero_identity_root(ctx, rhs))
        || (matches_direct_tan_cot_product_zero_identity_root(ctx, rhs)
            && matches_direct_half_angle_square_zero_identity_root(ctx, lhs))
}

pub(super) fn try_standard_compact_tan_cot_half_angle_zero_pair_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (lhs, rhs) = match ctx.get(expr).clone() {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => (lhs, rhs),
        _ => return None,
    };

    if !matches_compact_tan_cot_half_angle_zero_pair_root(ctx, lhs, rhs) {
        return None;
    }

    let zero = ctx.num(0);
    let steps = if collect_steps {
        vec![build_root_shortcut_compact_step(
            expr,
            zero,
            "Collapse Exact Zero Additive Subexpression",
            "Collapse Exact Zero Additive Subexpression",
        )]
    } else {
        Vec::new()
    };
    Some((zero, steps))
}

pub(super) fn try_standard_half_angle_subset_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let view = AddView::from_expr(ctx, expr);
    if !(5..=6).contains(&view.terms.len()) {
        return None;
    }

    let matches_trig_identity_subset = |ctx: &mut Context, subset_expr: ExprId| {
        expr_contains_trig_builtin_local(ctx, subset_expr)
            && !expr_contains_hyperbolic_builtin_local(ctx, subset_expr)
            && (matches_direct_small_zero_identity_root(ctx, subset_expr)
                || matches_direct_trig_mixed_double_angle_zero_identity_root(ctx, subset_expr))
    };

    for subset_size in [2usize, 3usize] {
        for first_index in 0..view.terms.len() {
            for second_index in (first_index + 1)..view.terms.len() {
                if subset_size == 2 {
                    let subset_terms = [view.terms[first_index], view.terms[second_index]];
                    let subset_expr = build_signed_sum_expr_root(ctx, &subset_terms);
                    if !matches_trig_identity_subset(ctx, subset_expr) {
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
                    if remaining_terms.len() != 3 {
                        continue;
                    }

                    let remaining_expr = AddView {
                        root: expr,
                        terms: remaining_terms,
                    }
                    .rebuild(ctx);
                    if try_standard_exact_zero_equivalence_shortcut(
                        options,
                        ctx,
                        remaining_expr,
                        false,
                    )
                    .is_some()
                        || isolated_simplify_rewrites_to_zero(options, ctx, remaining_expr)
                    {
                        let zero = ctx.num(0);
                        return Some(run_rebuilt_root_shortcut_simplify(
                            options,
                            ctx,
                            expr,
                            zero,
                            collect_steps,
                        ));
                    }
                    continue;
                }

                for third_index in (second_index + 1)..view.terms.len() {
                    let subset_terms = [
                        view.terms[first_index],
                        view.terms[second_index],
                        view.terms[third_index],
                    ];
                    let subset_expr = build_signed_sum_expr_root(ctx, &subset_terms);
                    if !matches_trig_identity_subset(ctx, subset_expr) {
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
                    if remaining_terms.len() != 3 {
                        continue;
                    }

                    let remaining_expr = AddView {
                        root: expr,
                        terms: remaining_terms,
                    }
                    .rebuild(ctx);
                    if try_standard_exact_zero_equivalence_shortcut(
                        options,
                        ctx,
                        remaining_expr,
                        false,
                    )
                    .is_some()
                        || isolated_simplify_rewrites_to_zero(options, ctx, remaining_expr)
                    {
                        let zero = ctx.num(0);
                        return Some(run_rebuilt_root_shortcut_simplify(
                            options,
                            ctx,
                            expr,
                            zero,
                            collect_steps,
                        ));
                    }
                }
            }
        }
    }

    None
}

pub(super) fn embedded_trig_product_to_sum_candidate_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Mul(_, _) => {
            let factor_count = flatten_mul_chain(ctx, expr).len();
            if factor_count > 3 {
                try_rewrite_product_to_sum_expr(ctx, expr).map(|rewrite| rewrite.rewritten)
            } else if let Expr::Mul(lhs, rhs) = ctx.get(expr) {
                let lhs = *lhs;
                let rhs = *rhs;
                if expr_contains_trig_builtin_local(ctx, lhs)
                    && is_supported_nested_direct_equivalence_partner(ctx, rhs)
                {
                    try_rewrite_product_to_sum_expr(ctx, lhs)
                        .map(|rewrite| smart_mul(ctx, rewrite.rewritten, rhs))
                } else if expr_contains_trig_builtin_local(ctx, rhs)
                    && is_supported_nested_direct_equivalence_partner(ctx, lhs)
                {
                    try_rewrite_product_to_sum_expr(ctx, rhs)
                        .map(|rewrite| smart_mul(ctx, lhs, rewrite.rewritten))
                } else {
                    None
                }
            } else {
                None
            }
        }
        Expr::Div(num, den) => {
            let num = *num;
            let den = *den;
            if expr_contains_trig_builtin_local(ctx, num)
                && is_supported_nested_direct_equivalence_partner(ctx, den)
            {
                try_rewrite_product_to_sum_expr(ctx, num)
                    .map(|rewrite| ctx.add(Expr::Div(rewrite.rewritten, den)))
            } else if expr_contains_trig_builtin_local(ctx, den)
                && is_supported_nested_direct_equivalence_partner(ctx, num)
            {
                try_rewrite_product_to_sum_expr(ctx, den)
                    .map(|rewrite| ctx.add(Expr::Div(num, rewrite.rewritten)))
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(super) fn try_standard_embedded_trig_product_to_sum_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let rewritten = embedded_trig_product_to_sum_candidate_root(ctx, expr)?;

    Some(run_named_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        rewritten,
        "Aplicar producto a suma en el factor trigonométrico",
        "Product-to-Sum Identity",
        collect_steps,
    ))
}

pub(super) fn try_standard_pythagorean_additive_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let mut current = AddView::from_expr(ctx, expr).rebuild(ctx);
    let mut shortcut_steps = Vec::new();
    let mut changed = false;

    loop {
        let normalized = AddView::from_expr(ctx, current).rebuild(ctx);
        if let Some(rewritten) =
            try_rewrite_structural_numeric_pythagorean_add_pair(ctx, normalized)
        {
            let before = current;
            current = rewritten;
            changed = true;
            if collect_steps {
                let mut step = Step::new_compact(
                    "Pythagorean Identity",
                    "Pythagorean Identity",
                    before,
                    current,
                );
                step.importance = crate::step::ImportanceLevel::High;
                shortcut_steps.push(step);
            }
            continue;
        }
        if let Some(rewrite) = try_rewrite_pythagorean_identity_add_expr(ctx, normalized) {
            let before = current;
            current = rewrite.rewritten;
            changed = true;
            if collect_steps {
                let mut step = Step::new_compact(
                    "Pythagorean Identity",
                    "Pythagorean Identity",
                    before,
                    current,
                );
                step.importance = crate::step::ImportanceLevel::High;
                shortcut_steps.push(step);
            }
            continue;
        }

        let Some(rewrite) = try_rewrite_combine_constants_expr(ctx, current) else {
            break;
        };
        if compare_expr(ctx, current, rewrite.rewritten) == Ordering::Equal {
            break;
        }

        let before = current;
        current = rewrite.rewritten;
        changed = true;
        if collect_steps {
            let mut step =
                Step::new_compact(&rewrite.description, "Combine Constants", before, current);
            step.importance = crate::step::ImportanceLevel::High;
            shortcut_steps.push(step);
        }
    }

    if !changed {
        return None;
    }

    if collect_steps {
        if let Some(first) = shortcut_steps.first_mut() {
            first.global_before = Some(expr);
        }
        if let Some(last) = shortcut_steps.last_mut() {
            last.global_after = Some(current);
        }
    }

    Some((current, shortcut_steps))
}

pub(super) fn try_standard_pythagorean_additive_pipeline_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        let add_term_count = AddView::from_expr(ctx, expr).terms.len();
        if add_term_count >= 5 && expr_contains_log_builtin_local(ctx, expr) {
            return None;
        }
    }

    let (rewritten, mut shortcut_steps) =
        try_standard_pythagorean_additive_shortcut(ctx, expr, collect_steps)?;

    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
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

pub(super) fn try_standard_pythagorean_generic_coefficient_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let rewrite = try_rewrite_pythagorean_generic_coefficient_add_expr(ctx, expr)?;
    let mut simplifier = crate::Simplifier::with_default_rules();
    std::mem::swap(&mut simplifier.context, ctx);
    let (result, inner_steps, _stats) = simplifier.simplify_with_stats(
        rewrite.rewritten,
        crate::SimplifyOptions {
            suppress_depth_overflow_warnings: true,
            ..options.clone()
        },
    );
    std::mem::swap(&mut simplifier.context, ctx);

    let mut shortcut_steps = Vec::new();
    if collect_steps {
        let mut step = Step::new_compact(
            &rewrite.desc,
            "Pythagorean with Generic Coefficient",
            expr,
            rewrite.rewritten,
        );
        step.global_before = Some(expr);
        step.global_after = Some(rewrite.rewritten);
        step.importance = crate::step::ImportanceLevel::High;
        shortcut_steps.push(step);
        shortcut_steps.extend(inner_steps);
    }
    Some((result, shortcut_steps))
}

pub(super) fn extract_partitioned_phase_shift_zero_chunks_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    crate::rules::arithmetic::extract_repeated_trig_phase_shift_pair_zero_chunks(ctx, expr)
}

pub(super) fn matches_direct_trig_product_to_sum_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    matches_direct_trig_product_to_sum_sin_sin_zero_identity_root(ctx, expr)
        || matches_direct_trig_product_to_sum_sin_cos_zero_identity_root(ctx, expr)
        || matches_direct_trig_product_to_sum_cos_cos_zero_identity_root(ctx, expr)
}

pub(super) fn matches_direct_trig_product_to_sum_and_odd_half_partition_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if !(5..=6).contains(&terms.len()) {
        return false;
    }

    for first_index in 0..terms.len().saturating_sub(1) {
        for second_index in (first_index + 1)..terms.len() {
            let odd_terms: Vec<_> = terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index == first_index || index == second_index).then_some(term)
                })
                .collect();
            if odd_terms.len() != 2 {
                continue;
            }

            let trig_terms: Vec<_> = terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| {
                    (index != first_index && index != second_index).then_some(term)
                })
                .collect();
            if trig_terms.len() != terms.len() - 2 {
                continue;
            }

            let odd_expr = build_signed_sum_expr_root(ctx, &odd_terms);
            if !matches_direct_odd_half_power_zero_scope_root(ctx, odd_expr) {
                continue;
            }

            let trig_expr = build_signed_sum_expr_root(ctx, &trig_terms);
            if matches_direct_trig_product_to_sum_zero_identity_root(ctx, trig_expr) {
                return true;
            }
        }
    }

    false
}

pub(super) fn matches_direct_trig_product_to_sum_and_geometric_difference_partition_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    if terms.len() != 6 {
        return false;
    }

    let mut stack = vec![(0usize, Vec::<usize>::new())];
    while let Some((next_index, chosen)) = stack.pop() {
        if chosen.len() == 3 {
            let trig_terms: Vec<_> = terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| chosen.contains(&index).then_some(term))
                .collect();
            let residual_terms: Vec<_> = terms
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, term)| (!chosen.contains(&index)).then_some(term))
                .collect();
            if residual_terms.len() != 3 {
                continue;
            }

            let trig_expr = build_signed_sum_expr_root(ctx, &trig_terms);
            if matches_direct_trig_product_to_sum_zero_identity_root(ctx, trig_expr)
                && matches_geometric_difference_terms_root(ctx, &residual_terms)
            {
                return true;
            }
            continue;
        }

        for index in next_index..terms.len() {
            let mut next = chosen.clone();
            next.push(index);
            stack.push((index + 1, next));
        }
    }

    false
}

pub(super) fn try_standard_repeated_phase_shift_pair_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if let Some((first_chunk, second_chunk)) =
        crate::rules::arithmetic::extract_repeated_trig_phase_shift_pair_zero_chunks(ctx, expr)
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
        if crate::rules::arithmetic::extract_repeated_trig_phase_shift_pair_zero_chunks(
            ctx,
            residual_expr,
        )
        .is_some()
        {
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

    None
}

pub(super) fn try_standard_trig_double_angle_cos_variant_zero_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        let add_term_count = AddView::from_expr(ctx, expr).terms.len();
        if add_term_count >= 5 && expr_contains_log_builtin_local(ctx, expr) {
            return None;
        }
    }

    let rewrite =
        crate::rules::arithmetic::try_build_exact_zero_trig_double_angle_cos_variant_zero_scope_rewrite(
            ctx, expr,
        )?;
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Double Angle Expansion",
        collect_steps,
    ))
}

pub(super) fn try_standard_shared_passthrough_pythagorean_factor_form_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (lhs_core, rhs_core) = extract_shared_additive_passthrough_sub_cores_root(ctx, expr)?;
    if matches_direct_trig_product_to_sum_sin_sin_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_sin_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_trig_product_to_sum_cos_cos_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_nested_fraction_simplified_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_sinh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_cosh_sum_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_hyperbolic_cosh_difference_to_product_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_recursive_hyperbolic_sinh_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_recursive_hyperbolic_cosh_sum_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_cos_square_diff_pair_root(ctx, lhs_core, rhs_core)
        || matches_direct_angle_sum_diff_pair_root(ctx, lhs_core, rhs_core)
    {
        let zero = ctx.num(0);
        let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));
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

    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_pythagorean_factor_form_add_expr(ctx, source) else {
            continue;
        };
        if compare_expr(ctx, rewrite.rewritten, target) == Ordering::Equal {
            let zero = ctx.num(0);
            return Some(finish_standard_root_shortcut(
                ctx,
                expr,
                crate::rule::Rewrite::new(zero).desc("Pythagorean Identity"),
                "Pythagorean Identity",
                collect_steps,
            ));
        }
    }
    let residual_expr = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if expr_contains_trig_or_hyperbolic_builtin_local(ctx, residual_expr)
        && try_standard_exact_zero_equivalence_shortcut(options, ctx, residual_expr, false)
            .is_some()
    {
        let zero = ctx.num(0);
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
    None
}

pub(super) fn try_standard_reciprocal_product_pythagorean_zero_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let plan = try_rewrite_reciprocal_product_pythagorean_zero_add_expr(ctx, expr)?;
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::new(plan.rewritten).desc(plan.desc),
        "Pythagorean Identity",
        collect_steps,
    ))
}

pub(super) fn try_standard_reciprocal_pythagorean_pair_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let rewritten = if let Some(rewrite) = try_rewrite_sec_tan_pythagorean_identity_expr(ctx, expr)
    {
        rewrite.rewritten
    } else if let Some(rewrite) = try_rewrite_csc_cot_pythagorean_identity_expr(ctx, expr) {
        rewrite.rewritten
    } else {
        return None;
    };

    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::new(rewritten).desc("Pythagorean Identity"),
        "Pythagorean Identity",
        collect_steps,
    ))
}

pub(super) fn try_standard_reciprocal_pythagorean_zero_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches_direct_sec_tan_pythagorean_zero_identity_root(ctx, expr)
        && !matches_direct_csc_cot_pythagorean_zero_identity_root(ctx, expr)
    {
        return None;
    }

    let zero = ctx.num(0);
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::new(zero).desc("Pythagorean Identity"),
        "Pythagorean Identity",
        collect_steps,
    ))
}

fn build_standard_trig_square_double_angle_term(ctx: &mut Context, arg: ExprId) -> ExprId {
    let two = ctx.num(2);
    let doubled_arg = smart_mul(ctx, two, arg);
    ctx.call_builtin(BuiltinFn::Sin, vec![doubled_arg])
}

fn try_rewrite_standard_trig_binomial_square_double_angle_pair(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<crate::rule::Rewrite> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for square_idx in 0..view.terms.len() {
        let (square_term, square_sign) = view.terms[square_idx];
        let Some((arg, is_sum)) = extract_standard_trig_binomial_square_data(ctx, square_term)
        else {
            continue;
        };
        let double_angle = build_standard_trig_square_double_angle_term(ctx, arg);
        let required_sign = match (square_sign, is_sum) {
            (Sign::Pos, true) => Sign::Neg,
            (Sign::Pos, false) => Sign::Pos,
            (Sign::Neg, true) => Sign::Pos,
            (Sign::Neg, false) => Sign::Neg,
        };

        for angle_idx in 0..view.terms.len() {
            if angle_idx == square_idx {
                continue;
            }
            let (angle_term, angle_sign) = view.terms[angle_idx];
            if angle_sign != required_sign
                || compare_expr(ctx, angle_term, double_angle) != Ordering::Equal
            {
                continue;
            }

            let mut new_terms = smallvec::SmallVec::<[(ExprId, Sign); 8]>::new();
            for (idx, term) in view.terms.iter().copied().enumerate() {
                if idx != square_idx && idx != angle_idx {
                    new_terms.push(term);
                }
            }
            if new_terms.iter().any(
                |(term, _sign)| matches!(ctx.get(*term), Expr::Number(value) if value.is_one()),
            ) {
                continue;
            }
            new_terms.push((ctx.num(1), square_sign));
            let rewritten = AddView {
                root: expr,
                terms: new_terms,
            }
            .rebuild(ctx);
            let desc = if is_sum {
                "(sin(u)+cos(u))^2 = 1 + sin(2u)"
            } else {
                "(sin(u)-cos(u))^2 = 1 - sin(2u)"
            };
            return Some(crate::rule::Rewrite::new(rewritten).desc(desc));
        }
    }

    None
}

pub(super) fn try_standard_trig_binomial_square_double_angle_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let rewrite = try_rewrite_standard_trig_binomial_square_double_angle_pair(ctx, expr)?;
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

    if compare_expr(ctx, result, rewrite.new_expr) == Ordering::Equal {
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Trig Square Identity",
            collect_steps,
        ));
    }

    let mut shortcut_steps = Vec::new();
    if collect_steps {
        shortcut_steps.push(build_root_shortcut_step_from_rewrite(
            ctx,
            expr,
            &rewrite,
            "Trig Square Identity",
        ));
        shortcut_steps.extend(inner_steps);
    }
    Some((result, shortcut_steps))
}

fn try_rewrite_structural_numeric_pythagorean_add_pair(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    for i in 0..view.terms.len() {
        for j in (i + 1)..view.terms.len() {
            let (lhs_term, lhs_sign) = view.terms[i];
            let (rhs_term, rhs_sign) = view.terms[j];
            let Some((lhs_coeff, lhs_name, lhs_arg, lhs_effective_sign)) =
                extract_signed_numeric_trig_pow2(ctx, lhs_term, lhs_sign)
            else {
                continue;
            };
            let Some((rhs_coeff, rhs_name, rhs_arg, rhs_effective_sign)) =
                extract_signed_numeric_trig_pow2(ctx, rhs_term, rhs_sign)
            else {
                continue;
            };
            if lhs_name == rhs_name
                || lhs_coeff != rhs_coeff
                || lhs_effective_sign != rhs_effective_sign
                || compare_expr(ctx, lhs_arg, rhs_arg) != Ordering::Equal
            {
                continue;
            }

            let mut remaining_terms = smallvec::SmallVec::<[(ExprId, Sign); 8]>::new();
            for (idx, term) in view.terms.iter().copied().enumerate() {
                if idx != i && idx != j {
                    remaining_terms.push(term);
                }
            }
            remaining_terms.push((ctx.add(Expr::Number(lhs_coeff)), lhs_effective_sign));
            return Some(
                AddView {
                    root: expr,
                    terms: remaining_terms,
                }
                .rebuild(ctx),
            );
        }
    }

    None
}

pub(super) fn extract_negative_cos_double_angle_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (coeff, base) = extract_coef_and_base(ctx, expr);
    if coeff != BigRational::from_integer((-1).into()) {
        return None;
    }
    let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, base) else {
        return None;
    };
    extract_double_angle_arg_relaxed(ctx, arg)
}

pub(super) fn has_negative_numeric_pythagorean_pair(ctx: &Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return false;
    }

    for i in 0..view.terms.len() {
        for j in (i + 1)..view.terms.len() {
            let (lhs_term, lhs_sign) = view.terms[i];
            let (rhs_term, rhs_sign) = view.terms[j];
            let Some((lhs_coeff, lhs_name, lhs_arg, lhs_effective_sign)) =
                extract_signed_numeric_trig_pow2(ctx, lhs_term, lhs_sign)
            else {
                continue;
            };
            let Some((rhs_coeff, rhs_name, rhs_arg, rhs_effective_sign)) =
                extract_signed_numeric_trig_pow2(ctx, rhs_term, rhs_sign)
            else {
                continue;
            };
            if lhs_name != rhs_name
                && lhs_coeff == rhs_coeff
                && lhs_effective_sign == Sign::Neg
                && rhs_effective_sign == Sign::Neg
                && compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal
            {
                return true;
            }
        }
    }

    false
}

pub(super) fn has_structural_numeric_pythagorean_pair(ctx: &Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return false;
    }

    for i in 0..view.terms.len() {
        for j in (i + 1)..view.terms.len() {
            let (lhs_term, lhs_sign) = view.terms[i];
            let (rhs_term, rhs_sign) = view.terms[j];
            let Some((lhs_coeff, lhs_name, lhs_arg, lhs_effective_sign)) =
                extract_signed_numeric_trig_pow2(ctx, lhs_term, lhs_sign)
            else {
                continue;
            };
            let Some((rhs_coeff, rhs_name, rhs_arg, rhs_effective_sign)) =
                extract_signed_numeric_trig_pow2(ctx, rhs_term, rhs_sign)
            else {
                continue;
            };
            if lhs_name != rhs_name
                && lhs_coeff == rhs_coeff
                && lhs_effective_sign == rhs_effective_sign
                && compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal
            {
                return true;
            }
        }
    }

    false
}

pub(super) fn has_numeric_pythagorean_complement_pair(ctx: &Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let mut constant_is_positive = None;
    let mut trig_sign = None;
    let mut trig_coeff = None;

    for (term, sign) in view.terms.iter().copied() {
        match ctx.get(term) {
            Expr::Number(n) if n.is_one() => {
                constant_is_positive = Some(sign == Sign::Pos);
            }
            _ => {
                let Some((coeff, _name, _arg, effective_sign)) =
                    extract_signed_numeric_trig_pow2(ctx, term, sign)
                else {
                    return false;
                };
                trig_coeff = Some(coeff);
                trig_sign = Some(effective_sign);
            }
        }
    }

    matches!(trig_coeff, Some(coeff) if coeff.is_one())
        && matches!(
            (constant_is_positive, trig_sign),
            (Some(true), Some(Sign::Neg)) | (Some(false), Some(Sign::Pos))
        )
}
