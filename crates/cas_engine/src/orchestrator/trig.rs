//! Orquestador: familia `trig` (troceo P1).
//!
//! Ver la cabecera de `orchestrator.rs` para el contexto.

use super::*;

fn extract_trig_binomial_square_identity_data_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, bool)> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *exponent)? != 2 {
        return None;
    }

    let (left, right, is_sum) = match ctx.get(*base) {
        Expr::Add(left, right) => (*left, *right, true),
        Expr::Sub(left, right) => (*left, *right, false),
        _ => return None,
    };

    let (lhs_fn, lhs_arg) = extract_plain_sin_or_cos_arg_root(ctx, left)?;
    let (rhs_fn, rhs_arg) = extract_plain_sin_or_cos_arg_root(ctx, right)?;
    if compare_expr(ctx, lhs_arg, rhs_arg) != Ordering::Equal {
        return None;
    }
    let trig_kinds = [lhs_fn, rhs_fn];
    if !trig_kinds.contains(&BuiltinFn::Sin) || !trig_kinds.contains(&BuiltinFn::Cos) {
        return None;
    }

    Some((lhs_arg, is_sum))
}

pub(super) fn build_plain_trig_pow2_root(
    ctx: &mut Context,
    trig_fn: BuiltinFn,
    arg: ExprId,
) -> ExprId {
    let trig_expr = ctx.call_builtin(trig_fn, vec![arg]);
    let two = ctx.num(2);
    ctx.add(Expr::Pow(trig_expr, two))
}

pub(super) fn extract_direct_cos_fourth_power_reduction_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *denominator)? != 8 {
        return None;
    }

    let view = AddView::from_expr(ctx, *numerator);
    if view.terms.len() != 3 {
        return None;
    }

    let mut saw_positive_three = false;
    let mut double_angle_arg = None;
    let mut quadruple_angle_arg = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(3) {
            if term_sign != Sign::Pos || saw_positive_three {
                return None;
            }
            saw_positive_three = true;
            continue;
        }

        let (coeff, base) = extract_coef_and_base(ctx, term_expr);
        let signed_coeff = if term_sign == Sign::Neg {
            -coeff
        } else {
            coeff
        };
        if signed_coeff == BigRational::from_integer(4.into()) {
            if double_angle_arg.is_some() {
                return None;
            }
            double_angle_arg = extract_positive_cos_double_angle_arg_root(ctx, base);
            double_angle_arg?;
            continue;
        }

        if signed_coeff.is_one() {
            if quadruple_angle_arg.is_some() {
                return None;
            }
            quadruple_angle_arg = extract_positive_cos_quadruple_angle_arg_root(ctx, base);
            quadruple_angle_arg?;
            continue;
        }

        return None;
    }

    let double_angle_arg = double_angle_arg?;
    let quadruple_angle_arg = quadruple_angle_arg?;
    if !saw_positive_three
        || compare_expr(ctx, double_angle_arg, quadruple_angle_arg) != Ordering::Equal
    {
        return None;
    }

    Some(double_angle_arg)
}

fn extract_direct_sin_cos_square_product_reduction_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let Expr::Div(numerator, denominator) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *denominator)? != 8 {
        return None;
    }

    let view = AddView::from_expr(ctx, *numerator);
    if view.terms.len() != 2 {
        return None;
    }

    let mut saw_positive_one = false;
    let mut quadruple_angle_arg = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Pos || saw_positive_one {
                return None;
            }
            saw_positive_one = true;
            continue;
        }

        let (coeff, base) = extract_coef_and_base(ctx, term_expr);
        let signed_coeff = if term_sign == Sign::Neg {
            -coeff
        } else {
            coeff
        };
        if signed_coeff != BigRational::from_integer((-1).into()) {
            return None;
        }

        if quadruple_angle_arg.is_some() {
            return None;
        }
        quadruple_angle_arg = extract_positive_cos_quadruple_angle_arg_root(ctx, base);
        quadruple_angle_arg?;
    }

    if !saw_positive_one {
        return None;
    }
    quadruple_angle_arg
}

pub(super) fn build_plain_trig_pow4_root(
    ctx: &mut Context,
    trig_fn: BuiltinFn,
    arg: ExprId,
) -> ExprId {
    let trig_expr = ctx.call_builtin(trig_fn, vec![arg]);
    let four = ctx.num(4);
    ctx.add(Expr::Pow(trig_expr, four))
}

fn extract_scaled_sin_fourth_power_target_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let (coeff, trig_name, arg) = extract_coeff_trig_pow4(ctx, expr)?;
    (trig_name == "sin" && coeff == BigRational::from_integer(8.into())).then_some(arg)
}

fn extract_scaled_sin_fourth_power_reduction_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return None;
    }

    let mut saw_positive_three = false;
    let mut double_angle_arg = None;
    let mut quadruple_angle_arg = None;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(3) {
            if term_sign != Sign::Pos || saw_positive_three {
                return None;
            }
            saw_positive_three = true;
            continue;
        }

        let (coeff, base) = extract_coef_and_base(ctx, term_expr);
        let signed_coeff = if term_sign == Sign::Neg {
            -coeff
        } else {
            coeff
        };

        if signed_coeff == BigRational::from_integer((-4).into()) {
            if double_angle_arg.is_some() {
                return None;
            }
            double_angle_arg = extract_positive_cos_double_angle_arg_root(ctx, base);
            double_angle_arg?;
            continue;
        }

        if signed_coeff.is_one() {
            if quadruple_angle_arg.is_some() {
                return None;
            }
            quadruple_angle_arg = extract_positive_cos_quadruple_angle_arg_root(ctx, base);
            quadruple_angle_arg?;
            continue;
        }

        return None;
    }

    let double_angle_arg = double_angle_arg?;
    let quadruple_angle_arg = quadruple_angle_arg?;
    if !saw_positive_three
        || compare_expr(ctx, double_angle_arg, quadruple_angle_arg) != Ordering::Equal
    {
        return None;
    }

    Some(double_angle_arg)
}

pub(super) fn build_plain_sin_cos_square_product_root(ctx: &mut Context, arg: ExprId) -> ExprId {
    let two = ctx.num(2);
    let sin_expr = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_expr = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    let sin_sq = ctx.add(Expr::Pow(sin_expr, two));
    let cos_sq = ctx.add(Expr::Pow(cos_expr, two));
    build_mul_expr_from_factors_root(ctx, &[sin_sq, cos_sq])
}

pub(super) fn matches_direct_trig_binomial_square_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return false;
    }

    for index in 0..view.terms.len() {
        let (term_expr, term_sign) = view.terms[index];
        if term_sign != Sign::Pos {
            continue;
        }
        let Some((arg, is_sum)) = extract_trig_binomial_square_identity_data_root(ctx, term_expr)
        else {
            continue;
        };
        let mut saw_negative_one = false;
        let mut saw_double_angle = false;
        let mut bad_term = false;
        for (other_index, (other_expr, other_sign)) in view.terms.iter().copied().enumerate() {
            if other_index == index {
                continue;
            }
            let is_negative_one = extract_i64_integer(ctx, other_expr).is_some_and(|value| {
                matches!((value, other_sign), (1, Sign::Neg) | (-1, Sign::Pos))
            });
            let matches_double_angle =
                matches_trig_square_double_angle_term_root(ctx, other_expr, arg)
                    && other_sign == if is_sum { Sign::Neg } else { Sign::Pos };
            if is_negative_one {
                saw_negative_one = true;
            } else if matches_double_angle {
                saw_double_angle = true;
            } else {
                bad_term = true;
                break;
            }
        }
        if !bad_term && saw_negative_one && saw_double_angle {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_trig_binomial_square_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    let lhs_minus_rhs = ctx.add(Expr::Sub(lhs_core, rhs_core));
    if matches_direct_trig_binomial_square_zero_identity_root(ctx, lhs_minus_rhs) {
        return true;
    }

    let rhs_minus_lhs = ctx.add(Expr::Sub(rhs_core, lhs_core));
    matches_direct_trig_binomial_square_zero_identity_root(ctx, rhs_minus_lhs)
}

pub(super) fn extract_scaled_trig_sin_sin_product_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut first_sin_arg = None;
    let mut second_sin_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }

        let Some((BuiltinFn::Sin, arg)) = extract_plain_sin_or_cos_arg_root(ctx, factor) else {
            return None;
        };
        if first_sin_arg.is_none() {
            first_sin_arg = Some(arg);
        } else if second_sin_arg.is_none() {
            second_sin_arg = Some(arg);
        } else {
            return None;
        }
    }

    if numeric_coeff != BigRational::from_integer(2.into()) {
        return None;
    }

    Some((first_sin_arg?, second_sin_arg?))
}

fn extract_direct_trig_power_mixed_square_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let mut sin_arg = None;
    let mut cos_arg = None;
    for factor in factors {
        let (coeff, trig_name, arg, effective_sign) =
            extract_signed_numeric_trig_pow2(ctx, factor, Sign::Pos)?;
        if effective_sign != Sign::Pos || coeff != BigRational::one() {
            return None;
        }
        match trig_name {
            "sin" if sin_arg.is_none() => sin_arg = Some(arg),
            "cos" if cos_arg.is_none() => cos_arg = Some(arg),
            _ => return None,
        }
    }

    let sin_arg = sin_arg?;
    let cos_arg = cos_arg?;
    (compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal).then_some(sin_arg)
}

pub(super) fn matches_direct_trig_power_mixed_square_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (mixed_square, doubled_square) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(base_arg) = extract_direct_trig_power_mixed_square_target_root(ctx, mixed_square)
        else {
            continue;
        };
        let Some(double_arg) =
            extract_scaled_double_angle_sin_square_target_root(ctx, doubled_square)
        else {
            continue;
        };
        if compare_expr(ctx, base_arg, double_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_cos_fourth_power_reduction_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (reduced_expr, pow4_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(arg) = extract_direct_cos_fourth_power_reduction_target_root(ctx, reduced_expr)
        else {
            continue;
        };
        let expected = build_plain_trig_pow4_root(ctx, BuiltinFn::Cos, arg);
        if compare_expr(ctx, pow4_expr, expected) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_unordered_cos_arg_pair_up_to_sign_root(
    ctx: &Context,
    lhs_a: ExprId,
    lhs_b: ExprId,
    rhs_a: ExprId,
    rhs_b: ExprId,
) -> bool {
    (matches_expr_or_negation_root(ctx, lhs_a, rhs_a)
        && matches_expr_or_negation_root(ctx, lhs_b, rhs_b))
        || (matches_expr_or_negation_root(ctx, lhs_a, rhs_b)
            && matches_expr_or_negation_root(ctx, lhs_b, rhs_a))
}

pub(super) fn canonicalize_even_cos_arg_root(ctx: &mut Context, arg: ExprId) -> ExprId {
    let (coeff, base) = extract_coef_and_base(ctx, arg);
    if coeff < BigRational::zero() {
        build_coef_times_base(ctx, &(-coeff), base)
    } else {
        arg
    }
}

pub(super) fn build_plain_trig_sin_cos_product_root(
    ctx: &mut Context,
    sin_arg: ExprId,
    cos_arg: ExprId,
) -> ExprId {
    let sin_expr = ctx.call_builtin(BuiltinFn::Sin, vec![sin_arg]);
    let cos_expr = ctx.call_builtin(BuiltinFn::Cos, vec![cos_arg]);
    build_mul_expr_from_factors_root(ctx, &[sin_expr, cos_expr])
}

pub(super) fn build_scaled_trig_sin_cos_product_root(
    ctx: &mut Context,
    sin_arg: ExprId,
    cos_arg: ExprId,
) -> ExprId {
    let two = ctx.num(2);
    let product = build_plain_trig_sin_cos_product_root(ctx, sin_arg, cos_arg);
    smart_mul(ctx, two, product)
}

pub(super) fn extract_scaled_trig_cos_cos_product_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    let mut numeric_coeff = BigRational::one();
    let mut first_cos_arg = None;
    let mut second_cos_arg = None;

    for factor in factors {
        if let Expr::Number(n) = ctx.get(factor) {
            numeric_coeff *= n.clone();
            continue;
        }

        let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, factor) else {
            return None;
        };
        if first_cos_arg.is_none() {
            first_cos_arg = Some(arg);
        } else if second_cos_arg.is_none() {
            second_cos_arg = Some(arg);
        } else {
            return None;
        }
    }

    if numeric_coeff != BigRational::from_integer(2.into()) {
        return None;
    }

    Some((first_cos_arg?, second_cos_arg?))
}

pub(super) fn extract_scaled_trig_sin_cos_product_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
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

    if numeric_coeff != BigRational::from_integer(2.into()) {
        return None;
    }

    Some((sin_arg?, cos_arg?))
}

pub(super) fn extract_unit_trig_pow2_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<(&'static str, ExprId)> {
    let (coeff, name, arg) = extract_coeff_trig_pow2(ctx, expr)?;
    (coeff == BigRational::one()).then_some((name, arg))
}

pub(super) fn extract_unit_trig_pow_root(
    ctx: &Context,
    expr: ExprId,
    expected_fn: BuiltinFn,
    expected_power: i64,
) -> Option<ExprId> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *exponent) != Some(expected_power) {
        return None;
    }
    let (trig_fn, arg) = extract_plain_sin_or_cos_arg_root(ctx, *base)?;
    (trig_fn == expected_fn).then_some(arg)
}

pub(super) fn matches_direct_negative_double_cos_square_diff_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (square_diff, negative_cos) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(square_arg) = extract_mixed_sign_trig_square_difference_arg_root(ctx, square_diff)
        else {
            continue;
        };
        let Some(double_angle_arg) = extract_negative_cos_double_angle_arg_root(ctx, negative_cos)
        else {
            continue;
        };
        if compare_expr(ctx, square_arg, double_angle_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

fn extract_direct_cos_minus_sin_square_diff_target_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut positive_cos_arg = None;
    let mut negative_sin_arg = None;

    for (term_expr, term_sign) in view.terms {
        let (coeff, trig_name, arg, effective_sign) =
            extract_signed_numeric_trig_pow2(ctx, term_expr, term_sign)?;
        if coeff != BigRational::one() {
            return None;
        }
        match (trig_name, effective_sign) {
            ("cos", Sign::Pos) => {
                if positive_cos_arg.is_some() {
                    return None;
                }
                positive_cos_arg = Some(arg);
            }
            ("sin", Sign::Neg) => {
                if negative_sin_arg.is_some() {
                    return None;
                }
                negative_sin_arg = Some(arg);
            }
            _ => return None,
        }
    }

    let positive_cos_arg = positive_cos_arg?;
    let negative_sin_arg = negative_sin_arg?;
    (compare_expr(ctx, positive_cos_arg, negative_sin_arg) == Ordering::Equal)
        .then_some(positive_cos_arg)
}

pub(super) fn extract_direct_positive_double_cos_square_diff_target_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut cos_sq_arg = None;
    let mut saw_negative_one = false;

    for (term_expr, term_sign) in view.terms {
        if extract_i64_integer(ctx, term_expr) == Some(1) {
            if term_sign != Sign::Neg || saw_negative_one {
                return None;
            }
            saw_negative_one = true;
            continue;
        }

        let (coeff, trig_name, arg, effective_sign) =
            extract_signed_numeric_trig_pow2(ctx, term_expr, term_sign)?;
        if trig_name != "cos"
            || effective_sign != Sign::Pos
            || coeff != BigRational::from_integer(2.into())
            || cos_sq_arg.is_some()
        {
            return None;
        }
        cos_sq_arg = Some(arg);
    }

    saw_negative_one.then_some(cos_sq_arg?).or(None)
}

pub(super) fn matches_direct_positive_double_cos_square_diff_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (square_diff, positive_cos) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(square_arg) =
            extract_direct_positive_double_cos_square_diff_target_root(ctx, square_diff)
        else {
            continue;
        };
        let Some(double_angle_arg) = extract_positive_cos_double_angle_arg_root(ctx, positive_cos)
        else {
            continue;
        };
        if compare_expr(ctx, square_arg, double_angle_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_cos_minus_sin_square_diff_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (square_diff, positive_cos) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(square_arg) =
            extract_direct_cos_minus_sin_square_diff_target_root(ctx, square_diff)
        else {
            continue;
        };
        let Some(double_angle_arg) = extract_positive_cos_double_angle_arg_root(ctx, positive_cos)
        else {
            continue;
        };
        if compare_expr(ctx, square_arg, double_angle_arg) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_trig_cubic_cosine_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    crate::rules::arithmetic::try_build_direct_trig_cos_double_angle_polynomial_equivalence_rewrite(
        ctx, lhs_core, rhs_core,
    )
    .is_some()
        || crate::rules::arithmetic::try_build_direct_trig_sine_product_cubic_equivalence_rewrite(
            ctx, lhs_core, rhs_core,
        )
        .is_some()
}

fn extract_direct_reciprocal_trig_product_one_arg_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    for (plain_factor, reciprocal_factor) in [(factors[0], factors[1]), (factors[1], factors[0])] {
        let Expr::Function(plain_fn_id, plain_args) = ctx.get(plain_factor) else {
            continue;
        };
        if plain_args.len() != 1 {
            continue;
        }
        let (plain_fn, plain_arg) = if ctx.is_builtin(*plain_fn_id, BuiltinFn::Sin) {
            (BuiltinFn::Sin, plain_args[0])
        } else if ctx.is_builtin(*plain_fn_id, BuiltinFn::Cos) {
            (BuiltinFn::Cos, plain_args[0])
        } else if ctx.is_builtin(*plain_fn_id, BuiltinFn::Tan) {
            (BuiltinFn::Tan, plain_args[0])
        } else {
            continue;
        };
        let Expr::Function(fn_id, args) = ctx.get(reciprocal_factor) else {
            continue;
        };
        let reciprocal_fn = if ctx.is_builtin(*fn_id, BuiltinFn::Csc) {
            BuiltinFn::Csc
        } else if ctx.is_builtin(*fn_id, BuiltinFn::Sec) {
            BuiltinFn::Sec
        } else if ctx.is_builtin(*fn_id, BuiltinFn::Cot) {
            BuiltinFn::Cot
        } else {
            continue;
        };
        if args.len() != 1 {
            continue;
        }
        if compare_expr(ctx, plain_arg, args[0]) != Ordering::Equal {
            continue;
        }

        let matches = matches!(
            (plain_fn, reciprocal_fn),
            (BuiltinFn::Sin, BuiltinFn::Csc)
                | (BuiltinFn::Cos, BuiltinFn::Sec)
                | (BuiltinFn::Tan, BuiltinFn::Cot)
        );
        if matches {
            return Some(plain_arg);
        }
    }

    None
}

pub(super) fn matches_direct_reciprocal_trig_product_one_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (product_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        if !matches!(ctx.get(target_expr), Expr::Number(n) if n.is_one()) {
            continue;
        }
        if extract_direct_reciprocal_trig_product_one_arg_root(ctx, product_expr).is_some() {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_trig_reciprocal_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Expr::Div(numerator, denominator) = ctx.get(source).clone() else {
            continue;
        };
        if extract_i64_integer(ctx, numerator) != Some(1) {
            continue;
        }

        for (builtin, reciprocal_builtin) in [
            (BuiltinFn::Cos, BuiltinFn::Sec),
            (BuiltinFn::Sin, BuiltinFn::Csc),
        ] {
            let Some(base_arg) = extract_unary_builtin_arg_root(ctx, denominator, builtin) else {
                continue;
            };
            let Some(target_arg) = extract_unary_builtin_arg_root(ctx, target, reciprocal_builtin)
            else {
                continue;
            };
            if compare_expr(ctx, base_arg, target_arg) == Ordering::Equal {
                return true;
            }
        }
    }

    false
}

pub(super) fn matches_direct_trig_ratio_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Expr::Div(numerator, denominator) = ctx.get(source).clone() else {
            continue;
        };

        for (num_builtin, den_builtin, target_builtin) in [
            (BuiltinFn::Sin, BuiltinFn::Cos, BuiltinFn::Tan),
            (BuiltinFn::Cos, BuiltinFn::Sin, BuiltinFn::Cot),
        ] {
            let Some(num_arg) = extract_unary_builtin_arg_root(ctx, numerator, num_builtin) else {
                continue;
            };
            let Some(den_arg) = extract_unary_builtin_arg_root(ctx, denominator, den_builtin)
            else {
                continue;
            };
            if compare_expr(ctx, num_arg, den_arg) != Ordering::Equal {
                continue;
            }
            let Some(target_arg) = extract_unary_builtin_arg_root(ctx, target, target_builtin)
            else {
                continue;
            };
            if compare_expr(ctx, num_arg, target_arg) == Ordering::Equal {
                return true;
            }
        }
    }

    false
}

pub(super) fn matches_direct_cos_product_telescoping_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite_match) = crate::try_rewrite_cos_product_telescoping_expr(ctx, source)
        else {
            continue;
        };

        if compare_expr(ctx, rewrite_match.rewritten, target) == Ordering::Equal {
            return true;
        }

        let rewritten_normalized =
            cas_math::canonical_forms::normalize_core(ctx, rewrite_match.rewritten);
        let target_normalized = cas_math::canonical_forms::normalize_core(ctx, target);
        if compare_expr(ctx, rewritten_normalized, target_normalized) == Ordering::Equal {
            return true;
        }
    }

    false
}

pub(super) fn matches_direct_nested_zero_trig_ratio_or_reciprocal_residual_pair_root(
    ctx: &mut Context,
    expr: ExprId,
    trig_ratio_only: bool,
) -> bool {
    let matches_pair = |ctx: &mut Context, lhs: ExprId, rhs: ExprId| {
        if trig_ratio_only {
            matches_direct_trig_ratio_pair_root(ctx, lhs, rhs)
        } else {
            matches_direct_trig_reciprocal_pair_root(ctx, lhs, rhs)
        }
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

pub(super) fn matches_direct_trig_inverse_composition_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source_expr, target_expr) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(rewrite) = try_rewrite_trig_inverse_composition_expr(ctx, source_expr) else {
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
        let difference = ctx.add(Expr::Sub(rewritten, target));
        if cas_ast::count_nodes(ctx, difference) <= 36
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

pub(super) fn extract_plain_trig_product_pair_args_root(
    ctx: &mut Context,
    expr: ExprId,
    trig_fn: BuiltinFn,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let lhs_arg = extract_unary_builtin_arg_root(ctx, factors[0], trig_fn)?;
    let rhs_arg = extract_unary_builtin_arg_root(ctx, factors[1], trig_fn)?;
    Some((lhs_arg, rhs_arg))
}

pub(super) fn extract_plain_mixed_sin_cos_product_pair_args_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() != 2 {
        return None;
    }

    let (lhs_fn, lhs_arg) = extract_plain_sin_or_cos_arg_root(ctx, factors[0])?;
    let (rhs_fn, rhs_arg) = extract_plain_sin_or_cos_arg_root(ctx, factors[1])?;
    match (lhs_fn, rhs_fn) {
        (BuiltinFn::Sin, BuiltinFn::Cos) => Some((lhs_arg, rhs_arg)),
        (BuiltinFn::Cos, BuiltinFn::Sin) => Some((rhs_arg, lhs_arg)),
        _ => None,
    }
}

pub(super) fn extract_scaled_cos_square_sine_term_arg_root(
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
        if let Some((BuiltinFn::Sin, arg)) = extract_plain_sin_or_cos_arg_root(ctx, factor) {
            if sin_arg.is_some() {
                return None;
            }
            sin_arg = Some(arg);
            continue;
        }

        let Expr::Pow(base, exponent) = ctx.get(factor) else {
            return None;
        };
        let Expr::Number(n) = ctx.get(*exponent) else {
            return None;
        };
        if *n != BigRational::from_integer(2.into()) {
            return None;
        }
        let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, *base) else {
            return None;
        };
        if cos_arg.is_some() {
            return None;
        }
        cos_arg = Some(arg);
    }

    let (Some(sin_arg), Some(cos_arg)) = (sin_arg, cos_arg) else {
        return None;
    };
    (numeric_coeff == BigRational::from_integer(4.into())
        && compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal)
        .then_some(sin_arg)
}

pub(super) fn matches_direct_trig_cubic_cosine_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut sin_product_arg = None;
    let mut cos_linear = None;
    let mut cos_cubic = None;

    for (term_expr, term_sign) in view.terms {
        if let Some(arg) = extract_scaled_sin_double_angle_sine_term_arg_root(ctx, term_expr) {
            if sin_product_arg.is_some() {
                return false;
            }
            sin_product_arg = Some(arg);
            continue;
        }

        let (mut coeff, base) = extract_coef_and_base(ctx, term_expr);
        if term_sign == Sign::Neg {
            coeff = -coeff;
        }

        if let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, base) {
            if cos_linear.is_some() {
                return false;
            }
            cos_linear = Some((coeff, arg));
            continue;
        }

        let Expr::Pow(power_base, exponent) = ctx.get(base) else {
            return false;
        };
        let Expr::Number(n) = ctx.get(*exponent) else {
            return false;
        };
        if *n != BigRational::from_integer(3.into()) {
            return false;
        }
        let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, *power_base)
        else {
            return false;
        };
        if cos_cubic.is_some() {
            return false;
        }
        cos_cubic = Some((coeff, arg));
    }

    let (
        Some(sin_product_arg),
        Some((cos_linear_coeff, cos_linear_arg)),
        Some((cos_cubic_coeff, cos_cubic_arg)),
    ) = (sin_product_arg, cos_linear, cos_cubic)
    else {
        return false;
    };

    compare_expr(ctx, sin_product_arg, cos_linear_arg) == Ordering::Equal
        && compare_expr(ctx, sin_product_arg, cos_cubic_arg) == Ordering::Equal
        && cos_linear_coeff == BigRational::from_integer((-4).into())
        && cos_cubic_coeff == BigRational::from_integer(4.into())
}

pub(super) fn matches_direct_trig_ratio_alias_pair_root(
    ctx: &mut Context,
    lhs_core: ExprId,
    rhs_core: ExprId,
) -> bool {
    for (source, target) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Expr::Div(numerator, denominator) = ctx.get(source).clone() else {
            continue;
        };

        for (num_builtin, den_builtin, target_builtin) in [
            (BuiltinFn::Sin, BuiltinFn::Cos, BuiltinFn::Tan),
            (BuiltinFn::Cos, BuiltinFn::Sin, BuiltinFn::Cot),
        ] {
            let Some(num_arg) = extract_unary_builtin_arg_root(ctx, numerator, num_builtin) else {
                continue;
            };
            let Some(den_arg) = extract_unary_builtin_arg_root(ctx, denominator, den_builtin)
            else {
                continue;
            };
            let Some(target_arg) = extract_unary_builtin_arg_root(ctx, target, target_builtin)
            else {
                continue;
            };
            let Some(num_base_arg) = extract_double_angle_arg_relaxed(ctx, num_arg) else {
                continue;
            };
            let Some(den_base_arg) = extract_double_angle_arg_relaxed(ctx, den_arg) else {
                continue;
            };
            let Some(target_base_arg) = extract_double_angle_arg_relaxed(ctx, target_arg) else {
                continue;
            };

            if compare_expr(ctx, num_arg, target_arg) == Ordering::Equal
                && compare_expr(ctx, den_arg, num_arg) != Ordering::Equal
                && compare_expr(ctx, num_base_arg, den_base_arg) == Ordering::Equal
                && compare_expr(ctx, num_base_arg, target_base_arg) == Ordering::Equal
            {
                return true;
            }
        }
    }

    false
}

pub(super) fn matches_direct_nested_zero_trig_ratio_alias_residual_pair_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let matches_pair = |ctx: &mut Context, lhs: ExprId, rhs: ExprId| {
        matches_direct_trig_ratio_alias_pair_root(ctx, lhs, rhs)
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

pub(super) fn matches_direct_tan_cot_product_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    for product_index in 0..view.terms.len() {
        let (product_expr, product_sign) = normalize_signed_add_term_root(
            ctx,
            view.terms[product_index].0,
            view.terms[product_index].1,
        );
        let (other_expr, other_sign) = normalize_signed_add_term_root(
            ctx,
            view.terms[1 - product_index].0,
            view.terms[1 - product_index].1,
        );
        if product_sign == other_sign || extract_i64_integer(ctx, other_expr) != Some(1) {
            continue;
        }

        let factors = flatten_mul_chain(ctx, product_expr);
        if factors.len() != 2 {
            continue;
        }

        for (first, second) in [(factors[0], factors[1]), (factors[1], factors[0])] {
            let Some(tan_arg) = extract_unary_builtin_arg_root(ctx, first, BuiltinFn::Tan) else {
                continue;
            };
            let Some(cot_arg) = extract_unary_builtin_arg_root(ctx, second, BuiltinFn::Cot) else {
                continue;
            };
            if compare_expr(ctx, tan_arg, cot_arg) == Ordering::Equal {
                return true;
            }
        }
    }

    false
}

pub(super) fn matches_direct_tan_cot_sec_csc_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut tan_arg = None;
    let mut cot_arg = None;
    let mut product_arg = None;
    let mut product_sign = None;

    for (term_expr, term_sign) in view.terms {
        let (term_expr, term_sign) = normalize_signed_add_term_root(ctx, term_expr, term_sign);
        if let Some(arg) = extract_unary_builtin_arg_root(ctx, term_expr, BuiltinFn::Tan) {
            if tan_arg.is_some() || term_sign != Sign::Pos {
                return false;
            }
            tan_arg = Some(arg);
            continue;
        }
        if let Some(arg) = extract_unary_builtin_arg_root(ctx, term_expr, BuiltinFn::Cot) {
            if cot_arg.is_some() || term_sign != Sign::Pos {
                return false;
            }
            cot_arg = Some(arg);
            continue;
        }

        let factors = flatten_mul_chain(ctx, term_expr);
        if factors.len() != 2 || product_arg.is_some() || term_sign != Sign::Neg {
            return false;
        }
        for (first, second) in [(factors[0], factors[1]), (factors[1], factors[0])] {
            let Some(sec_arg) = extract_unary_builtin_arg_root(ctx, first, BuiltinFn::Sec) else {
                continue;
            };
            let Some(csc_arg) = extract_unary_builtin_arg_root(ctx, second, BuiltinFn::Csc) else {
                continue;
            };
            if compare_expr(ctx, sec_arg, csc_arg) == Ordering::Equal {
                product_arg = Some(sec_arg);
                product_sign = Some(term_sign);
                break;
            }
        }
        if product_arg.is_none() {
            return false;
        }
    }

    let (Some(tan_arg), Some(cot_arg), Some(product_arg), Some(Sign::Neg)) =
        (tan_arg, cot_arg, product_arg, product_sign)
    else {
        return false;
    };

    compare_expr(ctx, tan_arg, cot_arg) == Ordering::Equal
        && compare_expr(ctx, tan_arg, product_arg) == Ordering::Equal
}

pub(super) fn matches_direct_positive_double_cos_square_diff_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut cos_sq_arg = None;
    let mut sin_sq_arg = None;
    let mut cos_double_arg = None;

    for (term_expr, term_sign) in view.terms {
        if let Some(arg) = extract_positive_cos_double_angle_arg_root(ctx, term_expr) {
            if term_sign != Sign::Neg || cos_double_arg.is_some() {
                return false;
            }
            cos_double_arg = Some(arg);
            continue;
        }

        let Some((coeff, trig_name, arg, effective_sign)) =
            extract_signed_numeric_trig_pow2(ctx, term_expr, term_sign)
        else {
            return false;
        };
        if coeff != BigRational::one() {
            return false;
        }

        match (trig_name, effective_sign) {
            ("cos", Sign::Pos) if cos_sq_arg.is_none() => cos_sq_arg = Some(arg),
            ("sin", Sign::Neg) if sin_sq_arg.is_none() => sin_sq_arg = Some(arg),
            _ => return false,
        }
    }

    let (Some(cos_sq_arg), Some(sin_sq_arg), Some(cos_double_arg)) =
        (cos_sq_arg, sin_sq_arg, cos_double_arg)
    else {
        return false;
    };

    compare_expr(ctx, cos_sq_arg, sin_sq_arg) == Ordering::Equal
        && compare_expr(ctx, cos_sq_arg, cos_double_arg) == Ordering::Equal
}

pub(super) fn matches_direct_negative_double_cos_square_diff_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 3 {
        return false;
    }

    let mut sin_sq_arg = None;
    let mut cos_sq_arg = None;
    let mut cos_double_arg = None;

    for (term_expr, term_sign) in view.terms {
        if let Some(arg) = extract_positive_cos_double_angle_arg_root(ctx, term_expr) {
            if term_sign != Sign::Pos || cos_double_arg.is_some() {
                return false;
            }
            cos_double_arg = Some(arg);
            continue;
        }

        let Some((coeff, trig_name, arg, effective_sign)) =
            extract_signed_numeric_trig_pow2(ctx, term_expr, term_sign)
        else {
            return false;
        };
        if coeff != BigRational::one() {
            return false;
        }

        match (trig_name, effective_sign) {
            ("sin", Sign::Pos) if sin_sq_arg.is_none() => sin_sq_arg = Some(arg),
            ("cos", Sign::Neg) if cos_sq_arg.is_none() => cos_sq_arg = Some(arg),
            _ => return false,
        }
    }

    let (Some(sin_sq_arg), Some(cos_sq_arg), Some(cos_double_arg)) =
        (sin_sq_arg, cos_sq_arg, cos_double_arg)
    else {
        return false;
    };

    compare_expr(ctx, sin_sq_arg, cos_sq_arg) == Ordering::Equal
        && compare_expr(ctx, sin_sq_arg, cos_double_arg) == Ordering::Equal
}

pub(super) fn matches_direct_inverse_trig_composition_zero_identity_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }

    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }
    if !expr_contains_any_builtin_local(
        ctx,
        expr,
        &[
            BuiltinFn::Asin,
            BuiltinFn::Arcsin,
            BuiltinFn::Atan,
            BuiltinFn::Arctan,
        ],
    ) {
        return false;
    }

    let (lhs_term, rhs_term) = match (view.terms[0].1, view.terms[1].1) {
        (Sign::Pos, Sign::Neg) => (view.terms[0].0, view.terms[1].0),
        (Sign::Neg, Sign::Pos) => (view.terms[1].0, view.terms[0].0),
        _ => return false,
    };

    for (source_expr, target_expr) in [(lhs_term, rhs_term), (rhs_term, lhs_term)] {
        let Some(plan) =
            cas_math::inverse_trig_composition_support::try_plan_inverse_trig_composition_expr(
                ctx,
                source_expr,
                false,
                false,
            )
        else {
            continue;
        };
        let rewritten = strip_multiplicative_one_root(ctx, plan.rewritten);
        let target = strip_multiplicative_one_root(ctx, target_expr);
        if compare_expr(ctx, rewritten, target) == Ordering::Equal {
            return true;
        }
        let difference = ctx.add(Expr::Sub(rewritten, target));
        if cas_ast::count_nodes(ctx, difference) <= 36
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

pub(super) fn try_standard_assumed_dyadic_cos_product_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let Expr::Mul(_, _) = ctx.get(expr) else {
        return None;
    };

    let plan = cas_math::trig_multi_angle_support::try_plan_dyadic_cos_product_with_policy(
        ctx,
        expr,
        matches!(
            options.shared.semantics.domain_mode,
            crate::DomainMode::Assume
        ),
        matches!(
            options.shared.semantics.domain_mode,
            crate::DomainMode::Strict
        ),
    )?;
    if !matches!(
        plan.policy,
        cas_math::trig_dyadic_policy_support::DyadicSinNonzeroPolicyDecision::Apply {
            assume_sin_nonzero: true
        }
    ) {
        return None;
    }

    let parent_ctx = build_root_shortcut_parent_ctx(options, ctx, expr);
    let rule = crate::rules::trigonometry::DyadicCosProductToSinRule;
    let rewrite = crate::rule::Rule::apply(&rule, ctx, expr, &parent_ctx)?;
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Dyadic Cos Product",
        collect_steps,
    ))
}

pub(super) fn expr_contains_reciprocal_trig_builtin_local(ctx: &Context, expr: ExprId) -> bool {
    expr_contains_any_builtin_local(
        ctx,
        expr,
        &[
            BuiltinFn::Tan,
            BuiltinFn::Cot,
            BuiltinFn::Sec,
            BuiltinFn::Csc,
        ],
    )
}

pub(super) fn classify_root_exact_zero_multiterm_trig_numeric_subset_status(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> Option<&'static str> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let terms = AddView::from_expr(ctx, expr).terms;
    if !(5..=17).contains(&terms.len()) {
        return None;
    }

    let has_trig_or_hyperbolic = terms
        .iter()
        .any(|(term, _)| expr_contains_trig_or_hyperbolic_builtin_local(ctx, *term));
    let has_non_trig_or_hyperbolic = terms
        .iter()
        .any(|(term, _)| !expr_contains_trig_or_hyperbolic_builtin_local(ctx, *term));
    if !has_trig_or_hyperbolic || !has_non_trig_or_hyperbolic {
        return None;
    }

    let Some((subset_expr, partner_expr)) =
        extract_small_trig_or_hyperbolic_numeric_subset_root(ctx, expr)
    else {
        return Some("no_candidate");
    };

    if !is_supported_small_trig_zero_pair_side_root(ctx, subset_expr, true) {
        return Some("subset_not_supported_pair_side");
    }

    if !multiterm_trig_numeric_subset_rewrites_to_zero_runtime_safe(options, ctx, subset_expr) {
        return Some("subset_not_isolated_zero");
    }

    if !is_supported_nested_zero_child_partner(ctx, partner_expr) {
        return Some("partner_unsupported");
    }

    if !supported_nested_zero_partner_rewrites_to_zero(options, ctx, partner_expr) {
        return Some("partner_not_zero");
    }

    Some("candidate_ready")
}

pub(super) fn profile_root_exact_zero_multiterm_trig_numeric_subset_status(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    prefix: &'static str,
) {
    if !crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled() {
        return;
    }

    let Some(bucket) =
        classify_root_exact_zero_multiterm_trig_numeric_subset_status(options, ctx, expr)
    else {
        return;
    };

    let name = match (prefix, bucket) {
        (
            "root.exact_zero.entry.multiterm_trig_numeric_subset",
            "no_candidate",
        ) => "root.exact_zero.entry.multiterm_trig_numeric_subset.no_candidate",
        (
            "root.exact_zero.entry.multiterm_trig_numeric_subset",
            "subset_not_supported_pair_side",
        ) => {
            "root.exact_zero.entry.multiterm_trig_numeric_subset.subset_not_supported_pair_side"
        }
        (
            "root.exact_zero.entry.multiterm_trig_numeric_subset",
            "subset_not_isolated_zero",
        ) => "root.exact_zero.entry.multiterm_trig_numeric_subset.subset_not_isolated_zero",
        (
            "root.exact_zero.entry.multiterm_trig_numeric_subset",
            "partner_unsupported",
        ) => "root.exact_zero.entry.multiterm_trig_numeric_subset.partner_unsupported",
        (
            "root.exact_zero.entry.multiterm_trig_numeric_subset",
            "partner_not_zero",
        ) => "root.exact_zero.entry.multiterm_trig_numeric_subset.partner_not_zero",
        (
            "root.exact_zero.entry.multiterm_trig_numeric_subset",
            "candidate_ready",
        ) => "root.exact_zero.entry.multiterm_trig_numeric_subset.candidate_ready",
        (
            "phase.core.post_pass.exact_zero.multiterm_trig_numeric_subset",
            "no_candidate",
        ) => "phase.core.post_pass.exact_zero.multiterm_trig_numeric_subset.no_candidate",
        (
            "phase.core.post_pass.exact_zero.multiterm_trig_numeric_subset",
            "subset_not_supported_pair_side",
        ) => {
            "phase.core.post_pass.exact_zero.multiterm_trig_numeric_subset.subset_not_supported_pair_side"
        }
        (
            "phase.core.post_pass.exact_zero.multiterm_trig_numeric_subset",
            "subset_not_isolated_zero",
        ) => "phase.core.post_pass.exact_zero.multiterm_trig_numeric_subset.subset_not_isolated_zero",
        (
            "phase.core.post_pass.exact_zero.multiterm_trig_numeric_subset",
            "partner_unsupported",
        ) => "phase.core.post_pass.exact_zero.multiterm_trig_numeric_subset.partner_unsupported",
        (
            "phase.core.post_pass.exact_zero.multiterm_trig_numeric_subset",
            "partner_not_zero",
        ) => "phase.core.post_pass.exact_zero.multiterm_trig_numeric_subset.partner_not_zero",
        (
            "phase.core.post_pass.exact_zero.multiterm_trig_numeric_subset",
            "candidate_ready",
        ) => "phase.core.post_pass.exact_zero.multiterm_trig_numeric_subset.candidate_ready",
        _ => return,
    };
    record_profiled_orchestrator_route_hit(ctx, expr, name);
}

pub(super) fn try_extract_multiterm_trig_numeric_subset_zero_chunks_root(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let (subset_expr, partner_expr) =
        extract_small_trig_or_hyperbolic_numeric_subset_root(ctx, expr)?;
    if !is_supported_small_trig_zero_pair_side_root(ctx, subset_expr, true) {
        return None;
    }
    if !multiterm_trig_numeric_subset_rewrites_to_zero_runtime_safe(options, ctx, subset_expr) {
        return None;
    }
    if !is_supported_nested_zero_child_partner(ctx, partner_expr) {
        return None;
    }
    if !supported_nested_zero_partner_rewrites_to_zero(options, ctx, partner_expr) {
        return None;
    }
    Some((subset_expr, partner_expr))
}

pub(super) fn try_standard_multiterm_trig_numeric_subset_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    fn multiterm_trig_numeric_subset_chunk_steps_should_stay_compact(
        ctx: &Context,
        subset_expr: ExprId,
        partner_expr: ExprId,
    ) -> bool {
        !expr_contains_trig_or_hyperbolic_builtin_local(ctx, partner_expr)
            && cas_ast::count_nodes(ctx, partner_expr) > 24
            && cas_ast::count_nodes(ctx, subset_expr) <= 24
    }

    let Some((subset_expr, partner_expr)) =
        extract_small_trig_or_hyperbolic_numeric_subset_root(ctx, expr)
    else {
        record_profiled_orchestrator_route_hit(
            ctx,
            expr,
            "root.multiterm_trig_subset.status.no_candidate",
        );
        return None;
    };

    if !is_supported_small_trig_zero_pair_side_root(ctx, subset_expr, true) {
        record_profiled_orchestrator_route_hit(
            ctx,
            expr,
            "root.multiterm_trig_subset.status.subset_not_supported_pair_side",
        );
        return None;
    }
    if !multiterm_trig_numeric_subset_rewrites_to_zero_runtime_safe(options, ctx, subset_expr) {
        record_profiled_orchestrator_route_hit(
            ctx,
            expr,
            "root.multiterm_trig_subset.status.subset_not_isolated_zero",
        );
        return None;
    }
    if !is_supported_nested_zero_child_partner(ctx, partner_expr) {
        record_profiled_orchestrator_route_hit(
            ctx,
            expr,
            "root.multiterm_trig_subset.status.partner_unsupported",
        );
        return None;
    }
    if !supported_nested_zero_partner_rewrites_to_zero(options, ctx, partner_expr) {
        record_profiled_orchestrator_route_hit(
            ctx,
            expr,
            "root.multiterm_trig_subset.status.partner_not_zero",
        );
        return None;
    }

    record_profiled_orchestrator_route_hit(
        ctx,
        expr,
        "root.multiterm_trig_subset.status.candidate_ready",
    );

    let zero = ctx.num(0);
    let mut shortcut_steps = Vec::new();
    if collect_steps {
        if multiterm_trig_numeric_subset_chunk_steps_should_stay_compact(
            ctx,
            subset_expr,
            partner_expr,
        ) {
            let mut first_step = build_root_shortcut_compact_step(
                subset_expr,
                zero,
                "Collapse Exact Zero Additive Subexpression",
                "Collapse Exact Zero Additive Subexpression",
            );
            first_step.global_before = Some(expr);
            first_step.global_after = Some(partner_expr);
            shortcut_steps.push(first_step);

            let mut second_step = build_root_shortcut_compact_step(
                partner_expr,
                zero,
                "Collapse Exact Zero Additive Subexpression",
                "Collapse Exact Zero Additive Subexpression",
            );
            second_step.global_before = Some(partner_expr);
            second_step.global_after = Some(zero);
            shortcut_steps.push(second_step);
            return Some((zero, shortcut_steps));
        }
        if let Some(steps) = try_build_chunk_pair_zero_shortcut_steps_root(
            options,
            ctx,
            expr,
            subset_expr,
            partner_expr,
        ) {
            return Some((zero, steps));
        }
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
    }
    let _ = options;
    Some((zero, shortcut_steps))
}

pub(super) fn multiterm_trig_numeric_subset_rewrites_to_zero_runtime_safe(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    subset_expr: ExprId,
) -> bool {
    if expr_contains_trig_builtin_local(ctx, subset_expr)
        && !expr_contains_hyperbolic_builtin_local(ctx, subset_expr)
        && !expr_contains_log_builtin_local(ctx, subset_expr)
        && matches_direct_symbolic_trig_sum_to_product_zero_identity_root(ctx, subset_expr)
    {
        return true;
    }

    matches_direct_small_zero_identity_root(ctx, subset_expr)
        || matches_direct_hyperbolic_cosh_cubic_zero_identity_root(ctx, subset_expr)
        || try_standard_small_trig_zero_pair_shortcut(options, ctx, subset_expr, false).is_some()
        || child_isolated_exact_zero(options, ctx, subset_expr)
}

pub(super) fn try_standard_reciprocal_trig_zero_pair_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let matches_zero_pair = |ctx: &mut Context, lhs: ExprId, rhs: ExprId| {
        expr_contains_reciprocal_trig_builtin_local(ctx, lhs)
            && is_small_trig_or_hyperbolic_zero_child(options, ctx, lhs)
            && is_small_trig_or_hyperbolic_zero_child(options, ctx, rhs)
    };

    let matched = match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => {
            let lhs = *lhs;
            let rhs = *rhs;
            matches_zero_pair(ctx, lhs, rhs) || matches_zero_pair(ctx, rhs, lhs)
        }
        _ => false,
    };

    if !matched {
        return None;
    }

    let zero = ctx.num(0);
    if collect_steps && matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        if let Some(steps) = try_build_recursive_additive_zero_shortcut_steps(options, ctx, expr) {
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

pub(super) fn try_standard_small_trig_zero_pair_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        let add_term_count = AddView::from_expr(ctx, expr).terms.len();
        if add_term_count >= 8 && expr_contains_log_builtin_local(ctx, expr) {
            return None;
        }
    }

    let (lhs, rhs) = match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => (*lhs, *rhs),
        _ => return None,
    };
    if !matches!(ctx.get(lhs), Expr::Add(_, _) | Expr::Sub(_, _))
        || !matches!(ctx.get(rhs), Expr::Add(_, _) | Expr::Sub(_, _))
    {
        return None;
    }

    let matches_small_zero_identity = |ctx: &mut Context, child: ExprId| {
        matches_direct_small_zero_identity_root(ctx, child)
            || matches_direct_hyperbolic_cosh_cubic_zero_identity_root(ctx, child)
    };
    let matches_isolated_zero =
        |ctx: &mut Context, child: ExprId| child_isolated_exact_zero(options, ctx, child);
    let matches_pair_side = |ctx: &Context, child: ExprId| {
        is_supported_small_trig_zero_pair_side_root(ctx, child, !collect_steps)
    };
    let matches_supported_zero_partner = |ctx: &mut Context, trig_side: ExprId, partner: ExprId| {
        matches_pair_side(ctx, trig_side)
            && matches_isolated_zero(ctx, trig_side)
            && supported_nested_zero_partner_rewrites_to_zero(options, ctx, partner)
    };

    let matches_zero_pair = |ctx: &mut Context, lhs: ExprId, rhs: ExprId| {
        matches_pair_side(ctx, lhs)
            && (matches_small_zero_identity(ctx, lhs) || matches_isolated_zero(ctx, lhs))
            && ((matches_pair_side(ctx, rhs)
                && (matches_small_zero_identity(ctx, rhs) || matches_isolated_zero(ctx, rhs)))
                || matches_isolated_zero(ctx, rhs))
    };

    let matched = matches_zero_pair(ctx, lhs, rhs)
        || matches_zero_pair(ctx, rhs, lhs)
        || matches_supported_zero_partner(ctx, lhs, rhs)
        || matches_supported_zero_partner(ctx, rhs, lhs);

    if !matched {
        return None;
    }

    let zero = ctx.num(0);
    if collect_steps && matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        if let Some(steps) = try_build_recursive_additive_zero_shortcut_steps(options, ctx, expr) {
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

pub(super) fn try_standard_direct_small_trig_zero_child_with_supported_zero_partner_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (lhs, rhs) = match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => (*lhs, *rhs),
        _ => return None,
    };

    for (direct_zero_child, partner) in [(lhs, rhs), (rhs, lhs)] {
        if !is_supported_nested_zero_child_partner(ctx, partner) {
            continue;
        }
        if expr_contains_trig_or_hyperbolic_builtin_local(ctx, partner) {
            continue;
        }
        if !expr_contains_trig_or_hyperbolic_builtin_local(ctx, direct_zero_child) {
            continue;
        }
        if !(matches_direct_small_zero_identity_root(ctx, direct_zero_child)
            || matches_direct_hyperbolic_cosh_cubic_zero_identity_root(ctx, direct_zero_child)
            || classify_root_exact_zero_multiterm_trig_numeric_subset_status(
                options,
                ctx,
                direct_zero_child,
            ) == Some("candidate_ready")
            || try_standard_small_trig_zero_pair_shortcut(options, ctx, direct_zero_child, false)
                .is_some())
        {
            continue;
        }
        if !supported_nested_zero_partner_rewrites_to_zero(options, ctx, partner) {
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

pub(super) fn try_standard_direct_trig_mixed_zero_pair_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (lhs, rhs) = match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) => (*lhs, *rhs),
        _ => return None,
    };

    let matches_zero_pair = |ctx: &mut Context, trig_side: ExprId, other_side: ExprId| {
        matches_narrow_trig_mixed_double_angle_zero_candidate_root(ctx, trig_side)
            && child_isolated_exact_zero(options, ctx, trig_side)
            && (matches_direct_small_zero_identity_root(ctx, other_side)
                || is_small_trig_or_hyperbolic_zero_child(options, ctx, other_side))
    };

    if !(matches_zero_pair(ctx, lhs, rhs) || matches_zero_pair(ctx, rhs, lhs)) {
        return None;
    }

    let zero = ctx.num(0);
    Some(run_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        zero,
        collect_steps,
    ))
}

pub(super) fn try_standard_direct_trig_power_reduction_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (rewritten, rule_name) = if let Some(arg) =
        extract_direct_cos_fourth_power_reduction_target_root(ctx, expr)
    {
        (
            build_plain_trig_pow4_root(ctx, BuiltinFn::Cos, arg),
            "Power Reduction Identity",
        )
    } else if let Some(arg) = extract_direct_sin_cos_square_product_reduction_target_root(ctx, expr)
    {
        (
            build_plain_sin_cos_square_product_root(ctx, arg),
            "Power Reduction Identity",
        )
    } else {
        return None;
    };

    Some(run_named_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        rewritten,
        "Power Reduction Identity",
        rule_name,
        collect_steps,
    ))
}

pub(super) fn try_standard_scaled_sin_fourth_power_reduction_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (lhs_core, rhs_core) =
        crate::rules::arithmetic::extract_two_term_core_difference(ctx, expr)?;

    for (scaled_term, reduced_term) in [(lhs_core, rhs_core), (rhs_core, lhs_core)] {
        let Some(scaled_arg) = extract_scaled_sin_fourth_power_target_root(ctx, scaled_term) else {
            continue;
        };
        let Some(reduced_arg) =
            extract_scaled_sin_fourth_power_reduction_target_root(ctx, reduced_term)
        else {
            continue;
        };
        if compare_expr(ctx, scaled_arg, reduced_arg) != Ordering::Equal {
            continue;
        }

        let zero = ctx.num(0);
        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            zero,
            "Power Reduction Identity",
            "Power Reduction Identity",
            collect_steps,
        ));
    }

    None
}

pub(super) fn try_standard_direct_positive_double_cos_square_diff_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let arg = extract_direct_positive_double_cos_square_diff_target_root(ctx, expr)?;
    let rewritten = build_positive_cos_double_angle_expr_root(ctx, arg);

    Some(run_named_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        rewritten,
        "Double Angle Expansion",
        "Double Angle Expansion",
        collect_steps,
    ))
}

pub(super) fn try_standard_direct_cos_square_diff_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (lhs, rhs) = match ctx.get(expr) {
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) => (*lhs, *rhs),
        _ => return None,
    };
    if !matches_direct_cos_square_diff_pair_root(ctx, lhs, rhs)
        && !matches_direct_negative_double_cos_square_diff_zero_identity_root(ctx, expr)
    {
        return None;
    }

    let zero = ctx.num(0);
    Some(run_rebuilt_root_shortcut_simplify(
        options,
        ctx,
        expr,
        zero,
        collect_steps,
    ))
}

pub(super) fn extract_direct_trig_sum_product_zero_cores_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    if let Some((lhs_core, rhs_core)) =
        extract_shared_additive_passthrough_sub_cores_root(ctx, expr)
    {
        return Some((lhs_core, rhs_core));
    }

    if let Some((lhs_core, rhs_core)) =
        crate::rules::arithmetic::extract_two_term_core_difference(ctx, expr)
    {
        return Some((lhs_core, rhs_core));
    }

    let terms = AddView::from_expr(ctx, expr).terms;
    if !(3..=6).contains(&terms.len()) {
        return None;
    }

    let full_mask = (1usize << terms.len()) - 1;
    let mut product_to_sum_fallback = None;
    for left_mask in 1..full_mask {
        let right_mask = full_mask ^ left_mask;
        if right_mask == 0 {
            continue;
        }

        let mut left_terms = Vec::new();
        let mut right_terms = Vec::new();
        for (index, term) in terms.iter().copied().enumerate() {
            if ((left_mask >> index) & 1) == 1 {
                left_terms.push(term);
            } else {
                right_terms.push(term);
            }
        }
        if left_terms.is_empty() || right_terms.is_empty() {
            continue;
        }

        let lhs_chunk = build_signed_sum_expr_root(ctx, &left_terms);
        let rhs_chunk = build_signed_sum_expr_root(ctx, &right_terms);
        if is_plain_two_term_sin_cos_sum_or_diff_root(ctx, lhs_chunk)
            && is_trig_sum_product_candidate_root(ctx, rhs_chunk)
        {
            return Some((lhs_chunk, rhs_chunk));
        }
        if product_to_sum_fallback.is_none()
            && is_trig_sum_product_candidate_root(ctx, lhs_chunk)
            && is_plain_two_term_sin_cos_sum_or_diff_root(ctx, rhs_chunk)
        {
            product_to_sum_fallback = Some((lhs_chunk, rhs_chunk));
        }
    }

    product_to_sum_fallback
}

pub(super) fn is_plain_two_term_sin_cos_sum_or_diff_root(ctx: &mut Context, expr: ExprId) -> bool {
    let terms = AddView::from_expr(ctx, expr).terms;
    terms.len() == 2
        && terms.iter().all(|(term_expr, _)| {
            let (_coeff, base) = extract_coef_and_base(ctx, *term_expr);
            extract_plain_sin_or_cos_arg_root(ctx, base).is_some()
        })
}

pub(super) fn is_trig_sum_product_candidate_root(ctx: &mut Context, expr: ExprId) -> bool {
    let (_coeff, base) = extract_coef_and_base(ctx, expr);
    let mut trig_factor_count = 0usize;
    for factor in flatten_mul_chain(ctx, base) {
        if extract_plain_sin_or_cos_arg_root(ctx, factor).is_some() {
            trig_factor_count += 1;
            continue;
        }
        if matches!(ctx.get(factor), Expr::Number(_)) {
            continue;
        }
        return false;
    }

    trig_factor_count >= 2
}

pub(super) fn try_standard_direct_trig_sum_product_zero_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let (lhs_core, rhs_core) = extract_direct_trig_sum_product_zero_cores_root(ctx, expr)?;
    if is_plain_two_term_sin_cos_sum_or_diff_root(ctx, lhs_core)
        && is_trig_sum_product_candidate_root(ctx, rhs_core)
    {
        let zero = ctx.num(0);
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            crate::rule::Rewrite::with_local(zero, "Aplicar suma a producto", expr, zero),
            "Aplicar suma a producto",
            collect_steps,
        ));
    }
    if is_trig_sum_product_candidate_root(ctx, lhs_core)
        && is_plain_two_term_sin_cos_sum_or_diff_root(ctx, rhs_core)
    {
        let zero = ctx.num(0);
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            crate::rule::Rewrite::with_local(zero, "Aplicar producto a suma", expr, zero),
            "Aplicar producto a suma",
            collect_steps,
        ));
    }

    let whole_expr_is_direct_trig_sum_product_zero =
        matches_direct_symbolic_trig_sum_to_product_zero_identity_root(ctx, expr)
            || matches_direct_trig_product_to_sum_zero_identity_root(ctx, expr);
    let lhs_has_sum_to_product_rewrite =
        try_rewrite_sum_to_product_contraction_expr(ctx, lhs_core).is_some();
    let lhs_has_product_to_sum_rewrite = try_rewrite_product_to_sum_expr(ctx, lhs_core).is_some()
        || rewrite_direct_trig_product_to_sum_double_angle_target_root(ctx, lhs_core).is_some();

    let rule_name = if rewrites_sum_to_product_target_root(ctx, lhs_core, rhs_core)
        || (whole_expr_is_direct_trig_sum_product_zero && lhs_has_sum_to_product_rewrite)
    {
        "Aplicar suma a producto"
    } else if rewrites_product_to_sum_target_root(ctx, lhs_core, rhs_core)
        || (whole_expr_is_direct_trig_sum_product_zero && lhs_has_product_to_sum_rewrite)
    {
        "Aplicar producto a suma"
    } else {
        return None;
    };

    let zero = ctx.num(0);
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::with_local(zero, rule_name, expr, zero),
        rule_name,
        collect_steps,
    ))
}

pub(super) fn canonicalize_even_cos_in_simple_expr_root(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let normalized_inner = canonicalize_even_cos_in_simple_expr_root(ctx, inner);
            if compare_expr(ctx, normalized_inner, inner) == Ordering::Equal {
                expr
            } else {
                ctx.add(Expr::Neg(normalized_inner))
            }
        }
        _ => {
            let Some((BuiltinFn::Cos, arg)) = extract_plain_sin_or_cos_arg_root(ctx, expr) else {
                return expr;
            };
            let normalized_arg = canonicalize_even_cos_arg_root(ctx, arg);
            if compare_expr(ctx, normalized_arg, arg) == Ordering::Equal {
                expr
            } else {
                ctx.call_builtin(BuiltinFn::Cos, vec![normalized_arg])
            }
        }
    }
}

pub(super) fn extract_scaled_plain_sin_or_cos_arg_root(
    ctx: &mut Context,
    expr: ExprId,
    expected_fn: BuiltinFn,
) -> Option<ExprId> {
    if let Some((trig_fn, arg)) = extract_plain_sin_or_cos_arg_root(ctx, expr) {
        return (trig_fn == expected_fn).then_some(arg);
    }

    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut trig_arg = None;
    let mut scalar_seen = false;
    for factor in factors {
        if let Some((trig_fn, arg)) = extract_plain_sin_or_cos_arg_root(ctx, factor) {
            if trig_fn != expected_fn || trig_arg.is_some() {
                return None;
            }
            trig_arg = Some(arg);
            continue;
        }

        match ctx.get(factor) {
            Expr::Number(_) if !scalar_seen => {
                scalar_seen = true;
            }
            _ => return None,
        }
    }

    trig_arg
}

fn extract_sin_arctan_source_arg_root(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Function(outer_fn, outer_args) = ctx.get(expr) else {
        return None;
    };
    if outer_args.len() != 1 || !ctx.is_builtin(*outer_fn, BuiltinFn::Sin) {
        return None;
    }

    let Expr::Function(inner_fn, inner_args) = ctx.get(outer_args[0]) else {
        return None;
    };
    if inner_args.len() != 1 {
        return None;
    }
    if ctx.is_builtin(*inner_fn, BuiltinFn::Arctan) || ctx.is_builtin(*inner_fn, BuiltinFn::Atan) {
        return Some(inner_args[0]);
    }

    None
}

fn extract_inverse_trig_ratio_anchor_base_root(ctx: &mut Context, expr: ExprId) -> Option<ExprId> {
    if let Some(base) = extract_sin_arctan_source_arg_root(ctx, expr) {
        return Some(base);
    }

    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let radicand = extract_square_root_base(ctx, den)?;
    let one = ctx.num(1);
    let two = ctx.num(2);
    let num_sq = ctx.add(Expr::Pow(num, two));
    let expected_radicand = ctx.add(Expr::Add(num_sq, one));
    (compare_expr(ctx, radicand, expected_radicand) == Ordering::Equal).then_some(num)
}

fn build_inverse_trig_ratio_anchor_product_root(
    ctx: &mut Context,
    base: ExprId,
    partner: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let two = ctx.num(2);
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let base_sq = ctx.add(Expr::Pow(base, two));
    let radicand = ctx.add(Expr::Add(base_sq, one));
    let sqrt = ctx.add(Expr::Pow(radicand, half));
    let numerator = build_mul_expr_from_factors_root(ctx, &[base, sqrt, partner]);
    ctx.add(Expr::Div(numerator, radicand))
}

fn canonicalize_inverse_trig_ratio_small_polynomial_partner_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if factor_short_geometric_sum_partner_root(ctx, expr).is_some() {
        return Some(expr);
    }
    if let Some(base) = extract_direct_short_geometric_product_base_root(ctx, expr) {
        return Some(build_direct_short_geometric_sum_expanded_target_root(
            ctx, base,
        ));
    }
    if let Some((base, constants)) = extract_direct_two_linear_shift_product_root(ctx, expr) {
        return build_direct_two_linear_shift_expanded_target_root(ctx, base, &constants);
    }
    if let Some(factored) = factor_known_small_polynomial_partner_root(ctx, expr) {
        if let Some((base, constants)) = extract_direct_two_linear_shift_product_root(ctx, factored)
        {
            return build_direct_two_linear_shift_expanded_target_root(ctx, base, &constants);
        }
    }

    None
}

pub(super) fn try_standard_inverse_trig_anchor_small_polynomial_partner_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let factors = flatten_mul_chain(ctx, expr);
    if !(2..=4).contains(&factors.len()) {
        return None;
    }

    for anchor_index in 0..factors.len() {
        let anchor_expr = factors[anchor_index];
        let Some(anchor_base) = extract_inverse_trig_ratio_anchor_base_root(ctx, anchor_expr)
        else {
            continue;
        };
        let partner_factors: Vec<_> = factors
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, factor)| (index != anchor_index).then_some(factor))
            .collect();
        let partner_expr = build_mul_expr_from_factors_root(ctx, &partner_factors);
        let Some(partner_canonical) =
            canonicalize_inverse_trig_ratio_small_polynomial_partner_root(ctx, partner_expr)
        else {
            continue;
        };
        let rewritten_raw =
            build_inverse_trig_ratio_anchor_product_root(ctx, anchor_base, partner_canonical);
        let rewritten =
            isolated_simplify_expr_if_changed(options, ctx, rewritten_raw).unwrap_or(rewritten_raw);
        if compare_expr(ctx, rewritten, expr) == Ordering::Equal
            && compare_expr(ctx, partner_canonical, partner_expr) == Ordering::Equal
        {
            continue;
        }

        let shortcut_steps = if collect_steps {
            vec![build_root_shortcut_compact_step(
                expr,
                rewritten,
                "Canonizar producto con ancla trig-inversa cociente y partner polinómico pequeño",
                "Inverse Trig Ratio Anchor Small Polynomial Partner",
            )]
        } else {
            Vec::new()
        };
        return Some((rewritten, shortcut_steps));
    }

    None
}

pub(super) fn is_potential_small_trig_zero_identity_root(ctx: &Context, expr: ExprId) -> bool {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        || !expr_contains_trig_or_hyperbolic_builtin_local(ctx, expr)
    {
        return false;
    }

    let terms = AddView::from_expr(ctx, expr).terms;
    if !(2..=4).contains(&terms.len()) {
        return false;
    }

    let has_positive = terms.iter().any(|(_, sign)| *sign == Sign::Pos);
    let has_negative = terms.iter().any(|(_, sign)| *sign == Sign::Neg);
    has_positive && has_negative
}

pub(super) fn is_supported_small_trig_zero_pair_side_root(
    ctx: &Context,
    expr: ExprId,
    allow_expanded_runtime_shape: bool,
) -> bool {
    if is_potential_small_trig_zero_identity_root(ctx, expr) {
        return true;
    }

    if !allow_expanded_runtime_shape {
        return false;
    }

    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _))
        || !expr_contains_trig_builtin_local(ctx, expr)
        || expr_contains_log_builtin_local(ctx, expr)
    {
        return false;
    }

    let terms = AddView::from_expr(ctx, expr).terms;
    let supported_term_count = if expr_contains_hyperbolic_builtin_local(ctx, expr) {
        (5..=8).contains(&terms.len())
    } else {
        (5..=6).contains(&terms.len())
    };
    if !supported_term_count {
        return false;
    }

    let has_positive = terms.iter().any(|(_, sign)| *sign == Sign::Pos);
    let has_negative = terms.iter().any(|(_, sign)| *sign == Sign::Neg);
    has_positive && has_negative
}

pub(super) fn try_standard_positive_double_cos_square_diff_factor_shortcut(
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
        let Some(arg) = extract_direct_positive_double_cos_square_diff_target_root(ctx, factor)
        else {
            continue;
        };

        let mut rewritten_factors = factors.clone();
        rewritten_factors[index] = build_positive_cos_double_angle_expr_root(ctx, arg);
        let rewritten = build_nonexpanding_locally_simplified_mul_expr_from_factors_root(
            ctx,
            &rewritten_factors,
        );
        let rewrite =
            crate::rule::Rewrite::with_local(rewritten, "Double Angle Expansion", expr, rewritten);
        return Some(finish_standard_root_shortcut(
            ctx,
            expr,
            rewrite,
            "Double Angle Expansion",
            collect_steps,
        ));
    }

    None
}

pub(super) fn try_standard_trig_power_reduction_factor_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let profiling =
        crate::orchestrator_shortcut_profiler::orchestrator_shortcut_profiling_enabled();
    let factors = flatten_mul_chain(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (index, factor) in factors.iter().copied().enumerate() {
        let mixed_square_arg = if profiling {
            run_profiled_root_shortcut("root.mul.23a.trig_power.factor.mixed_square", || {
                extract_direct_trig_power_mixed_square_target_root(ctx, factor)
            })
        } else {
            extract_direct_trig_power_mixed_square_target_root(ctx, factor)
        };
        let replacement = if let Some(arg) = mixed_square_arg {
            Some(build_scaled_double_angle_sin_square_root(ctx, arg))
        } else if let Some(arg) = if profiling {
            run_profiled_root_shortcut("root.mul.23b.trig_power.factor.cos_fourth", || {
                extract_direct_cos_fourth_power_reduction_target_root(ctx, factor)
            })
        } else {
            extract_direct_cos_fourth_power_reduction_target_root(ctx, factor)
        } {
            Some(build_plain_trig_pow4_root(ctx, BuiltinFn::Cos, arg))
        } else {
            let sin_cos_square_product_arg = if profiling {
                run_profiled_root_shortcut(
                    "root.mul.23c.trig_power.factor.sin_cos_square_product",
                    || extract_direct_sin_cos_square_product_reduction_target_root(ctx, factor),
                )
            } else {
                extract_direct_sin_cos_square_product_reduction_target_root(ctx, factor)
            };
            sin_cos_square_product_arg.map(|arg| build_plain_sin_cos_square_product_root(ctx, arg))
        };

        let Some(replacement) = replacement else {
            continue;
        };

        let mut rewritten_factors = factors.clone();
        rewritten_factors[index] = replacement;
        let rewritten = build_mul_expr_from_factors_root(ctx, &rewritten_factors);

        return Some(run_named_rebuilt_root_shortcut_simplify(
            options,
            ctx,
            expr,
            rewritten,
            "Power Reduction Identity",
            "Power Reduction Identity",
            collect_steps,
        ));
    }

    for first_index in 0..factors.len() {
        for second_index in (first_index + 1)..factors.len() {
            if !matches!(ctx.get(factors[first_index]), Expr::Pow(_, _))
                || !matches!(ctx.get(factors[second_index]), Expr::Pow(_, _))
            {
                continue;
            }
            let pair_factor = build_mul_expr_from_factors_root(
                ctx,
                &[factors[first_index], factors[second_index]],
            );
            if profiling {
                crate::orchestrator_shortcut_profiler::record_orchestrator_shortcut_sample(
                    "root.mul.23d.trig_power.pair.mixed_square",
                    render_expr_for_orchestrator_profile(ctx, pair_factor),
                );
            }
            let pair_arg = if profiling {
                run_profiled_root_shortcut("root.mul.23d.trig_power.pair.mixed_square", || {
                    extract_direct_trig_power_mixed_square_target_root(ctx, pair_factor)
                })
            } else {
                extract_direct_trig_power_mixed_square_target_root(ctx, pair_factor)
            };
            let Some(arg) = pair_arg else {
                continue;
            };

            let replacement = build_scaled_double_angle_sin_square_root(ctx, arg);
            let rewritten_factors = factors
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(index, factor)| {
                    if index == first_index {
                        Some(replacement)
                    } else if index == second_index {
                        None
                    } else {
                        Some(factor)
                    }
                })
                .collect::<Vec<_>>();
            let rewritten = build_mul_expr_from_factors_root(ctx, &rewritten_factors);

            return Some(run_named_rebuilt_root_shortcut_simplify(
                options,
                ctx,
                expr,
                rewritten,
                "Power Reduction Identity",
                "Power Reduction Identity",
                collect_steps,
            ));
        }
    }

    None
}

pub(super) fn try_standard_trig_power_reduction_zero_shortcut(
    _options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !expr_contains_any_builtin_local(ctx, expr, &[BuiltinFn::Sin, BuiltinFn::Cos]) {
        return None;
    }

    let (lhs_core, rhs_core) =
        crate::rules::arithmetic::extract_two_term_core_difference(ctx, expr)?;
    let rewrite =
        crate::rules::arithmetic::try_build_direct_trig_power_reduction_equivalence_rewrite(
            ctx, lhs_core, rhs_core,
        )?;
    Some(finish_root_shortcut_with_rewrite_meta(
        ctx,
        expr,
        rewrite,
        "Power Reduction Identity",
        collect_steps,
    ))
}

pub(super) fn try_standard_inverse_trig_composition_subset_zero_shortcut(
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
    if !expr_contains_any_builtin_local(
        ctx,
        expr,
        &[
            BuiltinFn::Asin,
            BuiltinFn::Arcsin,
            BuiltinFn::Atan,
            BuiltinFn::Arctan,
        ],
    ) {
        return None;
    }

    for first_index in 0..view.terms.len().saturating_sub(1) {
        for second_index in (first_index + 1)..view.terms.len() {
            let subset_terms = [view.terms[first_index], view.terms[second_index]];
            let subset_expr = build_signed_sum_expr_root(ctx, &subset_terms);
            if !expr_contains_any_builtin_local(
                ctx,
                subset_expr,
                &[
                    BuiltinFn::Asin,
                    BuiltinFn::Arcsin,
                    BuiltinFn::Atan,
                    BuiltinFn::Arctan,
                ],
            ) || !matches_direct_small_zero_or_known_pair_base_root(ctx, subset_expr)
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
            if !(3..=6).contains(&remaining_terms.len()) {
                continue;
            }

            let remaining_expr = AddView {
                root: expr,
                terms: remaining_terms,
            }
            .rebuild(ctx);
            if !is_supported_nested_zero_child_partner(ctx, remaining_expr) {
                continue;
            }
            if !narrow_known_pair_subset_remaining_rewrites_to_zero(options, ctx, remaining_expr) {
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

    None
}

pub(super) fn try_standard_inverse_trig_plus_sqrt_subset_zero_shortcut(
    options: &crate::phase::SimplifyOptions,
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let view = AddView::from_expr(ctx, expr);
    if !(9..=10).contains(&view.terms.len()) {
        return None;
    }
    if !expr_contains_sqrt_or_half_power_local(ctx, expr)
        || !expr_contains_log_builtin_local(ctx, expr)
    {
        return None;
    }
    if !expr_contains_any_builtin_local(
        ctx,
        expr,
        &[
            BuiltinFn::Asin,
            BuiltinFn::Arcsin,
            BuiltinFn::Atan,
            BuiltinFn::Arctan,
        ],
    ) {
        return None;
    }

    for first_index in 0..view.terms.len().saturating_sub(1) {
        for second_index in (first_index + 1)..view.terms.len() {
            let subset_terms = [view.terms[first_index], view.terms[second_index]];
            let subset_expr = build_signed_sum_expr_root(ctx, &subset_terms);
            if !expr_contains_any_builtin_local(
                ctx,
                subset_expr,
                &[
                    BuiltinFn::Asin,
                    BuiltinFn::Arcsin,
                    BuiltinFn::Atan,
                    BuiltinFn::Arctan,
                ],
            ) || !matches_direct_small_zero_or_known_pair_base_root(ctx, subset_expr)
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
            if !(7..=8).contains(&remaining_terms.len()) {
                continue;
            }

            let remaining_expr = AddView {
                root: expr,
                terms: remaining_terms,
            }
            .rebuild(ctx);
            if try_standard_sqrt_perfect_square_abs_subset_zero_shortcut(
                options,
                ctx,
                remaining_expr,
                false,
            )
            .is_none()
            {
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

    None
}

pub(super) fn try_standard_sin_sum_triple_identity_zero_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    try_rewrite_sin_sum_triple_identity_zero_expr(ctx, expr)?;
    let zero = ctx.num(0);
    Some(finish_standard_root_shortcut(
        ctx,
        expr,
        crate::rule::Rewrite::new(zero).desc("sin(t) + sin(3t) = 2·sin(2t)·cos(t)"),
        "Sin Sum Triple Identity Zero",
        collect_steps,
    ))
}

pub(super) fn extract_standard_trig_binomial_square_data(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, bool)> {
    let Expr::Pow(base, exponent) = ctx.get(expr) else {
        return None;
    };
    if extract_i64_integer(ctx, *exponent)? != 2 {
        return None;
    }

    let (left, right, is_sum) = match ctx.get(*base) {
        Expr::Add(left, right) => (*left, *right, true),
        Expr::Sub(left, right) => (*left, *right, false),
        _ => return None,
    };

    let extract_trig_arg = |term: ExprId| -> Option<(bool, ExprId)> {
        let Expr::Function(fn_id, args) = ctx.get(term) else {
            return None;
        };
        let [arg] = args.as_slice() else {
            return None;
        };
        if ctx.is_builtin(*fn_id, BuiltinFn::Sin) {
            return Some((true, *arg));
        }
        if ctx.is_builtin(*fn_id, BuiltinFn::Cos) {
            return Some((false, *arg));
        }
        None
    };

    let (lhs_kind, lhs_arg) = extract_trig_arg(left)?;
    let (rhs_kind, rhs_arg) = extract_trig_arg(right)?;
    if lhs_kind == rhs_kind || compare_expr(ctx, lhs_arg, rhs_arg) != Ordering::Equal {
        return None;
    }

    Some((lhs_arg, is_sum))
}

pub(super) fn try_standard_trig_fourth_power_difference_shortcut(
    ctx: &mut Context,
    expr: ExprId,
    collect_steps: bool,
) -> Option<(ExprId, Vec<Step>)> {
    let normalized = AddView::from_expr(ctx, expr).rebuild(ctx);
    let rewrite = try_rewrite_trig_fourth_power_difference_add_expr(ctx, normalized)?;
    let mut current = rewrite.rewritten;
    let mut shortcut_steps = Vec::new();

    if collect_steps {
        shortcut_steps.push(build_root_shortcut_compact_step(
            expr,
            current,
            "sin⁴(x) - cos⁴(x) = sin²(x) - cos²(x)",
            "Trig Fourth Power Difference",
        ));
    }

    if let Some(rewrite) = try_rewrite_pythagorean_generic_coefficient_add_expr(ctx, current) {
        let before = current;
        current = rewrite.rewritten;
        if collect_steps {
            shortcut_steps.push(build_root_shortcut_compact_step(
                before,
                current,
                "A·sin²(x) + A·cos²(x) = A",
                "Pythagorean with Generic Coefficient",
            ));
        }
    }

    if let Some((result, mut extra_steps)) =
        try_standard_pythagorean_additive_shortcut(ctx, current, collect_steps)
    {
        current = result;
        if collect_steps {
            shortcut_steps.append(&mut extra_steps);
        }
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

pub(super) fn extract_signed_numeric_trig_pow2(
    ctx: &Context,
    term: ExprId,
    outer_sign: Sign,
) -> Option<(BigRational, &'static str, ExprId, Sign)> {
    let (mut coeff, name, arg) = extract_coeff_trig_pow2(ctx, term)?;
    let mut effective_sign = outer_sign;
    if coeff.is_negative() {
        coeff = -coeff;
        effective_sign = effective_sign.negate();
    }
    Some((coeff, name, arg, effective_sign))
}

pub(super) fn is_mixed_sign_trig_square_difference_root(ctx: &Context, expr: ExprId) -> bool {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return false;
    }

    let (lhs_term, lhs_sign) = view.terms[0];
    let (rhs_term, rhs_sign) = view.terms[1];
    let Some((lhs_coeff, lhs_name, lhs_arg, lhs_effective_sign)) =
        extract_signed_numeric_trig_pow2(ctx, lhs_term, lhs_sign)
    else {
        return false;
    };
    let Some((rhs_coeff, rhs_name, rhs_arg, rhs_effective_sign)) =
        extract_signed_numeric_trig_pow2(ctx, rhs_term, rhs_sign)
    else {
        return false;
    };

    lhs_name != rhs_name
        && lhs_effective_sign != rhs_effective_sign
        && lhs_coeff == rhs_coeff
        && compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal
}

fn extract_mixed_sign_trig_square_difference_arg_root(
    ctx: &Context,
    expr: ExprId,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let (lhs_term, lhs_sign) = view.terms[0];
    let (rhs_term, rhs_sign) = view.terms[1];
    let (lhs_coeff, lhs_name, lhs_arg, lhs_effective_sign) =
        extract_signed_numeric_trig_pow2(ctx, lhs_term, lhs_sign)?;
    let (rhs_coeff, rhs_name, rhs_arg, rhs_effective_sign) =
        extract_signed_numeric_trig_pow2(ctx, rhs_term, rhs_sign)?;

    (lhs_name != rhs_name
        && lhs_effective_sign != rhs_effective_sign
        && lhs_coeff == rhs_coeff
        && compare_expr(ctx, lhs_arg, rhs_arg) == Ordering::Equal)
        .then_some(lhs_arg)
}

pub(super) fn is_small_positive_additive_trig_passthrough_core_root(
    ctx: &mut Context,
    expr: ExprId,
) -> bool {
    if !matches!(ctx.get(expr), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return false;
    }

    let terms = AddView::from_expr(ctx, expr).terms;
    (2..=4).contains(&terms.len())
        && terms.iter().all(|(_, sign)| *sign == Sign::Pos)
        && expr_contains_trig_builtin_local(ctx, expr)
        && !matches_direct_small_zero_identity_root(ctx, expr)
        && !is_potential_small_trig_zero_identity_root(ctx, expr)
}

pub(super) fn extract_direct_tangent_addition_target_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, ExprId)> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() != 2 {
        return None;
    }

    let mut first_arg = None;
    let mut second_arg = None;
    for (term_expr, term_sign) in view.terms {
        if term_sign != Sign::Pos {
            return None;
        }
        let arg = extract_unary_builtin_arg_root(ctx, term_expr, BuiltinFn::Tan)?;
        if first_arg.is_none() {
            first_arg = Some(arg);
        } else if second_arg.is_none() {
            second_arg = Some(arg);
        } else {
            return None;
        }
    }

    Some((first_arg?, second_arg?))
}
