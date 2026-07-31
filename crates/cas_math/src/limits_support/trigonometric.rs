//! `limits_support`: familia `trigonometric`.
//!
//! Ver la cabecera de `limits_support.rs` para el contexto.

use super::*;

fn scaled_sine_argument(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 && ctx.is_builtin(fn_id, BuiltinFn::Sin) => {
            Some((BigRational::one(), args[0]))
        }
        Expr::Neg(inner) => {
            let (scale, argument) = scaled_sine_argument(ctx, inner)?;
            Some((-scale, argument))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = constant_rational_value(ctx, lhs) {
                let (inner_scale, argument) = scaled_sine_argument(ctx, rhs)?;
                return Some((scale * inner_scale, argument));
            }
            if let Some(scale) = constant_rational_value(ctx, rhs) {
                let (inner_scale, argument) = scaled_sine_argument(ctx, lhs)?;
                return Some((scale * inner_scale, argument));
            }
            None
        }
        _ => None,
    }
}

fn scaled_trig_zero_argument(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, BigRational, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && (ctx.is_builtin(fn_id, BuiltinFn::Sin)
                    || ctx.is_builtin(fn_id, BuiltinFn::Cos)) =>
        {
            let builtin = if ctx.is_builtin(fn_id, BuiltinFn::Sin) {
                BuiltinFn::Sin
            } else {
                BuiltinFn::Cos
            };
            Some((builtin, BigRational::one(), args[0]))
        }
        Expr::Neg(inner) => {
            let (builtin, scale, argument) = scaled_trig_zero_argument(ctx, inner)?;
            Some((builtin, -scale, argument))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = constant_rational_value(ctx, lhs) {
                let (builtin, inner_scale, argument) = scaled_trig_zero_argument(ctx, rhs)?;
                return Some((builtin, scale * inner_scale, argument));
            }
            if let Some(scale) = constant_rational_value(ctx, rhs) {
                let (builtin, inner_scale, argument) = scaled_trig_zero_argument(ctx, lhs)?;
                return Some((builtin, scale * inner_scale, argument));
            }
            None
        }
        _ => None,
    }
}

fn scaled_trig_zero_power_argument(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, BigRational, ExprId, usize)> {
    if let Some((builtin, scale, argument)) = scaled_trig_zero_argument(ctx, expr) {
        return Some((builtin, scale, argument, 1));
    }

    let (base, exponent) = parse_pow_int(ctx, expr)?;
    if exponent <= 0 {
        return None;
    }
    let exponent = usize::try_from(exponent).ok()?;
    let (builtin, scale, argument) = scaled_trig_zero_argument(ctx, base)?;
    if scale.is_zero() {
        return None;
    }

    let mut power_scale = BigRational::one();
    for _ in 0..exponent {
        power_scale *= &scale;
    }
    Some((builtin, power_scale, argument, exponent))
}

fn scaled_reciprocal_trig_zero_argument(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BuiltinFn, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            if ctx.is_builtin(fn_id, BuiltinFn::Csc) {
                Some((BigRational::one(), BuiltinFn::Sin, args[0]))
            } else if ctx.is_builtin(fn_id, BuiltinFn::Sec) {
                Some((BigRational::one(), BuiltinFn::Cos, args[0]))
            } else {
                None
            }
        }
        Expr::Neg(inner) => {
            let (scale, builtin, argument) = scaled_reciprocal_trig_zero_argument(ctx, inner)?;
            Some((-scale, builtin, argument))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = constant_rational_value(ctx, lhs) {
                let (inner_scale, builtin, argument) =
                    scaled_reciprocal_trig_zero_argument(ctx, rhs)?;
                return Some((scale * inner_scale, builtin, argument));
            }
            if let Some(scale) = constant_rational_value(ctx, rhs) {
                let (inner_scale, builtin, argument) =
                    scaled_reciprocal_trig_zero_argument(ctx, lhs)?;
                return Some((scale * inner_scale, builtin, argument));
            }
            None
        }
        _ => None,
    }
}

fn scaled_reciprocal_trig_zero_power_argument(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BuiltinFn, ExprId, usize)> {
    if let Some((scale, builtin, argument)) = scaled_reciprocal_trig_zero_argument(ctx, expr) {
        if scale.is_zero() {
            return None;
        }
        return Some((scale, builtin, argument, 1));
    }

    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (scale, builtin, argument, exponent) =
            scaled_reciprocal_trig_zero_power_argument(ctx, inner)?;
        return Some((-scale, builtin, argument, exponent));
    }
    if let Expr::Mul(lhs, rhs) = ctx.get(expr).clone() {
        if let Some(scale) = constant_rational_value(ctx, lhs) {
            let (inner_scale, builtin, argument, exponent) =
                scaled_reciprocal_trig_zero_power_argument(ctx, rhs)?;
            let scale = scale * inner_scale;
            if scale.is_zero() {
                return None;
            }
            return Some((scale, builtin, argument, exponent));
        }
        if let Some(scale) = constant_rational_value(ctx, rhs) {
            let (inner_scale, builtin, argument, exponent) =
                scaled_reciprocal_trig_zero_power_argument(ctx, lhs)?;
            let scale = scale * inner_scale;
            if scale.is_zero() {
                return None;
            }
            return Some((scale, builtin, argument, exponent));
        }
    }

    let (base, exponent) = parse_pow_int(ctx, expr)?;
    if exponent <= 0 {
        return None;
    }
    let exponent = usize::try_from(exponent).ok()?;
    let (scale, builtin, argument) = scaled_reciprocal_trig_zero_argument(ctx, base)?;
    if scale.is_zero() {
        return None;
    }

    let mut power_scale = BigRational::one();
    for _ in 0..exponent {
        power_scale *= &scale;
    }
    Some((power_scale, builtin, argument, exponent))
}

fn scaled_tan_cot_argument(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BuiltinFn, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => {
            if ctx.is_builtin(fn_id, BuiltinFn::Tan) {
                Some((BigRational::one(), BuiltinFn::Tan, args[0]))
            } else if ctx.is_builtin(fn_id, BuiltinFn::Cot) {
                Some((BigRational::one(), BuiltinFn::Cot, args[0]))
            } else {
                None
            }
        }
        Expr::Neg(inner) => {
            let (scale, builtin, argument) = scaled_tan_cot_argument(ctx, inner)?;
            Some((-scale, builtin, argument))
        }
        Expr::Mul(lhs, rhs) => {
            if let Some(scale) = constant_rational_value(ctx, lhs) {
                let (inner_scale, builtin, argument) = scaled_tan_cot_argument(ctx, rhs)?;
                return Some((scale * inner_scale, builtin, argument));
            }
            if let Some(scale) = constant_rational_value(ctx, rhs) {
                let (inner_scale, builtin, argument) = scaled_tan_cot_argument(ctx, lhs)?;
                return Some((scale * inner_scale, builtin, argument));
            }
            None
        }
        _ => None,
    }
}

fn scaled_trig_ratio_power_argument(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BuiltinFn, ExprId, BuiltinFn, ExprId, usize)> {
    if let Some((
        scale,
        numerator_builtin,
        numerator_argument,
        denominator_builtin,
        denominator_argument,
    )) = trig_ratio_source_components(ctx, expr)
    {
        if scale.is_zero() {
            return None;
        }
        return Some((
            scale,
            numerator_builtin,
            numerator_argument,
            denominator_builtin,
            denominator_argument,
            1,
        ));
    }

    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (
            scale,
            numerator_builtin,
            numerator_argument,
            denominator_builtin,
            denominator_argument,
            exponent,
        ) = scaled_trig_ratio_power_argument(ctx, inner)?;
        return Some((
            -scale,
            numerator_builtin,
            numerator_argument,
            denominator_builtin,
            denominator_argument,
            exponent,
        ));
    }
    if let Expr::Mul(lhs, rhs) = ctx.get(expr).clone() {
        if let Some(scale) = constant_rational_value(ctx, lhs) {
            let (
                inner_scale,
                numerator_builtin,
                numerator_argument,
                denominator_builtin,
                denominator_argument,
                exponent,
            ) = scaled_trig_ratio_power_argument(ctx, rhs)?;
            let scale = scale * inner_scale;
            if scale.is_zero() {
                return None;
            }
            return Some((
                scale,
                numerator_builtin,
                numerator_argument,
                denominator_builtin,
                denominator_argument,
                exponent,
            ));
        }
        if let Some(scale) = constant_rational_value(ctx, rhs) {
            let (
                inner_scale,
                numerator_builtin,
                numerator_argument,
                denominator_builtin,
                denominator_argument,
                exponent,
            ) = scaled_trig_ratio_power_argument(ctx, lhs)?;
            let scale = scale * inner_scale;
            if scale.is_zero() {
                return None;
            }
            return Some((
                scale,
                numerator_builtin,
                numerator_argument,
                denominator_builtin,
                denominator_argument,
                exponent,
            ));
        }
    }

    let (base, exponent) = parse_pow_int(ctx, expr)?;
    if exponent <= 0 {
        return None;
    }
    let exponent = usize::try_from(exponent).ok()?;
    let (scale, numerator_builtin, numerator_argument, denominator_builtin, denominator_argument) =
        trig_ratio_source_components(ctx, base)?;
    if scale.is_zero() {
        return None;
    }

    let mut power_scale = BigRational::one();
    for _ in 0..exponent {
        power_scale *= &scale;
    }
    Some((
        power_scale,
        numerator_builtin,
        numerator_argument,
        denominator_builtin,
        denominator_argument,
        exponent,
    ))
}

fn finite_trig_power_pole_components(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BuiltinFn, BigRational, ExprId, usize)> {
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let numerator = constant_rational_value(ctx, num)?;
        if numerator.is_zero() {
            return None;
        }

        let (builtin, den_scale, argument, exponent) = scaled_trig_zero_power_argument(ctx, den)?;
        return Some((numerator, builtin, den_scale, argument, exponent));
    }

    let (numerator, builtin, argument, exponent) =
        scaled_reciprocal_trig_zero_power_argument(ctx, expr)?;
    Some((numerator, builtin, BigRational::one(), argument, exponent))
}

fn tan_cot_ratio_builtins(source_builtin: BuiltinFn) -> Option<(BuiltinFn, BuiltinFn)> {
    match source_builtin {
        BuiltinFn::Tan => Some((BuiltinFn::Sin, BuiltinFn::Cos)),
        BuiltinFn::Cot => Some((BuiltinFn::Cos, BuiltinFn::Sin)),
        _ => None,
    }
}

fn trig_ratio_source_components(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, BuiltinFn, ExprId, BuiltinFn, ExprId)> {
    if let Some((scale, source_builtin, argument)) = scaled_tan_cot_argument(ctx, expr) {
        let (numerator_builtin, denominator_builtin) = tan_cot_ratio_builtins(source_builtin)?;
        return Some((
            scale,
            numerator_builtin,
            argument,
            denominator_builtin,
            argument,
        ));
    }

    let Expr::Div(numerator, denominator) = ctx.get(expr).clone() else {
        return None;
    };
    let (numerator_builtin, numerator_scale, numerator_argument) =
        scaled_trig_zero_argument(ctx, numerator)?;
    let (denominator_builtin, denominator_scale, denominator_argument) =
        scaled_trig_zero_argument(ctx, denominator)?;
    if denominator_scale.is_zero() {
        return None;
    }

    Some((
        numerator_scale / denominator_scale,
        numerator_builtin,
        numerator_argument,
        denominator_builtin,
        denominator_argument,
    ))
}

fn finite_trig_zero_tail_sign(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument: ExprId,
    var: ExprId,
    point: ExprId,
    point_value: Option<&BigRational>,
    side: FiniteLimitSide,
) -> Option<InfSign> {
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();

    match builtin {
        BuiltinFn::Sin => {
            let Some(point_value) = point_value else {
                return finite_direct_variable_special_trig_zero_tail_sign(
                    ctx, builtin, argument, var, point, side,
                );
            };
            let argument = match Polynomial::from_expr(ctx, argument, &var_name) {
                Ok(argument) => argument,
                Err(_) => {
                    let local = FiniteTrigZeroTailLocal {
                        var,
                        point,
                        point_value,
                        side,
                        var_name: &var_name,
                    };
                    return finite_table_trig_zero_tail_sign(ctx, builtin, argument, &local);
                }
            };
            let (argument_order, argument_derivative) =
                finite_polynomial_local_order_and_derivative(&argument, point_value)?;
            if argument_order == 0 {
                return None;
            }
            finite_local_tail_sign(&argument_derivative, argument_order, side)
        }
        BuiltinFn::Cos => {
            if point_value.is_none() {
                if let Some(tail_sign) = finite_direct_variable_special_trig_zero_tail_sign(
                    ctx, builtin, argument, var, point, side,
                ) {
                    return Some(tail_sign);
                }
            }

            let point_value = point_value?;
            let local = FiniteTrigZeroTailLocal {
                var,
                point,
                point_value,
                side,
                var_name: &var_name,
            };
            finite_table_trig_zero_tail_sign(ctx, builtin, argument, &local)
        }
        _ => None,
    }
}

fn finite_table_trig_zero_tail_sign(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument: ExprId,
    local: &FiniteTrigZeroTailLocal<'_>,
) -> Option<InfSign> {
    let argument_limit = try_limit_rules_at_finite(ctx, argument, local.var, local.point)?;
    let value_at_limit = finite_total_real_unary_trig_table_result(ctx, builtin, argument_limit)?;
    if !constant_rational_value(ctx, value_at_limit)?.is_zero() {
        return None;
    }

    let derivative_factor = match builtin {
        BuiltinFn::Sin => {
            let cos_at_limit =
                finite_total_real_unary_trig_table_result(ctx, BuiltinFn::Cos, argument_limit)?;
            constant_rational_value(ctx, cos_at_limit)?
        }
        BuiltinFn::Cos => {
            let sin_at_limit =
                finite_total_real_unary_trig_table_result(ctx, BuiltinFn::Sin, argument_limit)?;
            -constant_rational_value(ctx, sin_at_limit)?
        }
        _ => return None,
    };
    if derivative_factor.is_zero() {
        return None;
    }

    let tail_expr = finite_argument_tail_after_limit(ctx, argument, argument_limit);
    let tail = Polynomial::from_expr(ctx, tail_expr, local.var_name).ok()?;
    let (tail_order, tail_derivative) =
        finite_polynomial_local_order_and_derivative(&tail, local.point_value)?;
    if tail_order == 0 {
        return None;
    }

    let tail_sign = finite_local_tail_sign(&tail_derivative, tail_order, local.side)?;
    let derivative_sign = if derivative_factor.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    Some(if derivative_sign == tail_sign {
        InfSign::Pos
    } else {
        InfSign::Neg
    })
}

fn finite_direct_variable_special_trig_zero_tail_sign(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<InfSign> {
    if !structurally_equal_expr(ctx, argument, var) {
        return None;
    }

    let argument_limit = try_limit_rules_at_finite(ctx, argument, var, point)?;
    let derivative_sign =
        finite_direct_variable_trig_zero_derivative_sign(ctx, builtin, argument_limit)?;
    let point_tail_sign = match side {
        FiniteLimitSide::Left => InfSign::Neg,
        FiniteLimitSide::Right => InfSign::Pos,
    };
    Some(if derivative_sign == point_tail_sign {
        InfSign::Pos
    } else {
        InfSign::Neg
    })
}

fn finite_direct_variable_trig_zero_derivative_sign(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> Option<InfSign> {
    if let Some(sign) =
        finite_direct_variable_rational_pi_trig_zero_derivative_sign(ctx, builtin, argument_limit)
    {
        return Some(sign);
    }

    let value_at_limit = finite_total_real_unary_trig_table_result(ctx, builtin, argument_limit)?;
    if !constant_rational_value(ctx, value_at_limit)?.is_zero() {
        return None;
    }

    let derivative_value = match builtin {
        BuiltinFn::Sin => {
            let cos_at_limit =
                finite_total_real_unary_trig_table_result(ctx, BuiltinFn::Cos, argument_limit)?;
            constant_rational_value(ctx, cos_at_limit)?
        }
        BuiltinFn::Cos => {
            let sin_at_limit =
                finite_total_real_unary_trig_table_result(ctx, BuiltinFn::Sin, argument_limit)?;
            -constant_rational_value(ctx, sin_at_limit)?
        }
        _ => return None,
    };
    if derivative_value.is_zero() {
        return None;
    }

    Some(if derivative_value.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    })
}

fn finite_direct_variable_rational_pi_trig_zero_derivative_sign(
    ctx: &Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> Option<InfSign> {
    let k = extract_rational_pi_multiple(ctx, argument_limit)?;
    match builtin {
        BuiltinFn::Sin if k.is_integer() => integer_parity_cos_sign(&k),
        BuiltinFn::Cos => {
            let sin_sign = half_integer_sin_sign(&k)?;
            Some(match sin_sign {
                InfSign::Pos => InfSign::Neg,
                InfSign::Neg => InfSign::Pos,
            })
        }
        _ => None,
    }
}

fn integer_parity_cos_sign(k: &BigRational) -> Option<InfSign> {
    if !k.is_integer() {
        return None;
    }
    let n = k.to_integer();
    let two = BigInt::from(2);
    if (&n % &two).is_zero() {
        Some(InfSign::Pos)
    } else {
        Some(InfSign::Neg)
    }
}

fn half_integer_sin_sign(k: &BigRational) -> Option<InfSign> {
    if k.denom() != &BigInt::from(2) {
        return None;
    }

    let four = BigInt::from(4);
    let mut rem = k.numer() % &four;
    if rem.is_negative() {
        rem += &four;
    }

    if rem == BigInt::from(1) {
        Some(InfSign::Pos)
    } else if rem == BigInt::from(3) {
        Some(InfSign::Neg)
    } else {
        None
    }
}

pub(super) fn apply_finite_one_sided_trig_power_pole_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let point_value = match ctx.get(point) {
        Expr::Number(point_value) => Some(point_value.clone()),
        _ => None,
    };

    let (numerator, builtin, den_scale, argument, exponent) =
        finite_trig_power_pole_components(ctx, expr)?;
    let argument_tail = finite_trig_zero_tail_sign(
        ctx,
        builtin,
        argument,
        var,
        point,
        point_value.as_ref(),
        side,
    )?;
    let scale_tail = if den_scale.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    let den_tail = if exponent.is_multiple_of(2) || scale_tail == argument_tail {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    let numerator_tail = if numerator.is_positive() {
        InfSign::Pos
    } else {
        InfSign::Neg
    };
    Some(signed_abs_ratio_infinity(ctx, numerator_tail, den_tail))
}

pub(super) fn apply_finite_bilateral_trig_power_pole_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let left =
        apply_finite_one_sided_trig_power_pole_rule(ctx, expr, var, point, FiniteLimitSide::Left)?;
    let right =
        apply_finite_one_sided_trig_power_pole_rule(ctx, expr, var, point, FiniteLimitSide::Right)?;
    matching_finite_bilateral_one_sided_result(ctx, left, right)
}

pub(super) fn apply_finite_one_sided_trig_ratio_power_pole_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let point_value = match ctx.get(point) {
        Expr::Number(point_value) => Some(point_value.clone()),
        _ => None,
    };

    let (
        scale,
        numerator_builtin,
        numerator_argument,
        denominator_builtin,
        denominator_argument,
        exponent,
    ) = scaled_trig_ratio_power_argument(ctx, expr)?;
    let numerator_argument_limit = try_limit_rules_at_finite(ctx, numerator_argument, var, point)?;
    let denominator_argument_limit =
        try_limit_rules_at_finite(ctx, denominator_argument, var, point)?;

    let denominator_at_limit = finite_total_real_unary_trig_table_result(
        ctx,
        denominator_builtin,
        denominator_argument_limit,
    )?;
    if !constant_rational_value(ctx, denominator_at_limit)?.is_zero() {
        return None;
    }

    let numerator_at_limit = finite_total_real_unary_trig_table_result(
        ctx,
        numerator_builtin,
        numerator_argument_limit,
    )?;
    let numerator_value = constant_rational_value(ctx, numerator_at_limit)?;
    if numerator_value.is_zero() {
        return None;
    }

    let denominator_tail = finite_trig_zero_tail_sign(
        ctx,
        denominator_builtin,
        denominator_argument,
        var,
        point,
        point_value.as_ref(),
        side,
    )?;
    let denominator_tail = if exponent.is_multiple_of(2) {
        InfSign::Pos
    } else {
        denominator_tail
    };

    let scale_tail = rational_tail_sign(&scale);
    let numerator_tail = if exponent.is_multiple_of(2) || numerator_value.is_positive() {
        scale_tail
    } else {
        multiply_tail_signs(scale_tail, InfSign::Neg)
    };

    Some(signed_abs_ratio_infinity(
        ctx,
        numerator_tail,
        denominator_tail,
    ))
}

pub(super) fn apply_finite_bilateral_trig_ratio_power_pole_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let left = apply_finite_one_sided_trig_ratio_power_pole_rule(
        ctx,
        expr,
        var,
        point,
        FiniteLimitSide::Left,
    )?;
    let right = apply_finite_one_sided_trig_ratio_power_pole_rule(
        ctx,
        expr,
        var,
        point,
        FiniteLimitSide::Right,
    )?;
    matching_finite_bilateral_one_sided_result(ctx, left, right)
}

fn finite_inverse_trig_endpoint_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    endpoint: InverseTrigEndpoint,
) -> Option<ExprId> {
    match (builtin, endpoint) {
        (BuiltinFn::Asin | BuiltinFn::Arcsin, InverseTrigEndpoint::Lower) => {
            let pi_over_two = TrigValue::PiDiv(2).to_expr(ctx);
            Some(ctx.add(Expr::Neg(pi_over_two)))
        }
        (BuiltinFn::Asin | BuiltinFn::Arcsin, InverseTrigEndpoint::Upper) => {
            Some(TrigValue::PiDiv(2).to_expr(ctx))
        }
        (BuiltinFn::Acos | BuiltinFn::Arccos, InverseTrigEndpoint::Lower) => {
            Some(ctx.add(Expr::Constant(Constant::Pi)))
        }
        (BuiltinFn::Acos | BuiltinFn::Arccos, InverseTrigEndpoint::Upper) => Some(ctx.num(0)),
        _ => None,
    }
}

pub(super) fn finite_inverse_trig_endpoint_gap(
    argument: &Polynomial,
    var_name: &str,
    point_value: &BigRational,
) -> Option<(Polynomial, InverseTrigEndpoint)> {
    let one = rational_one();
    let argument_value = argument.eval(point_value);
    if argument_value == one {
        Some((
            Polynomial::one(var_name.to_string()).sub(argument),
            InverseTrigEndpoint::Upper,
        ))
    } else if argument_value == -one {
        Some((
            argument.add(&Polynomial::one(var_name.to_string())),
            InverseTrigEndpoint::Lower,
        ))
    } else {
        None
    }
}

pub(super) fn apply_finite_inverse_trig_polynomial_endpoint_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(
        builtin,
        BuiltinFn::Asin | BuiltinFn::Arcsin | BuiltinFn::Acos | BuiltinFn::Arccos
    ) {
        return None;
    }

    let argument = Polynomial::from_expr(ctx, args[0], &var_name).ok()?;
    let (endpoint_gap, endpoint) =
        finite_inverse_trig_endpoint_gap(&argument, &var_name, &point_value)?;
    let (gap_order, gap_derivative) =
        finite_polynomial_local_order_and_derivative(&endpoint_gap, &point_value)?;
    if !finite_local_tail_positive_on_both_sides(&gap_derivative, gap_order)? {
        return None;
    }

    finite_inverse_trig_endpoint_result(ctx, builtin, endpoint)
}

pub(super) fn apply_finite_one_sided_inverse_trig_endpoint_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
    side: FiniteLimitSide,
) -> Option<ExprId> {
    if depends_on(ctx, point, var) {
        return None;
    }
    let Expr::Variable(var_symbol) = ctx.get(var) else {
        return None;
    };
    let var_name = ctx.sym_name(*var_symbol).to_string();
    let Expr::Number(point_value) = ctx.get(point) else {
        return None;
    };
    let point_value = point_value.clone();
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(
        builtin,
        BuiltinFn::Asin | BuiltinFn::Arcsin | BuiltinFn::Acos | BuiltinFn::Arccos
    ) {
        return None;
    }

    let argument = Polynomial::from_expr(ctx, args[0], &var_name).ok()?;
    let (endpoint_gap, endpoint) =
        finite_inverse_trig_endpoint_gap(&argument, &var_name, &point_value)?;
    let (gap_order, gap_derivative) =
        finite_polynomial_local_order_and_derivative(&endpoint_gap, &point_value)?;
    if finite_local_tail_sign(&gap_derivative, gap_order, side)? != InfSign::Pos {
        return None;
    }

    finite_inverse_trig_endpoint_result(ctx, builtin, endpoint)
}

pub(super) fn apply_finite_sine_zero_quotient_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    apply_finite_zero_quotient_rule(ctx, expr, var, point, scaled_sine_argument)
}

pub(super) fn is_finite_domain_checked_trig_unary_builtin(builtin: BuiltinFn) -> bool {
    matches!(
        builtin,
        BuiltinFn::Tan | BuiltinFn::Sec | BuiltinFn::Csc | BuiltinFn::Cot
    )
}

pub(super) fn finite_total_real_unary_trig_table_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument: ExprId,
) -> Option<ExprId> {
    if !matches!(
        builtin,
        BuiltinFn::Sin | BuiltinFn::Cos | BuiltinFn::Atan | BuiltinFn::Arctan
    ) {
        return None;
    }

    lookup_trig_or_inverse(ctx, builtin.name(), argument)
        .map(|hit| trig_table_value_to_limit_expr(ctx, hit.value))
}

pub(super) fn trig_table_value_to_limit_expr(ctx: &mut Context, value: &TrigValue) -> ExprId {
    match value {
        TrigValue::Fraction(numerator, 1) => ctx.num(*numerator),
        _ => value.to_expr(ctx),
    }
}

pub(super) fn finite_domain_checked_trig_unary_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument_limit: ExprId,
) -> Option<ExprId> {
    if let Some(argument_value) = numeric_limit_value(ctx, argument_limit) {
        if argument_value.is_zero() {
            return match builtin {
                BuiltinFn::Tan => Some(ctx.num(0)),
                BuiltinFn::Sec => Some(ctx.num(1)),
                _ => None,
            };
        }

        let argument_expr = ctx.add(Expr::Number(argument_value));
        return finite_domain_checked_trig_unary_table_result(ctx, builtin, argument_expr);
    }

    finite_domain_checked_trig_unary_table_result(ctx, builtin, argument_limit)
}

fn finite_domain_checked_trig_unary_table_result(
    ctx: &mut Context,
    builtin: BuiltinFn,
    argument: ExprId,
) -> Option<ExprId> {
    if !matches!(
        builtin,
        BuiltinFn::Tan | BuiltinFn::Sec | BuiltinFn::Csc | BuiltinFn::Cot
    ) {
        return None;
    }

    let hit = lookup_trig_or_inverse(ctx, builtin.name(), argument)?;
    if matches!(*hit.value, TrigValue::Undefined) {
        return None;
    }
    Some(trig_table_value_to_limit_expr(ctx, hit.value))
}

pub(super) fn apply_finite_domain_checked_trig_unary_composition_rule(
    ctx: &mut Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }
    let builtin = ctx.builtin_of(fn_id)?;
    if !is_finite_domain_checked_trig_unary_builtin(builtin) {
        return None;
    }

    let argument_limit = try_limit_rules_at_finite(ctx, args[0], var, point)?;
    finite_domain_checked_trig_unary_result(ctx, builtin, argument_limit)
}

pub(super) fn finite_residual_has_empty_punctured_inverse_trig_domain(
    ctx: &Context,
    expr: ExprId,
    var: ExprId,
    point: ExprId,
) -> bool {
    let Some(finite_point) = finite_residual_point(ctx, var, point) else {
        return false;
    };
    let Some((builtin, argument_expr)) = finite_single_function_arg(ctx, expr) else {
        return false;
    };
    if !matches!(
        builtin,
        BuiltinFn::Asin | BuiltinFn::Arcsin | BuiltinFn::Acos | BuiltinFn::Arccos
    ) {
        return false;
    }

    let Ok(argument) = Polynomial::from_expr(ctx, argument_expr, &finite_point.var_name) else {
        return false;
    };
    let Some((endpoint_gap, _endpoint)) = finite_inverse_trig_endpoint_gap(
        &argument,
        &finite_point.var_name,
        &finite_point.point_value,
    ) else {
        return false;
    };

    finite_polynomial_tail_negative_on_both_sides(&endpoint_gap, &finite_point.point_value)
        .unwrap_or(false)
}
