//! `symbolic_integration_support`: familia `hyperbolic`.
//!
//! Ver la cabecera de `symbolic_integration_support.rs` para el contexto.

use super::*;

pub(super) fn inverse_hyperbolic_scale_over_constant_root(
    scale_without_root: BigRational,
    constant: &BigRational,
) -> Option<(BigRational, Option<BigRational>)> {
    if let Some(root) = exact_rational_sqrt(constant) {
        return Some((scale_without_root / root, None));
    }

    constant.is_positive().then(|| {
        (
            scale_without_root / constant.clone(),
            Some(constant.clone()),
        )
    })
}

pub(super) fn normalize_asinh_reciprocal_display_base(
    base: Polynomial,
    constant: BigRational,
) -> (Polynomial, BigRational) {
    if exact_rational_sqrt(&constant).is_some() {
        return (base, constant);
    }

    let content = base.content();
    if content.is_zero() || !content.is_positive() || content.is_one() {
        return (base, constant);
    }

    let normalized_constant = constant.clone() / content.clone();
    if normalized_constant.is_positive() && exact_rational_sqrt(&normalized_constant).is_some() {
        (base.div_scalar(&content), normalized_constant)
    } else {
        (base, constant)
    }
}

pub(super) fn normalize_atanh_reciprocal_display_base(
    base: Polynomial,
    constant: BigRational,
) -> (Polynomial, BigRational) {
    let content = base.content();
    if content.is_zero() || !content.is_positive() || content.is_one() {
        return (base, constant);
    }

    let normalized_constant = constant.clone() / content.clone();
    if normalized_constant.is_positive() {
        (base.div_scalar(&content), normalized_constant)
    } else {
        (base, constant)
    }
}

pub(super) fn atanh_gap_constant_and_alignment(
    base: &Polynomial,
    gap_factor: &Polynomial,
) -> Option<(BigRational, BigRational)> {
    let base_slope = base.coeffs.get(1).cloned()?;
    let gap_slope = gap_factor.coeffs.get(1).cloned()?;
    if base_slope.is_zero() || gap_slope.is_zero() {
        return None;
    }
    let gap_alignment = base_slope / gap_slope;
    if gap_alignment.is_zero() {
        return None;
    }
    let alignment_reciprocal = BigRational::one() / gap_alignment.clone();
    let gap = gap_factor.div_scalar(&alignment_reciprocal);
    let constant = positive_constant_difference(base, &gap)?;
    Some((constant, gap_alignment))
}

pub(super) fn inverse_hyperbolic_sqrt_reciprocal_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    kind: InverseHyperbolicSqrtReciprocalKind,
) -> Option<ExprId> {
    let parts = inverse_hyperbolic_sqrt_reciprocal_parts(ctx, expr, var, kind)?;
    let constant = ctx.add(Expr::Number(parts.constant));
    let base = parts.base.to_expr(ctx);
    let quotient = ctx.add(Expr::Div(constant, base));
    let sqrt_arg = ctx.call_builtin(BuiltinFn::Sqrt, vec![quotient]);
    let builtin = match kind {
        InverseHyperbolicSqrtReciprocalKind::Asinh => BuiltinFn::Asinh,
        InverseHyperbolicSqrtReciprocalKind::Atanh => BuiltinFn::Atanh,
    };
    let antiderivative = ctx.call_builtin(builtin, vec![sqrt_arg]);
    if let Some(sqrt_factor) = parts.scale_sqrt_factor {
        scale_rational_sqrt_term(ctx, parts.scale, sqrt_factor, antiderivative)
    } else {
        Some(scale_rational_term(ctx, parts.scale, antiderivative))
    }
}

fn inverse_hyperbolic_sqrt_reciprocal_positive_condition(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    kind: InverseHyperbolicSqrtReciprocalKind,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            return inverse_hyperbolic_sqrt_reciprocal_positive_condition(ctx, inner, var, kind);
        }
        Expr::Mul(left, right) if !contains_named_var(ctx, left, var) => {
            return inverse_hyperbolic_sqrt_reciprocal_positive_condition(ctx, right, var, kind);
        }
        Expr::Mul(left, right) if !contains_named_var(ctx, right, var) => {
            return inverse_hyperbolic_sqrt_reciprocal_positive_condition(ctx, left, var, kind);
        }
        _ => {}
    }

    let parts = inverse_hyperbolic_sqrt_reciprocal_parts(ctx, expr, var, kind)?;
    let condition_poly = match kind {
        InverseHyperbolicSqrtReciprocalKind::Asinh => parts.base,
        InverseHyperbolicSqrtReciprocalKind::Atanh => parts
            .base
            .sub(&Polynomial::new(vec![parts.constant], var.to_string())),
    };
    Some(condition_poly.to_expr(ctx))
}

pub fn integrate_symbolic_is_inverse_hyperbolic_sqrt_reciprocal_target(
    ctx: &Context,
    expr: ExprId,
    var: &str,
) -> bool {
    inverse_hyperbolic_sqrt_reciprocal_parts(
        ctx,
        expr,
        var,
        InverseHyperbolicSqrtReciprocalKind::Asinh,
    )
    .is_some()
        || inverse_hyperbolic_sqrt_reciprocal_parts(
            ctx,
            expr,
            var,
            InverseHyperbolicSqrtReciprocalKind::Atanh,
        )
        .is_some()
}

pub(super) fn hyperbolic_square_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    if !is_number(ctx, exp, 2) {
        return None;
    }

    let (fn_id, args) = match ctx.get(base).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };

    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(builtin, BuiltinFn::Sinh | BuiltinFn::Cosh | BuiltinFn::Tanh) {
        return None;
    }

    let arg = args[0];
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    if matches!(builtin, BuiltinFn::Tanh) {
        let var_expr = ctx.var(var);
        let tanh_arg = ctx.call_builtin(BuiltinFn::Tanh, vec![arg]);
        let correction = scale_rational_term(ctx, -BigRational::one() / a, tanh_arg);
        return Some(ctx.add(Expr::Add(var_expr, correction)));
    }

    let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![arg]);
    let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let product = mul2_raw(ctx, sinh_arg, cosh_arg);
    let two_scale = BigRational::from_integer(2.into());
    let oscillatory = scale_rational_term(ctx, BigRational::one() / (two_scale * a), product);

    let var_expr = ctx.var(var);
    let two = ctx.num(2);
    let half_linear = ctx.add(Expr::Div(var_expr, two));

    match builtin {
        BuiltinFn::Sinh => Some(ctx.add(Expr::Sub(oscillatory, half_linear))),
        BuiltinFn::Cosh => Some(ctx.add(Expr::Add(oscillatory, half_linear))),
        _ => None,
    }
}

pub(super) fn hyperbolic_tanh_even_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    let power = if is_number(ctx, exp, 4) {
        4
    } else if is_number(ctx, exp, 6) {
        6
    } else if is_number(ctx, exp, 8) {
        8
    } else {
        return None;
    };

    let (fn_id, args) = match ctx.get(base).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Tanh) {
        return None;
    }

    let arg = args[0];
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    let tanh_arg = ctx.call_builtin(BuiltinFn::Tanh, vec![arg]);
    let three = ctx.num(3);
    let tanh_cubed = ctx.add(Expr::Pow(tanh_arg, three));
    let cubic_term = scale_rational_term(ctx, BigRational::new(1.into(), 3.into()), tanh_cubed);
    let mut tanh_terms = ctx.add(Expr::Add(tanh_arg, cubic_term));
    if power >= 6 {
        let five = ctx.num(5);
        let tanh_fifth = ctx.add(Expr::Pow(tanh_arg, five));
        let fifth_term = scale_rational_term(ctx, BigRational::new(1.into(), 5.into()), tanh_fifth);
        tanh_terms = ctx.add(Expr::Add(tanh_terms, fifth_term));
    }
    if power == 8 {
        let seven = ctx.num(7);
        let tanh_seventh = ctx.add(Expr::Pow(tanh_arg, seven));
        let seventh_term =
            scale_rational_term(ctx, BigRational::new(1.into(), 7.into()), tanh_seventh);
        tanh_terms = ctx.add(Expr::Add(tanh_terms, seventh_term));
    }
    let var_expr = ctx.var(var);
    let primitive = if a.is_negative() {
        let scaled_tanh_terms = if power == 8 {
            scale_reciprocal_integration_result_preserving_presentation(
                ctx,
                -BigRational::one() / a,
                tanh_terms,
            )
        } else {
            scale_rational_term(ctx, -BigRational::one() / a, tanh_terms)
        };
        ctx.add(Expr::Add(var_expr, scaled_tanh_terms))
    } else {
        let scaled_tanh_terms = if a.is_one() {
            tanh_terms
        } else if power == 8 {
            scale_reciprocal_integration_result_preserving_presentation(
                ctx,
                BigRational::one() / a,
                tanh_terms,
            )
        } else {
            scale_rational_term(ctx, BigRational::one() / a, tanh_terms)
        };
        ctx.add(Expr::Sub(var_expr, scaled_tanh_terms))
    };
    if power == 8 {
        Some(cas_ast::hold::wrap_hold(ctx, primitive))
    } else {
        Some(primitive)
    }
}

pub fn integrate_symbolic_is_affine_hyperbolic_square_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return false;
    };
    hyperbolic_square_affine_antiderivative(ctx, base, exp, var).is_some()
}

pub fn integrate_symbolic_is_affine_hyperbolic_cubic_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return false;
    };
    is_number(ctx, exp, 3)
        && hyperbolic_odd_power_limited_affine_antiderivative(ctx, base, exp, var).is_some()
}

pub fn integrate_symbolic_is_affine_hyperbolic_fifth_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return false;
    };
    let Expr::Function(_, args) = ctx.get(base).clone() else {
        return false;
    };
    if args.len() != 1 || is_var(ctx, args[0], var) {
        return false;
    }
    is_number(ctx, exp, 5)
        && hyperbolic_odd_power_limited_affine_antiderivative(ctx, base, exp, var).is_some()
}

pub fn integrate_symbolic_is_affine_hyperbolic_seventh_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return false;
    };
    let Expr::Function(_, args) = ctx.get(base).clone() else {
        return false;
    };
    if args.len() != 1 {
        return false;
    }
    is_number(ctx, exp, 7)
        && hyperbolic_odd_power_limited_affine_antiderivative(ctx, base, exp, var).is_some()
}

pub(super) fn hyperbolic_square_product_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut sinh_arg = None;
    let mut cosh_arg = None;

    for factor in factors {
        if let Some(value) = rational_constant_value(ctx, factor) {
            scale *= value;
            continue;
        }

        if let Some(arg) = squared_unary_builtin_arg(ctx, factor, BuiltinFn::Sinh) {
            if sinh_arg.replace(arg).is_some() {
                return None;
            }
            continue;
        }

        if let Some(arg) = squared_unary_builtin_arg(ctx, factor, BuiltinFn::Cosh) {
            if cosh_arg.replace(arg).is_some() {
                return None;
            }
            continue;
        }

        return None;
    }

    if scale.is_zero() {
        return None;
    }

    let sinh_arg = sinh_arg?;
    let cosh_arg = cosh_arg?;
    if compare_expr(ctx, sinh_arg, cosh_arg) != Ordering::Equal {
        return None;
    }

    let (a, _) = get_linear_coeffs(ctx, sinh_arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    let four = ctx.num(4);
    let quadruple_arg = mul2_raw(ctx, four, sinh_arg);
    let sinh_quadruple = ctx.call_builtin(BuiltinFn::Sinh, vec![quadruple_arg]);
    let oscillatory = scale_rational_term(
        ctx,
        scale.clone() / (BigRational::from_integer(32.into()) * a),
        sinh_quadruple,
    );
    let var_expr = ctx.var(var);
    let linear = scale_rational_term(ctx, scale / BigRational::from_integer(8.into()), var_expr);

    Some(ctx.add(Expr::Sub(oscillatory, linear)))
}

pub fn integrate_symbolic_is_hyperbolic_square_product_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    hyperbolic_square_product_antiderivative(ctx, expr, var).is_some()
}

pub(super) fn hyperbolic_power_times_derivative_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut powered: Option<(BuiltinFn, ExprId, i64)> = None;
    let mut derivative_factor: Option<(BuiltinFn, ExprId)> = None;

    for factor in factors {
        match ctx.get(factor).clone() {
            Expr::Pow(base, exp) => {
                let (fn_id, args) = match ctx.get(base).clone() {
                    Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
                    _ => {
                        scale *= rational_constant_value(ctx, factor)?;
                        continue;
                    }
                };
                let builtin = ctx.builtin_of(fn_id)?;
                if !matches!(builtin, BuiltinFn::Sinh | BuiltinFn::Cosh) {
                    scale *= rational_constant_value(ctx, factor)?;
                    continue;
                }
                let power = bounded_positive_integer_power(ctx, exp, 2, 5)?;
                if powered.replace((builtin, args[0], power)).is_some() {
                    return None;
                }
            }
            Expr::Function(fn_id, args) if args.len() == 1 => match ctx.builtin_of(fn_id) {
                Some(builtin @ (BuiltinFn::Sinh | BuiltinFn::Cosh)) => {
                    if derivative_factor.replace((builtin, args[0])).is_some() {
                        return None;
                    }
                }
                _ => scale *= rational_constant_value(ctx, factor)?,
            },
            _ => scale *= rational_constant_value(ctx, factor)?,
        }
    }

    let (powered_builtin, powered_arg, power) = powered?;
    let (derivative_builtin, derivative_arg) = derivative_factor?;
    if compare_expr(ctx, powered_arg, derivative_arg) != Ordering::Equal {
        return None;
    }

    match (powered_builtin, derivative_builtin) {
        (BuiltinFn::Sinh, BuiltinFn::Cosh) | (BuiltinFn::Cosh, BuiltinFn::Sinh) => {}
        _ => return None,
    }

    let (a, _) = get_linear_coeffs(ctx, powered_arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    let base = ctx.call_builtin(powered_builtin, vec![powered_arg]);
    let next_power = ctx.num(power + 1);
    let primitive = ctx.add(Expr::Pow(base, next_power));
    let denominator = BigRational::from_integer((power + 1).into()) * a;
    Some(scale_rational_term(ctx, scale / denominator, primitive))
}

pub(super) fn hyperbolic_odd_power_limited_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    let power = if is_number(ctx, exp, 3) {
        3
    } else if is_number(ctx, exp, 5) {
        5
    } else if is_number(ctx, exp, 7) {
        7
    } else {
        return None;
    };

    let (fn_id, args) = match ctx.get(base).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };

    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(builtin, BuiltinFn::Sinh | BuiltinFn::Cosh) {
        return None;
    }

    let arg = args[0];
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    let companion_builtin = match builtin {
        BuiltinFn::Sinh => BuiltinFn::Cosh,
        BuiltinFn::Cosh => BuiltinFn::Sinh,
        _ => return None,
    };
    let companion = ctx.call_builtin(companion_builtin, vec![arg]);

    let primitive = match power {
        3 => hyperbolic_cube_primitive(ctx, builtin, companion)?,
        5 => hyperbolic_fifth_primitive(ctx, builtin, companion)?,
        7 => hyperbolic_seventh_primitive(ctx, builtin, companion)?,
        _ => return None,
    };

    if a.is_one() {
        Some(primitive)
    } else {
        Some(scale_rational_term(ctx, BigRational::one() / a, primitive))
    }
}

fn hyperbolic_cube_primitive(
    ctx: &mut Context,
    builtin: BuiltinFn,
    companion: ExprId,
) -> Option<ExprId> {
    let three = ctx.num(3);
    let companion_cubed = ctx.add(Expr::Pow(companion, three));
    let cubic_term =
        scale_rational_term(ctx, BigRational::new(1.into(), 3.into()), companion_cubed);

    match builtin {
        BuiltinFn::Sinh => Some(ctx.add(Expr::Sub(cubic_term, companion))),
        BuiltinFn::Cosh => Some(ctx.add(Expr::Add(companion, cubic_term))),
        _ => None,
    }
}

fn hyperbolic_fifth_primitive(
    ctx: &mut Context,
    builtin: BuiltinFn,
    companion: ExprId,
) -> Option<ExprId> {
    let three = ctx.num(3);
    let five = ctx.num(5);
    let companion_cubed = ctx.add(Expr::Pow(companion, three));
    let companion_fifth = ctx.add(Expr::Pow(companion, five));

    let cubic_term =
        scale_rational_term(ctx, BigRational::new(2.into(), 3.into()), companion_cubed);
    let fifth_term =
        scale_rational_term(ctx, BigRational::new(1.into(), 5.into()), companion_fifth);

    match builtin {
        BuiltinFn::Sinh => {
            let first_two = ctx.add(Expr::Sub(fifth_term, cubic_term));
            Some(ctx.add(Expr::Add(first_two, companion)))
        }
        BuiltinFn::Cosh => {
            let first_two = ctx.add(Expr::Add(companion, cubic_term));
            Some(ctx.add(Expr::Add(first_two, fifth_term)))
        }
        _ => None,
    }
}

fn hyperbolic_seventh_primitive(
    ctx: &mut Context,
    builtin: BuiltinFn,
    companion: ExprId,
) -> Option<ExprId> {
    let three = ctx.num(3);
    let five = ctx.num(5);
    let seven = ctx.num(7);
    let companion_cubed = ctx.add(Expr::Pow(companion, three));
    let companion_fifth = ctx.add(Expr::Pow(companion, five));
    let companion_seventh = ctx.add(Expr::Pow(companion, seven));

    let fifth_term =
        scale_rational_term(ctx, BigRational::new(3.into(), 5.into()), companion_fifth);
    let seventh_term =
        scale_rational_term(ctx, BigRational::new(1.into(), 7.into()), companion_seventh);

    match builtin {
        BuiltinFn::Sinh => {
            let without_linear = ctx.add(Expr::Sub(companion_cubed, companion));
            let first_three = ctx.add(Expr::Sub(without_linear, fifth_term));
            Some(ctx.add(Expr::Add(first_three, seventh_term)))
        }
        BuiltinFn::Cosh => {
            let first_two = ctx.add(Expr::Add(companion, companion_cubed));
            let first_three = ctx.add(Expr::Add(first_two, fifth_term));
            Some(ctx.add(Expr::Add(first_three, seventh_term)))
        }
        _ => None,
    }
}

fn hyperbolic_reciprocal_power_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
    denominator_builtin: BuiltinFn,
    power: i64,
) -> Option<ExprId> {
    let policy = hyperbolic_reciprocal_table_policy(denominator_builtin, power)?;
    let arg = reciprocal_hyperbolic_power_arg(
        ctx,
        den,
        policy.denominator_builtin,
        policy.power.exponent(),
    )?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let scale = symbolic_linear_cofactor_scale_expr(ctx, num, arg, var)?;
    if is_number(ctx, scale, 0) {
        return None;
    }

    Some(build_hyperbolic_reciprocal_table_integral(
        ctx,
        policy,
        arg,
        scale,
        hyperbolic_reciprocal_primitive_scale_ops(),
    ))
}

fn hyperbolic_reciprocal_primitive_scale_ops() -> HyperbolicReciprocalPrimitiveScaleOps {
    HyperbolicReciprocalPrimitiveScaleOps::new(
        scale_rational_term,
        rational_over_expr,
        scale_expr_reciprocal_integration_result,
        negate_scalar_expr,
    )
}

pub(super) fn hyperbolic_log_derivative_ratio_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den, scale) = hyperbolic_log_derivative_ratio_parts(ctx, num, den, var)?;
    Some(hyperbolic_log_derivative_ratio_antiderivative_from_parts(
        ctx, den, scale,
    ))
}

pub(super) fn hyperbolic_tanh_log_cosh_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (arg, scale) = hyperbolic_tanh_log_cosh_parts(ctx, expr, var)?;
    let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![arg]);
    let log_abs = ln_abs(ctx, cosh_arg);
    Some(scale_expr_reciprocal_integration_result(
        ctx, scale, log_abs,
    ))
}

pub(super) fn hyperbolic_tanh_reciprocal_log_sinh_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (arg, scale) = hyperbolic_tanh_reciprocal_log_sinh_parts(ctx, num, den, var)?;
    Some(hyperbolic_tanh_reciprocal_log_sinh_antiderivative_from_parts(ctx, arg, scale))
}

fn hyperbolic_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = hyperbolic_reciprocal_derivative_parts(ctx, num, den, var)?;
    hyperbolic_reciprocal_derivative_antiderivative_from_parts(ctx, parts)
}

pub(super) fn hyperbolic_reciprocal_square_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    hyperbolic_reciprocal_power_antiderivative(ctx, num, den, var, BuiltinFn::Cosh, 1)
        .or_else(|| {
            hyperbolic_reciprocal_power_antiderivative(ctx, num, den, var, BuiltinFn::Sinh, 1)
        })
        .or_else(|| {
            hyperbolic_reciprocal_power_antiderivative(ctx, num, den, var, BuiltinFn::Cosh, 2)
        })
        .or_else(|| {
            hyperbolic_reciprocal_power_antiderivative(ctx, num, den, var, BuiltinFn::Cosh, 4)
        })
        .or_else(|| {
            hyperbolic_reciprocal_power_antiderivative(ctx, num, den, var, BuiltinFn::Sinh, 2)
        })
        .or_else(|| {
            hyperbolic_reciprocal_power_antiderivative(ctx, num, den, var, BuiltinFn::Sinh, 4)
        })
        .or_else(|| hyperbolic_reciprocal_derivative_antiderivative(ctx, num, den, var))
}

pub(super) fn constant_scaled_hyperbolic_reciprocal_square_antiderivative(
    ctx: &mut Context,
    constant: ExprId,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    let scaled_num = mul2_raw(ctx, constant, num);
    hyperbolic_reciprocal_square_antiderivative(ctx, scaled_num, den, var)
}

pub(super) fn sqrt_hyperbolic_log_derivative_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_hyperbolic_log_derivative_parts(ctx, expr, var)?;
    let log_arg = ctx.call_builtin(parts.log_builtin, vec![parts.arg]);
    let integral = ln_abs(ctx, log_arg);
    Some(scale_rational_term(ctx, parts.scale, integral))
}

pub fn integrate_symbolic_is_sqrt_hyperbolic_log_derivative_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    sqrt_hyperbolic_log_derivative_parts(ctx, expr, var).is_some()
}

pub(super) fn sqrt_hyperbolic_reciprocal_square_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_hyperbolic_reciprocal_square_parts(ctx, expr, var)?;
    let integral = match parts.denominator_builtin {
        BuiltinFn::Cosh => ctx.call_builtin(BuiltinFn::Tanh, vec![parts.arg]),
        BuiltinFn::Sinh => {
            let one = ctx.num(1);
            let tanh_arg = ctx.call_builtin(BuiltinFn::Tanh, vec![parts.arg]);
            let reciprocal_tanh = ctx.add(Expr::Div(one, tanh_arg));
            ctx.add(Expr::Neg(reciprocal_tanh))
        }
        _ => return None,
    };
    Some(scale_expr_reciprocal_integration_result(
        ctx,
        parts.scale,
        integral,
    ))
}

pub fn integrate_symbolic_is_sqrt_hyperbolic_reciprocal_square_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    sqrt_hyperbolic_reciprocal_square_parts(ctx, expr, var).is_some()
}

pub(super) fn sqrt_hyperbolic_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_hyperbolic_reciprocal_derivative_parts(ctx, expr, var)?;
    let policy = hyperbolic_reciprocal_derivative_policy(parts.denominator_builtin)?;
    let integral = build_hyperbolic_reciprocal_derivative_integral(ctx, policy, parts.arg);
    Some(scale_expr_reciprocal_integration_result(
        ctx,
        parts.scale,
        integral,
    ))
}

pub fn integrate_symbolic_is_sqrt_hyperbolic_reciprocal_derivative_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    sqrt_hyperbolic_reciprocal_derivative_parts(ctx, expr, var).is_some()
}

pub(super) fn sqrt_hyperbolic_log_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_hyperbolic_log_derivative_parts(ctx, expr, var)?;
    build_hyperbolic_denominator_nonzero_condition(ctx, parts.log_builtin, parts.arg)
}

pub(super) fn sqrt_hyperbolic_reciprocal_square_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_hyperbolic_reciprocal_square_parts(ctx, expr, var)?;
    sqrt_hyperbolic_reciprocal_parts_required_nonzero(ctx, &parts)
}

pub(super) fn sqrt_hyperbolic_reciprocal_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_hyperbolic_reciprocal_derivative_parts(ctx, expr, var)?;
    sqrt_hyperbolic_reciprocal_parts_required_nonzero(ctx, &parts)
}

pub(super) fn asinh_affine_antiderivative(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (coeff, offset) = get_linear_coeffs(ctx, arg, var)?;
    let coeff = rational_constant_value(ctx, coeff)?;
    if coeff.is_zero() {
        return None;
    }

    let asinh_arg = ctx.call_builtin(BuiltinFn::Asinh, vec![arg]);
    let leading_term = mul2_raw(ctx, arg, asinh_arg);

    let two = ctx.num(2);
    let arg_sq = ctx.add(Expr::Pow(arg, two));
    let one = ctx.num(1);
    let sqrt_arg = ctx.add(Expr::Add(arg_sq, one));
    let sqrt_arg = if is_number(ctx, offset, 0) {
        sqrt_arg
    } else {
        cas_ast::hold::wrap_hold(ctx, sqrt_arg)
    };
    let sqrt_term = ctx.call_builtin(BuiltinFn::Sqrt, vec![sqrt_arg]);
    let primitive = ctx.add(Expr::Sub(leading_term, sqrt_term));

    let scale = BigRational::one() / coeff;
    if scale.is_one() {
        Some(primitive)
    } else {
        let scaled_leading = if is_number(ctx, offset, 0) {
            let var_expr = ctx.var(var);
            mul2_raw(ctx, var_expr, asinh_arg)
        } else {
            scale_rational_term(ctx, scale.clone(), leading_term)
        };
        let sqrt_scale = -scale;
        let scaled_sqrt = scale_rational_term(ctx, sqrt_scale, sqrt_term);
        Some(ctx.add(Expr::Add(scaled_leading, scaled_sqrt)))
    }
}

pub(super) fn atanh_affine_antiderivative(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (coeff, offset) = get_linear_coeffs(ctx, arg, var)?;
    let coeff = rational_constant_value(ctx, coeff)?;
    if coeff.is_zero() {
        return None;
    }

    let atanh_arg = ctx.call_builtin(BuiltinFn::Atanh, vec![arg]);
    let leading_term = mul2_raw(ctx, arg, atanh_arg);

    let log_arg = unit_minus_square(ctx, arg);
    let log_term = ctx.call_builtin(BuiltinFn::Ln, vec![log_arg]);
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let half_log = mul2_raw(ctx, half, log_term);
    let primitive = ctx.add(Expr::Add(leading_term, half_log));

    let scale = BigRational::one() / coeff;
    if scale.is_one() {
        Some(primitive)
    } else {
        let scaled_leading = if is_number(ctx, offset, 0) {
            let var_expr = ctx.var(var);
            mul2_raw(ctx, var_expr, atanh_arg)
        } else {
            scale_rational_term(ctx, scale.clone(), leading_term)
        };
        let half_scale = scale / BigRational::from_integer(2.into());
        let scaled_log = scale_rational_term(ctx, half_scale, log_term);
        Some(ctx.add(Expr::Add(scaled_log, scaled_leading)))
    }
}

fn acosh_radicands(ctx: &mut Context, arg: ExprId) -> (ExprId, ExprId) {
    let one = ctx.num(1);
    let left = ctx.add(Expr::Sub(arg, one));
    let right = ctx.add(Expr::Add(arg, one));
    (left, right)
}

fn acosh_polynomial_radicands(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let one_poly = Polynomial::one(var.to_string());
    let left = arg_poly.sub(&one_poly).to_expr(ctx);
    let right = arg_poly.add(&one_poly).to_expr(ctx);
    Some((left, right))
}

pub(super) fn acosh_affine_antiderivative(
    ctx: &mut Context,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (coeff, offset) = get_linear_coeffs(ctx, arg, var)?;
    let coeff = rational_constant_value(ctx, coeff)?;
    if coeff.is_zero() {
        return None;
    }

    let acosh_arg = ctx.call_builtin(BuiltinFn::Acosh, vec![arg]);
    let leading_term = mul2_raw(ctx, arg, acosh_arg);

    let (left, right) =
        acosh_polynomial_radicands(ctx, arg, var).unwrap_or_else(|| acosh_radicands(ctx, arg));
    let sqrt_left = ctx.call_builtin(BuiltinFn::Sqrt, vec![left]);
    let sqrt_right = ctx.call_builtin(BuiltinFn::Sqrt, vec![right]);
    let sqrt_product = mul2_raw(ctx, sqrt_left, sqrt_right);

    if coeff.is_negative() {
        let held_sqrt_product = cas_ast::hold::wrap_hold(ctx, sqrt_product);
        let held_leading_term = cas_ast::hold::wrap_hold(ctx, leading_term);
        let primitive = ctx.add(Expr::Sub(held_sqrt_product, held_leading_term));
        let scale = -BigRational::one() / coeff;
        if scale.is_one() {
            return Some(primitive);
        }
        let scaled_sqrt = scale_rational_term(ctx, scale.clone(), held_sqrt_product);
        let leading_scale = -scale;
        let scaled_leading = if is_number(ctx, offset, 0) {
            let var_expr = ctx.var(var);
            mul2_raw(ctx, var_expr, acosh_arg)
        } else {
            scale_rational_term(ctx, leading_scale, held_leading_term)
        };
        return Some(ctx.add(Expr::Add(scaled_sqrt, scaled_leading)));
    }

    let primitive = ctx.add(Expr::Sub(leading_term, sqrt_product));

    let scale = BigRational::one() / coeff;
    if scale.is_one() {
        Some(primitive)
    } else {
        let scaled_leading = if is_number(ctx, offset, 0) {
            let var_expr = ctx.var(var);
            mul2_raw(ctx, var_expr, acosh_arg)
        } else {
            scale_rational_term(ctx, scale.clone(), leading_term)
        };
        let sqrt_scale = -scale;
        let scaled_sqrt = scale_rational_term(ctx, sqrt_scale, sqrt_product);
        Some(ctx.add(Expr::Add(scaled_leading, scaled_sqrt)))
    }
}

pub(super) fn monomial_over_sqrt_hyperbolic_reduction(
    ctx: &mut Context,
    n: usize,
    a: &BigRational,
    b: &BigRational,
    radicand: ExprId,
    var: &str,
) -> Option<ExprId> {
    if n <= 1 {
        let var_expr = ctx.var(var);
        let sqrt_term = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
        let one = ctx.num(1);
        let base_integrand = if n == 0 {
            ctx.add(Expr::Div(one, sqrt_term))
        } else {
            ctx.add(Expr::Div(var_expr, sqrt_term))
        };
        return integrate_symbolic_expr(ctx, base_integrand, var);
    }
    let n_rational = BigRational::from_integer((n as i64).into());
    let n_minus_one = BigRational::from_integer(((n - 1) as i64).into());
    let lower = monomial_over_sqrt_hyperbolic_reduction(ctx, n - 2, a, b, radicand, var)?;
    let lower_scale = -(a * &n_minus_one) / (b * &n_rational);
    let lower_term = scale_rational_term(ctx, lower_scale, lower);

    let var_expr = ctx.var(var);
    let head_power = ctx.num((n - 1) as i64);
    let head_monomial = ctx.add(Expr::Pow(var_expr, head_power));
    let sqrt_term = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let head_raw = mul2_raw(ctx, head_monomial, sqrt_term);
    let head_scale = BigRational::one() / (b * &n_rational);
    let head_term = scale_rational_term(ctx, head_scale, head_raw);
    Some(ctx.add(Expr::Add(lower_term, head_term)))
}

pub fn integrate_symbolic_is_asinh_affine_variable_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let (fn_id, args) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return false,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Asinh) {
        return false;
    }

    let Some((coeff, _)) = get_linear_coeffs(ctx, args[0], var) else {
        return false;
    };
    rational_constant_value(ctx, coeff).is_some_and(|coeff| !coeff.is_zero())
}

pub fn integrate_symbolic_is_atanh_affine_variable_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let (fn_id, args) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return false,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Atanh) {
        return false;
    }

    let Some((coeff, _)) = get_linear_coeffs(ctx, args[0], var) else {
        return false;
    };
    rational_constant_value(ctx, coeff).is_some_and(|coeff| !coeff.is_zero())
}

pub fn integrate_symbolic_is_acosh_affine_variable_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let (fn_id, args) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return false,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Acosh) {
        return false;
    }

    let Some((coeff, _)) = get_linear_coeffs(ctx, args[0], var) else {
        return false;
    };
    rational_constant_value(ctx, coeff).is_some_and(|coeff| !coeff.is_zero())
}

/// `(builtin, arg)` when `expr` is `sinh(arg)` or `cosh(arg)`.
fn hyperbolic_like_factor(ctx: &Context, expr: ExprId) -> Option<(BuiltinFn, ExprId)> {
    if let Expr::Function(fn_id, args) = ctx.get(expr) {
        if args.len() == 1 {
            if let Some(b @ (BuiltinFn::Sinh | BuiltinFn::Cosh)) = ctx.builtin_of(*fn_id) {
                return Some((b, args[0]));
            }
        }
    }
    None
}

/// `p(x) * trig(ax+b) * sinh(cx+d)` (and the cosh / exp variants): products that
/// pair a sinh/cosh with another transcendental (a trig or an exponential) have
/// no dedicated owner, but lowering the hyperbolic to its exponential form turns
/// them into the exp-times-trig / exp-times-exp families that DO. Lower exactly
/// one sinh/cosh factor -- `sinh(u) = (e^u - e^(-u))/2`, `cosh(u) = (e^u + e^(-u))/2`
/// -- DISTRIBUTE by the rest of the product (the integrator does not distribute a
/// `rest*(sum)` itself), and delegate. Resolves `sin(x)sinh(x)`, `cos(x)cosh(x)`,
/// `e^x sinh(x)`, `x sin(x) cosh(x)`. Requires a trig/exp partner so the
/// poly-times-hyperbolic and sinh^2 owners (which run first) keep their cases;
/// delegation self-gates anything whose lowered form is not elementary.
pub(super) fn hyperbolic_transcendental_product_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }
    let mut hyp: Option<(usize, BuiltinFn, ExprId)> = None;
    let mut has_trig_or_exp = false;
    for (i, factor) in factors.iter().enumerate() {
        if let Some((builtin, arg)) = hyperbolic_like_factor(ctx, *factor) {
            // Affine argument with a nonzero rational slope.
            if nonzero_linear_polynomial_from_expr(ctx, arg, var)
                .ok()
                .flatten()
                .is_none()
            {
                continue;
            }
            if hyp.is_some() {
                return None; // more than one hyperbolic factor is out of scope
            }
            hyp = Some((i, builtin, arg));
        } else if linear_trig_factor_parts(ctx, *factor, var)
            .ok()
            .flatten()
            .is_some()
            || linear_exp_factor_parts(ctx, *factor, var)
                .ok()
                .flatten()
                .is_some()
        {
            has_trig_or_exp = true;
        }
    }
    let (hyp_idx, hyp_builtin, hyp_arg) = hyp?;
    if !has_trig_or_exp {
        return None; // poly-times-hyperbolic is owned elsewhere
    }

    let rest_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != hyp_idx)
        .map(|(_, f)| *f)
        .collect();
    let rest = build_balanced_mul(ctx, &rest_factors);

    // Lower the hyperbolic to exp and distribute: rest*(e^u -/+ e^(-u))/2.
    let e = ctx.add(Expr::Constant(Constant::E));
    let exp_u = ctx.add(Expr::Pow(e, hyp_arg));
    let neg_arg = ctx.add(Expr::Neg(hyp_arg));
    let exp_neg_u = ctx.add(Expr::Pow(e, neg_arg));
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let rest_exp_u = mul2_raw(ctx, rest, exp_u);
    let rest_exp_neg_u = mul2_raw(ctx, rest, exp_neg_u);
    let term_plus = mul2_raw(ctx, half, rest_exp_u);
    let term_minus = mul2_raw(ctx, half, rest_exp_neg_u);
    let rewritten = match hyp_builtin {
        BuiltinFn::Sinh => ctx.add(Expr::Sub(term_plus, term_minus)),
        BuiltinFn::Cosh => ctx.add(Expr::Add(term_plus, term_minus)),
        _ => return None,
    };
    integrate_symbolic_expr(ctx, rewritten, var)
}

pub(super) fn linear_times_hyperbolic_linear_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (outer_sign, factors) = signed_mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (hyperbolic_index, factor) in factors.iter().enumerate() {
        let Some(hyperbolic_parts) =
            signed_linear_function_factor_parts(ctx, *factor, var, signed_hyperbolic_like_factor)
        else {
            continue;
        };

        let effective_sign = combine_factor_signs(outer_sign, hyperbolic_parts.sign);
        let cofactor =
            signed_factor_product_excluding_index(ctx, &factors, hyperbolic_index, effective_sign);
        let Ok(cofactor_poly) = Polynomial::from_expr(ctx, cofactor, var) else {
            continue;
        };
        if cofactor_poly.degree() != 1 {
            continue;
        }

        let cofactor_slope = cofactor_poly
            .coeffs
            .get(1)
            .cloned()
            .unwrap_or_else(BigRational::zero);
        if cofactor_slope.is_zero() {
            continue;
        }

        let arg_slope_expr = ctx.add(Expr::Number(hyperbolic_parts.arg_slope.clone()));
        let quotient = if hyperbolic_parts.arg_slope.is_one() {
            cofactor
        } else {
            ctx.add(Expr::Div(cofactor, arg_slope_expr))
        };
        let correction =
            cofactor_slope / (hyperbolic_parts.arg_slope.clone() * hyperbolic_parts.arg_slope);

        return match hyperbolic_parts.builtin {
            BuiltinFn::Sinh => {
                let cosh_arg = ctx.call_builtin(BuiltinFn::Cosh, vec![hyperbolic_parts.arg]);
                let quotient_cosh = mul2_raw(ctx, quotient, cosh_arg);
                let correction_sinh = scale_factor(ctx, correction, hyperbolic_parts.factor);
                Some(ctx.add(Expr::Sub(quotient_cosh, correction_sinh)))
            }
            BuiltinFn::Cosh => {
                let sinh_arg = ctx.call_builtin(BuiltinFn::Sinh, vec![hyperbolic_parts.arg]);
                let quotient_sinh = mul2_raw(ctx, quotient, sinh_arg);
                let correction_cosh = scale_factor(ctx, correction, hyperbolic_parts.factor);
                Some(ctx.add(Expr::Sub(quotient_sinh, correction_cosh)))
            }
            _ => None,
        };
    }

    None
}

pub fn integrate_symbolic_is_linear_times_hyperbolic_linear_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    linear_times_hyperbolic_linear_antiderivative(ctx, expr, var).is_some()
}

fn polynomial_hyperbolic_term(
    ctx: &mut Context,
    poly: &Polynomial,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    if poly.is_zero() {
        return None;
    }

    let hyperbolic = ctx.call_builtin(builtin, vec![arg]);
    let poly_expr = poly.to_expr(ctx);
    if poly.degree() == 0 && poly.coeffs.first().is_some_and(BigRational::is_one) {
        Some(hyperbolic)
    } else {
        Some(mul2_raw(ctx, poly_expr, hyperbolic))
    }
}

pub(super) fn polynomial_times_hyperbolic_linear_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (outer_sign, factors) = signed_mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (hyperbolic_index, factor) in factors.iter().enumerate() {
        let Some(hyperbolic_parts) =
            signed_linear_function_factor_parts(ctx, *factor, var, signed_hyperbolic_like_factor)
        else {
            continue;
        };

        let effective_sign = combine_factor_signs(outer_sign, hyperbolic_parts.sign);
        let cofactor =
            signed_factor_product_excluding_index(ctx, &factors, hyperbolic_index, effective_sign);
        let Ok(cofactor_poly) = Polynomial::from_expr(ctx, cofactor, var) else {
            continue;
        };
        if !(2..=MAX_HYPERBOLIC_POLYNOMIAL_BY_PARTS_DEGREE).contains(&cofactor_poly.degree()) {
            continue;
        }

        let slope_sq = hyperbolic_parts.arg_slope.clone() * hyperbolic_parts.arg_slope.clone();
        let slope_cube = slope_sq.clone() * hyperbolic_parts.arg_slope.clone();
        let slope_fourth = slope_cube.clone() * hyperbolic_parts.arg_slope.clone();
        let slope_fifth = slope_fourth.clone() * hyperbolic_parts.arg_slope.clone();
        let slope_sixth = slope_fifth.clone() * hyperbolic_parts.arg_slope.clone();
        let slope_seventh = slope_sixth.clone() * hyperbolic_parts.arg_slope.clone();
        let slope_eighth = slope_seventh.clone() * hyperbolic_parts.arg_slope.clone();
        let first_derivative = cofactor_poly.derivative();
        let second_derivative = first_derivative.derivative();
        let third_derivative = second_derivative.derivative();
        let fourth_derivative = third_derivative.derivative();
        let fifth_derivative = fourth_derivative.derivative();
        let sixth_derivative = fifth_derivative.derivative();
        let seventh_derivative = sixth_derivative.derivative();

        let even_poly = cofactor_poly
            .div_scalar(&hyperbolic_parts.arg_slope)
            .add(&second_derivative.div_scalar(&slope_cube))
            .add(&fourth_derivative.div_scalar(&slope_fifth))
            .add(&sixth_derivative.div_scalar(&slope_seventh));
        let odd_poly = first_derivative
            .div_scalar(&slope_sq)
            .add(&third_derivative.div_scalar(&slope_fourth))
            .add(&fifth_derivative.div_scalar(&slope_sixth))
            .add(&seventh_derivative.div_scalar(&slope_eighth));

        let (even_builtin, odd_builtin) = match hyperbolic_parts.builtin {
            BuiltinFn::Sinh => (BuiltinFn::Cosh, BuiltinFn::Sinh),
            BuiltinFn::Cosh => (BuiltinFn::Sinh, BuiltinFn::Cosh),
            _ => return None,
        };

        let even_term =
            polynomial_hyperbolic_term(ctx, &even_poly, even_builtin, hyperbolic_parts.arg)?;
        let Some(odd_term) =
            polynomial_hyperbolic_term(ctx, &odd_poly, odd_builtin, hyperbolic_parts.arg)
        else {
            return Some(even_term);
        };
        return Some(ctx.add(Expr::Sub(even_term, odd_term)));
    }

    None
}

pub fn integrate_symbolic_is_polynomial_times_hyperbolic_linear_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    if is_polynomial_times_linear_function_target(
        ctx,
        expr,
        var,
        2,
        MAX_HYPERBOLIC_POLYNOMIAL_BY_PARTS_DEGREE,
        signed_hyperbolic_like_factor,
    ) {
        return true;
    }

    additive_polynomial_times_hyperbolic_linear_antiderivative(ctx, expr, var).is_some()
}

pub(super) fn additive_polynomial_times_hyperbolic_linear_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let view = AddView::from_expr(ctx, expr);
    if view.terms.len() < 2 {
        return None;
    }

    let mut common_builtin = None;
    let mut common_arg = None;
    let mut common_factor = None;
    let mut cofactor_sum = Polynomial::zero(var.to_string());

    for (term, sign) in view.terms {
        let (builtin, arg, factor, mut cofactor) =
            polynomial_hyperbolic_linear_term_parts(ctx, term, var)?;
        if sign == Sign::Neg {
            cofactor = cofactor.neg();
        }

        if let Some(existing_builtin) = common_builtin {
            if builtin != existing_builtin || compare_expr(ctx, arg, common_arg?) != Ordering::Equal
            {
                return None;
            }
        } else {
            common_builtin = Some(builtin);
            common_arg = Some(arg);
            common_factor = Some(factor);
        }

        cofactor_sum = cofactor_sum.add(&cofactor);
    }

    if !(2..=MAX_HYPERBOLIC_POLYNOMIAL_BY_PARTS_DEGREE).contains(&cofactor_sum.degree()) {
        return None;
    }

    let cofactor = cofactor_sum.to_expr(ctx);
    let combined_expr = mul2_raw(ctx, cofactor, common_factor?);
    polynomial_times_hyperbolic_linear_antiderivative(ctx, combined_expr, var)
}

pub(super) fn atanh_surd_offset_antiderivative(
    ctx: &mut Context,
    numerator: &Polynomial,
    arg_poly: &Polynomial,
    offset_square: &BigRational,
) -> Option<ExprId> {
    let derivative = arg_poly.derivative();
    let mut scale = constant_polynomial_ratio(numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let mut arg_poly = arg_poly.clone();
    let mut offset_square = offset_square.clone();
    if let Some((normalized_arg, normalized_offset_square, denominator_scale)) =
        reduce_surd_offset_by_square_denominator(&arg_poly, &offset_square)
    {
        arg_poly = normalized_arg;
        offset_square = normalized_offset_square;
        scale *= denominator_scale;
    }
    if let Some((reduced_arg, reduced_offset_square, common_factor)) =
        reduce_surd_offset_by_common_square_factor(&arg_poly, &offset_square)
    {
        arg_poly = reduced_arg;
        offset_square = reduced_offset_square;
        scale /= common_factor;
    }

    let offset_expr = positive_rational_sqrt_expr(ctx, &offset_square)?;
    let arg = arg_poly.to_expr(ctx);
    let atanh_arg = ctx.add(Expr::Div(arg, offset_expr));
    let atanh = ctx.call_builtin(BuiltinFn::Atanh, vec![atanh_arg]);
    let numerator = if scale.is_one() {
        atanh
    } else {
        let scale_num = ctx.add(Expr::Number(scale));
        mul2_raw(ctx, scale_num, atanh)
    };
    Some(ctx.add(Expr::Div(numerator, offset_expr)))
}

pub(super) fn sqrt_hyperbolic_log_derivative_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_hyperbolic_log_derivative_parts(ctx, expr, var)?;
    Some(parts.radicand)
}

pub(super) fn sqrt_hyperbolic_reciprocal_square_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_hyperbolic_reciprocal_square_parts(ctx, expr, var)?;
    Some(sqrt_hyperbolic_reciprocal_parts_radicand(&parts))
}

pub(super) fn sqrt_hyperbolic_reciprocal_derivative_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_hyperbolic_reciprocal_derivative_parts(ctx, expr, var)?;
    Some(sqrt_hyperbolic_reciprocal_parts_radicand(&parts))
}

pub(super) fn tanh_nonzero_dominated_by_sinh_nonzero(
    ctx: &Context,
    condition: ExprId,
    all_conditions: &[ExprId],
) -> bool {
    let Some(tanh_arg) = unary_builtin_arg(ctx, condition, BuiltinFn::Tanh) else {
        return false;
    };

    all_conditions.iter().any(|other| {
        let Some(sinh_arg) = unary_builtin_arg(ctx, *other, BuiltinFn::Sinh) else {
            return false;
        };
        compare_expr(ctx, sinh_arg, tanh_arg) == Ordering::Equal
    })
}

pub(super) fn asinh_sqrt_reciprocal_positive_condition(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    inverse_hyperbolic_sqrt_reciprocal_positive_condition(
        ctx,
        expr,
        var,
        InverseHyperbolicSqrtReciprocalKind::Asinh,
    )
}

pub(super) fn atanh_sqrt_reciprocal_positive_condition(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    inverse_hyperbolic_sqrt_reciprocal_positive_condition(
        ctx,
        expr,
        var,
        InverseHyperbolicSqrtReciprocalKind::Atanh,
    )
}

pub(super) fn acosh_affine_radicands(ctx: &mut Context, expr: ExprId, var: &str) -> Vec<ExprId> {
    let (fn_id, args) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return vec![],
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Acosh) {
        return vec![];
    }

    if !integrate_symbolic_is_acosh_affine_variable_target(ctx, expr, var) {
        return vec![];
    }

    let (left, right) = acosh_radicands(ctx, args[0]);
    vec![left, right]
}
