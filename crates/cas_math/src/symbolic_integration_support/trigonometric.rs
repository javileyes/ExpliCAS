//! `symbolic_integration_support`: familia `trigonometric`.
//!
//! Ver la cabecera de `symbolic_integration_support.rs` para el contexto.

use super::*;

pub(super) fn reciprocal_trig_square_antiderivative(
    ctx: &mut Context,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;

    let integral = match builtin {
        BuiltinFn::Cos => ctx.call_builtin(BuiltinFn::Tan, vec![arg]),
        BuiltinFn::Sin => {
            let cot_arg = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
            ctx.add(Expr::Neg(cot_arg))
        }
        _ => return None,
    };

    let is_a_one = if let Expr::Number(n) = ctx.get(a) {
        n.is_one()
    } else {
        false
    };
    if is_a_one {
        Some(integral)
    } else if matches!(builtin, BuiltinFn::Sin) {
        let scaled = ctx.add(Expr::Div(integral, a));
        Some(cas_ast::hold::wrap_hold(ctx, scaled))
    } else {
        Some(ctx.add(Expr::Div(integral, a)))
    }
}

pub(super) fn trig_square_affine_antiderivative(
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
    if !matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos) {
        return None;
    }

    let arg = args[0];
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    let var_expr = ctx.var(var);
    let two = ctx.num(2);
    let half_linear = ctx.add(Expr::Div(var_expr, two));

    let two = ctx.num(2);
    let double_arg = mul2_raw(ctx, two, arg);
    let sin_double = ctx.call_builtin(BuiltinFn::Sin, vec![double_arg]);
    let four = BigRational::from_integer(4.into());
    let oscillatory_scale = match builtin {
        BuiltinFn::Sin => -BigRational::one() / (four * a),
        BuiltinFn::Cos => BigRational::one() / (four * a),
        _ => return None,
    };
    let oscillatory = scale_rational_term(ctx, oscillatory_scale, sin_double);

    Some(ctx.add(Expr::Add(half_linear, oscillatory)))
}

pub(super) fn trig_ratio_square_affine_antiderivative(
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
    if !matches!(builtin, BuiltinFn::Tan | BuiltinFn::Cot) {
        return None;
    }

    let arg = args[0];
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    Some(trig_ratio_square_antiderivative_from_parts(
        ctx,
        TrigRatioSquareParts { builtin, arg, a },
        var,
    ))
}

pub(super) fn trig_tan_cot_odd_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    let power = if is_number(ctx, exp, 3) {
        3
    } else if is_number(ctx, exp, 5) {
        5
    } else {
        return None;
    };
    let (fn_id, args) = match ctx.get(base).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };
    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(builtin, BuiltinFn::Tan | BuiltinFn::Cot) {
        return None;
    }
    let arg = args[0];
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }
    Some(match (builtin, power) {
        (BuiltinFn::Tan, 3) => trig_tan_third_antiderivative_from_parts(ctx, arg, a),
        (BuiltinFn::Tan, _) => trig_tan_fifth_antiderivative_from_parts(ctx, arg, a),
        (_, 3) => trig_cot_third_antiderivative_from_parts(ctx, arg, a),
        _ => trig_cot_fifth_antiderivative_from_parts(ctx, arg, a),
    })
}

pub(super) fn trig_tan_fourth_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    if !is_number(ctx, exp, 4) {
        return None;
    }

    let (fn_id, args) = match ctx.get(base).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Tan) {
        return None;
    }

    let arg = args[0];
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    Some(trig_tan_fourth_antiderivative_from_parts(ctx, arg, a, var))
}

pub(super) fn trig_tan_sixth_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    if !is_number(ctx, exp, 6) {
        return None;
    }

    let (fn_id, args) = match ctx.get(base).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Tan) {
        return None;
    }

    let arg = args[0];
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    Some(trig_tan_sixth_antiderivative_from_parts(ctx, arg, a, var))
}

pub(super) fn trig_tan_eighth_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    if !is_number(ctx, exp, 8) {
        return None;
    }

    let (fn_id, args) = match ctx.get(base).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Tan) {
        return None;
    }

    let arg = args[0];
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    Some(trig_tan_eighth_antiderivative_from_parts(ctx, arg, a, var))
}

pub(super) fn trig_cot_fourth_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    if !is_number(ctx, exp, 4) {
        return None;
    }

    let (fn_id, args) = match ctx.get(base).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Cot) {
        return None;
    }

    let arg = args[0];
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    Some(trig_cot_fourth_antiderivative_from_parts(ctx, arg, a, var))
}

pub(super) fn trig_cot_sixth_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    if !is_number(ctx, exp, 6) {
        return None;
    }

    let (fn_id, args) = match ctx.get(base).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Cot) {
        return None;
    }

    let arg = args[0];
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    Some(trig_cot_sixth_antiderivative_from_parts(ctx, arg, a, var))
}

pub(super) fn trig_cot_eighth_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    if !is_number(ctx, exp, 8) {
        return None;
    }

    let (fn_id, args) = match ctx.get(base).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Cot) {
        return None;
    }

    let arg = args[0];
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    Some(trig_cot_eighth_antiderivative_from_parts(ctx, arg, a, var))
}

pub(super) fn trig_abs_log_term(ctx: &mut Context, builtin: BuiltinFn, arg: ExprId) -> ExprId {
    let inner = ctx.call_builtin(builtin, vec![arg]);
    let abs_inner = ctx.call_builtin(BuiltinFn::Abs, vec![inner]);
    ctx.call_builtin(BuiltinFn::Ln, vec![abs_inner])
}

pub(super) fn reciprocal_trig_abs_log_term(
    ctx: &mut Context,
    reciprocal_builtin: BuiltinFn,
    arg: ExprId,
) -> ExprId {
    let log_arg = build_reciprocal_trig_log_argument(ctx, reciprocal_builtin, arg)
        .expect("sec/csc log argument");
    let abs_arg = ctx.call_builtin(BuiltinFn::Abs, vec![log_arg]);
    ctx.call_builtin(BuiltinFn::Ln, vec![abs_arg])
}

pub(super) fn reciprocal_trig_power_affine_arg(
    ctx: &mut Context,
    base: ExprId,
    var: &str,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    let arg = unary_builtin_arg(ctx, base, builtin)?;
    get_linear_coeffs(ctx, arg, var)?;
    Some(arg)
}

pub(super) fn trig_sec_third_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_affine_parts(ctx, base, exp, var, BuiltinFn::Sec, 3)?;
    Some(trig_sec_third_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_csc_third_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_affine_parts(ctx, base, exp, var, BuiltinFn::Csc, 3)?;
    Some(trig_csc_third_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_sec_fifth_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_affine_parts(ctx, base, exp, var, BuiltinFn::Sec, 5)?;
    Some(trig_sec_fifth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_csc_fifth_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_affine_parts(ctx, base, exp, var, BuiltinFn::Csc, 5)?;
    Some(trig_csc_fifth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_sec_fourth_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_affine_parts(ctx, base, exp, var, BuiltinFn::Sec, 4)?;
    Some(trig_sec_fourth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_sec_sixth_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_affine_parts(ctx, base, exp, var, BuiltinFn::Sec, 6)?;
    Some(trig_sec_sixth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_sec_eighth_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_affine_parts(ctx, base, exp, var, BuiltinFn::Sec, 8)?;
    Some(trig_sec_eighth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_csc_fourth_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_affine_parts(ctx, base, exp, var, BuiltinFn::Csc, 4)?;
    Some(trig_csc_fourth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_csc_sixth_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_affine_parts(ctx, base, exp, var, BuiltinFn::Csc, 6)?;
    Some(trig_csc_sixth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_csc_eighth_affine_antiderivative(
    ctx: &mut Context,
    base: ExprId,
    exp: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_affine_parts(ctx, base, exp, var, BuiltinFn::Csc, 8)?;
    Some(trig_csc_eighth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn reciprocal_trig_power_quotient_arg(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
    denominator_builtin: BuiltinFn,
    power: i64,
) -> Option<ExprId> {
    if !is_number(ctx, num, 1) {
        return None;
    }
    let arg = powered_unary_builtin_arg(ctx, den, denominator_builtin, power)?;
    get_linear_coeffs(ctx, arg, var)?;
    Some(arg)
}

pub(super) fn trig_ratio_square_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = trig_ratio_square_quotient_parts(ctx, num, den, var)?;
    Some(trig_ratio_square_antiderivative_from_parts(ctx, parts, var))
}

pub(super) fn trig_tan_cot_odd_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    for (numerator_fn, denominator_fn, power, tan_owner) in [
        (BuiltinFn::Sin, BuiltinFn::Cos, 3, true),
        (BuiltinFn::Cos, BuiltinFn::Sin, 3, false),
        (BuiltinFn::Sin, BuiltinFn::Cos, 5, true),
        (BuiltinFn::Cos, BuiltinFn::Sin, 5, false),
    ] {
        if let Some(parts) =
            trig_power_quotient_parts(ctx, num, den, var, numerator_fn, denominator_fn, power)
        {
            return Some(match (tan_owner, power) {
                (true, 3) => trig_tan_third_antiderivative_from_parts(ctx, parts.arg, parts.a),
                (true, _) => trig_tan_fifth_antiderivative_from_parts(ctx, parts.arg, parts.a),
                (false, 3) => trig_cot_third_antiderivative_from_parts(ctx, parts.arg, parts.a),
                _ => trig_cot_fifth_antiderivative_from_parts(ctx, parts.arg, parts.a),
            });
        }
    }
    None
}

pub(super) fn trig_tan_fourth_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = trig_tan_fourth_quotient_parts(ctx, num, den, var)?;
    Some(trig_tan_fourth_antiderivative_from_parts(
        ctx, parts.arg, parts.a, var,
    ))
}

pub(super) fn trig_cot_fourth_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = trig_cot_fourth_quotient_parts(ctx, num, den, var)?;
    Some(trig_cot_fourth_antiderivative_from_parts(
        ctx, parts.arg, parts.a, var,
    ))
}

pub(super) fn trig_tan_sixth_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = trig_tan_sixth_quotient_parts(ctx, num, den, var)?;
    Some(trig_tan_sixth_antiderivative_from_parts(
        ctx, parts.arg, parts.a, var,
    ))
}

pub(super) fn trig_cot_sixth_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = trig_cot_sixth_quotient_parts(ctx, num, den, var)?;
    Some(trig_cot_sixth_antiderivative_from_parts(
        ctx, parts.arg, parts.a, var,
    ))
}

pub(super) fn trig_tan_eighth_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = trig_tan_eighth_quotient_parts(ctx, num, den, var)?;
    Some(trig_tan_eighth_antiderivative_from_parts(
        ctx, parts.arg, parts.a, var,
    ))
}

pub(super) fn trig_cot_eighth_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = trig_cot_eighth_quotient_parts(ctx, num, den, var)?;
    Some(trig_cot_eighth_antiderivative_from_parts(
        ctx, parts.arg, parts.a, var,
    ))
}

pub(super) fn sine_multiple_angle_ratio_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    // sin(n*d) / (c * sin(d)) = U_{n-1}(cos d) / c (Chebyshev, second
    // kind): the Dirichlet-style quotient that the pre-integration
    // rewrite produces from multiple-angle cosine products.
    let (numerator_builtin, numerator_arg) = {
        let Expr::Function(fn_id, args) = ctx.get(num).clone() else {
            return None;
        };
        if args.len() != 1 {
            return None;
        }
        (ctx.builtin_of(fn_id)?, args[0])
    };
    if !matches!(numerator_builtin, BuiltinFn::Sin) {
        return None;
    }

    let mut scale = BigRational::one();
    let mut sine_arg = None;
    for factor in mul_leaves(ctx, den) {
        if let Some(value) = rational_constant_value(ctx, factor) {
            scale *= value;
            continue;
        }
        let Expr::Function(fn_id, args) = ctx.get(factor).clone() else {
            return None;
        };
        if args.len() != 1
            || !matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Sin))
            || sine_arg.is_some()
        {
            return None;
        }
        sine_arg = Some(args[0]);
    }
    let sine_arg = sine_arg?;
    if scale.is_zero() {
        return None;
    }

    let num_poly = Polynomial::from_expr(ctx, numerator_arg, var).ok()?;
    let den_poly = Polynomial::from_expr(ctx, sine_arg, var).ok()?;
    if den_poly.degree() != 1 {
        return None;
    }
    // numerator arg must be EXACTLY n * denominator arg (offsets scale too).
    let den_lead = den_poly.coeffs.get(1)?.clone();
    if den_lead.is_zero() {
        return None;
    }
    let num_lead = num_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let ratio = num_lead / den_lead;
    if !ratio.is_integer() || ratio < BigRational::from_integer(2.into()) {
        return None;
    }
    let n = ratio.to_integer();
    if n > 8.into() {
        return None;
    }
    let scaled: Vec<BigRational> = den_poly.coeffs.iter().map(|coeff| coeff * &ratio).collect();
    if scaled != num_poly.coeffs {
        return None;
    }

    // U_{k}(cos) by recurrence over rational coefficient vectors.
    let n_usize = usize::try_from(i64::try_from(&n).ok()?).ok()?;
    let mut previous: Vec<BigRational> = vec![BigRational::one()];
    let mut current: Vec<BigRational> =
        vec![BigRational::zero(), BigRational::from_integer(2.into())];
    for _ in 2..n_usize {
        let mut next = vec![BigRational::zero(); current.len() + 1];
        for (i, c) in current.iter().enumerate() {
            next[i + 1] += BigRational::from_integer(2.into()) * c;
        }
        for (i, c) in previous.iter().enumerate() {
            next[i] -= c.clone();
        }
        previous = std::mem::replace(&mut current, next);
    }

    let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![sine_arg]);
    let mut terms = Vec::new();
    for (degree, coeff) in current.iter().enumerate() {
        if coeff.is_zero() {
            continue;
        }
        let term = match degree {
            0 => ctx.num(1),
            1 => cos_arg,
            _ => {
                let exponent = ctx.num(i64::try_from(degree).ok()?);
                ctx.add(Expr::Pow(cos_arg, exponent))
            }
        };
        terms.push(scale_rational_term(ctx, coeff.clone(), term));
    }
    let polynomial_form = build_balanced_add(ctx, &terms);
    let rebuilt = scale_rational_term(ctx, BigRational::one() / scale, polynomial_form);
    integrate_symbolic_expr(ctx, rebuilt, var)
}

pub(super) fn trig_sec_third_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Cos, 3)?;
    Some(trig_sec_third_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_csc_third_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Sin, 3)?;
    Some(trig_csc_third_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_sec_fifth_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Cos, 5)?;
    Some(trig_sec_fifth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_csc_fifth_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Sin, 5)?;
    Some(trig_csc_fifth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_sec_fourth_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Cos, 4)?;
    Some(trig_sec_fourth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_sec_sixth_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Cos, 6)?;
    Some(trig_sec_sixth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_sec_eighth_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Cos, 8)?;
    Some(trig_sec_eighth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_csc_fourth_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Sin, 4)?;
    Some(trig_csc_fourth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_csc_sixth_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Sin, 6)?;
    Some(trig_csc_sixth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_csc_eighth_quotient_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = reciprocal_trig_power_quotient_parts(ctx, num, den, var, BuiltinFn::Sin, 8)?;
    Some(trig_csc_eighth_antiderivative_from_parts(
        ctx, parts.arg, parts.a,
    ))
}

pub(super) fn trig_sine_cosine_same_affine_product_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut sin_arg = None;
    let mut cos_arg = None;

    for factor in factors {
        match ctx.get(factor) {
            Expr::Function(fn_id, args)
                if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Sin) =>
            {
                if sin_arg.replace(args[0]).is_some() {
                    return None;
                }
            }
            Expr::Function(fn_id, args)
                if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Cos) =>
            {
                if cos_arg.replace(args[0]).is_some() {
                    return None;
                }
            }
            _ => scale *= rational_constant_value(ctx, factor)?,
        }
    }

    let sin_arg = sin_arg?;
    let cos_arg = cos_arg?;
    if compare_expr(ctx, sin_arg, cos_arg) != Ordering::Equal {
        return None;
    }

    let (a, _) = get_linear_coeffs(ctx, sin_arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    let sin = ctx.call_builtin(BuiltinFn::Sin, vec![sin_arg]);
    let two = ctx.num(2);
    let sin_squared = ctx.add(Expr::Pow(sin, two));
    Some(scale_rational_term(
        ctx,
        scale / BigRational::from_integer(2.into()) / a,
        sin_squared,
    ))
}

pub(super) fn trig_power_times_derivative_antiderivative(
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
                if !matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos) {
                    scale *= rational_constant_value(ctx, factor)?;
                    continue;
                }
                let power = bounded_positive_integer_power(ctx, exp, 2, 5)?;
                if powered.replace((builtin, args[0], power)).is_some() {
                    return None;
                }
            }
            Expr::Function(fn_id, args) if args.len() == 1 => match ctx.builtin_of(fn_id) {
                Some(builtin @ (BuiltinFn::Sin | BuiltinFn::Cos)) => {
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

    let sign = match (powered_builtin, derivative_builtin) {
        (BuiltinFn::Sin, BuiltinFn::Cos) => BigRational::one(),
        (BuiltinFn::Cos, BuiltinFn::Sin) => -BigRational::one(),
        _ => return None,
    };

    let (a, _) = get_linear_coeffs(ctx, powered_arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    let base = ctx.call_builtin(powered_builtin, vec![powered_arg]);
    let next_power = ctx.num(power + 1);
    let primitive = ctx.add(Expr::Pow(base, next_power));
    let denominator = BigRational::from_integer((power + 1).into()) * a;
    Some(scale_rational_term(
        ctx,
        sign * scale / denominator,
        primitive,
    ))
}

pub(super) fn trig_fourth_power_affine_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (builtin, arg, power) = trig_power_base(ctx, expr, 4, 4)?;
    if power != 4 {
        return None;
    }

    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    let var_expr = ctx.var(var);
    let var_term = scale_rational_term(ctx, BigRational::new(3.into(), 8.into()), var_expr);

    let two = ctx.num(2);
    let four = ctx.num(4);
    let double_arg = mul2_raw(ctx, two, arg);
    let quadruple_arg = mul2_raw(ctx, four, arg);
    let sin_double = ctx.call_builtin(BuiltinFn::Sin, vec![double_arg]);
    let sin_quadruple = ctx.call_builtin(BuiltinFn::Sin, vec![quadruple_arg]);

    let double_scale = BigRational::new(1.into(), 4.into()) / a.clone();
    let signed_double_scale = match builtin {
        BuiltinFn::Sin => -double_scale,
        BuiltinFn::Cos => double_scale,
        _ => return None,
    };
    let double_term = scale_rational_term(ctx, signed_double_scale, sin_double);
    let quadruple_term = scale_rational_term(
        ctx,
        BigRational::new(1.into(), 32.into()) / a,
        sin_quadruple,
    );

    let first = ctx.add(Expr::Add(var_term, double_term));
    let primitive = ctx.add(Expr::Add(first, quadruple_term));
    Some(cas_ast::hold::wrap_hold(ctx, primitive))
}

pub(super) fn trig_sixth_power_affine_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (builtin, arg, power) = trig_power_base(ctx, expr, 6, 6)?;
    if power != 6 {
        return None;
    }

    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    let var_expr = ctx.var(var);
    let var_term = scale_rational_term(ctx, BigRational::new(5.into(), 16.into()), var_expr);

    let two = ctx.num(2);
    let four = ctx.num(4);
    let six = ctx.num(6);
    let double_arg = mul2_raw(ctx, two, arg);
    let quadruple_arg = mul2_raw(ctx, four, arg);
    let sextuple_arg = mul2_raw(ctx, six, arg);
    let sin_double = ctx.call_builtin(BuiltinFn::Sin, vec![double_arg]);
    let sin_quadruple = ctx.call_builtin(BuiltinFn::Sin, vec![quadruple_arg]);
    let sin_sextuple = ctx.call_builtin(BuiltinFn::Sin, vec![sextuple_arg]);

    let double_scale = BigRational::new(15.into(), 64.into()) / a.clone();
    let signed_double_scale = match builtin {
        BuiltinFn::Sin => -double_scale,
        BuiltinFn::Cos => double_scale,
        _ => return None,
    };
    let sextuple_scale = BigRational::new(1.into(), 192.into()) / a.clone();
    let signed_sextuple_scale = match builtin {
        BuiltinFn::Sin => -sextuple_scale,
        BuiltinFn::Cos => sextuple_scale,
        _ => return None,
    };

    let double_term = scale_rational_term(ctx, signed_double_scale, sin_double);
    let quadruple_term = scale_rational_term(
        ctx,
        BigRational::new(3.into(), 64.into()) / a,
        sin_quadruple,
    );
    let sextuple_term = scale_rational_term(ctx, signed_sextuple_scale, sin_sextuple);

    let first = ctx.add(Expr::Add(var_term, double_term));
    let second = ctx.add(Expr::Add(first, quadruple_term));
    let primitive = ctx.add(Expr::Add(second, sextuple_term));
    Some(cas_ast::hold::wrap_hold(ctx, primitive))
}

pub(super) fn trig_eighth_power_affine_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (builtin, arg, power) = trig_power_base(ctx, expr, 8, 8)?;
    if power != 8 {
        return None;
    }

    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    let var_expr = ctx.var(var);
    let var_term = scale_rational_term(ctx, BigRational::new(35.into(), 128.into()), var_expr);

    let two = ctx.num(2);
    let four = ctx.num(4);
    let six = ctx.num(6);
    let eight = ctx.num(8);
    let double_arg = mul2_raw(ctx, two, arg);
    let quadruple_arg = mul2_raw(ctx, four, arg);
    let sextuple_arg = mul2_raw(ctx, six, arg);
    let octuple_arg = mul2_raw(ctx, eight, arg);
    let sin_double = ctx.call_builtin(BuiltinFn::Sin, vec![double_arg]);
    let sin_quadruple = ctx.call_builtin(BuiltinFn::Sin, vec![quadruple_arg]);
    let sin_sextuple = ctx.call_builtin(BuiltinFn::Sin, vec![sextuple_arg]);
    let sin_octuple = ctx.call_builtin(BuiltinFn::Sin, vec![octuple_arg]);

    let double_scale = BigRational::new(7.into(), 32.into()) / a.clone();
    let signed_double_scale = match builtin {
        BuiltinFn::Sin => -double_scale,
        BuiltinFn::Cos => double_scale,
        _ => return None,
    };
    let sextuple_scale = BigRational::new(1.into(), 96.into()) / a.clone();
    let signed_sextuple_scale = match builtin {
        BuiltinFn::Sin => -sextuple_scale,
        BuiltinFn::Cos => sextuple_scale,
        _ => return None,
    };

    let double_term = scale_rational_term(ctx, signed_double_scale, sin_double);
    let quadruple_term = scale_rational_term(
        ctx,
        BigRational::new(7.into(), 128.into()) / a.clone(),
        sin_quadruple,
    );
    let sextuple_term = scale_rational_term(ctx, signed_sextuple_scale, sin_sextuple);
    let octuple_term = scale_rational_term(
        ctx,
        BigRational::new(1.into(), 1024.into()) / a,
        sin_octuple,
    );

    let first = ctx.add(Expr::Add(var_term, double_term));
    let second = ctx.add(Expr::Add(first, quadruple_term));
    let third = ctx.add(Expr::Add(second, sextuple_term));
    let primitive = ctx.add(Expr::Add(third, octuple_term));
    Some(cas_ast::hold::wrap_hold(ctx, primitive))
}

pub(super) fn trig_ratio_power_factor(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
    min_power: i64,
    max_power: i64,
) -> Option<(BigRational, ExprId, i64)> {
    let factors = mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut powered: Option<(ExprId, i64)> = None;

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
                if ctx.builtin_of(fn_id) != Some(builtin) {
                    scale *= rational_constant_value(ctx, factor)?;
                    continue;
                }

                let power = bounded_positive_integer_power(ctx, exp, min_power, max_power)?;
                if powered.replace((args[0], power)).is_some() {
                    return None;
                }
            }
            Expr::Function(fn_id, args)
                if args.len() == 1 && min_power <= 1 && ctx.builtin_of(fn_id) == Some(builtin) =>
            {
                if powered.replace((args[0], 1)).is_some() {
                    return None;
                }
            }
            _ => scale *= rational_constant_value(ctx, factor)?,
        }
    }

    let (arg, power) = powered?;
    Some((scale, arg, power))
}

pub(super) fn trig_power_base(
    ctx: &Context,
    expr: ExprId,
    min_power: i64,
    max_power: i64,
) -> Option<(BuiltinFn, ExprId, i64)> {
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };
    let (fn_id, args) = match ctx.get(base).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };
    let builtin = ctx.builtin_of(fn_id)?;
    if !matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos) {
        return None;
    }
    let power = bounded_positive_integer_power(ctx, exp, min_power, max_power)?;
    Some((builtin, args[0], power))
}

pub(super) fn trig_ratio_power_reciprocal_square_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (outer_scale, num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (BigRational::one(), *num, *den),
        Expr::Mul(l, r) if !contains_named_var(ctx, *l, var) => {
            let Expr::Div(num, den) = ctx.get(*r) else {
                return None;
            };
            (rational_constant_value(ctx, *l)?, *num, *den)
        }
        Expr::Mul(l, r) if !contains_named_var(ctx, *r, var) => {
            let Expr::Div(num, den) = ctx.get(*l) else {
                return None;
            };
            (rational_constant_value(ctx, *r)?, *num, *den)
        }
        _ => return None,
    };
    let (scale, primitive_builtin, arg, power, sign) =
        trig_ratio_power_reciprocal_square_parts(ctx, num, den)?;

    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    let base = ctx.call_builtin(primitive_builtin, vec![arg]);
    let next_power = ctx.num(power + 1);
    let primitive = ctx.add(Expr::Pow(base, next_power));
    let denominator = BigRational::from_integer((power + 1).into()) * a;
    let scale = sign * outer_scale * scale / denominator;
    let scale = BigRational::new(scale.numer().clone(), scale.denom().clone());
    Some(scale_reciprocal_integration_result_preserving_presentation(
        ctx, scale, primitive,
    ))
}

pub(super) fn trig_odd_power_affine_antiderivative(
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
    if !matches!(builtin, BuiltinFn::Sin | BuiltinFn::Cos) {
        return None;
    }

    let arg = args[0];
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let a = rational_constant_value(ctx, a)?;
    if a.is_zero() {
        return None;
    }

    let companion_builtin = match builtin {
        BuiltinFn::Sin => BuiltinFn::Cos,
        BuiltinFn::Cos => BuiltinFn::Sin,
        _ => return None,
    };
    let companion = ctx.call_builtin(companion_builtin, vec![arg]);

    let primitive = match power {
        3 => trig_cube_primitive(ctx, builtin, companion)?,
        5 => trig_fifth_primitive(ctx, builtin, companion)?,
        7 => trig_seventh_primitive(ctx, builtin, companion)?,
        _ => return None,
    };

    if a.is_one() {
        Some(primitive)
    } else if power == 7 {
        Some(scale_reciprocal_integration_result_preserving_presentation(
            ctx,
            BigRational::one() / a,
            primitive,
        ))
    } else {
        Some(scale_rational_term(ctx, BigRational::one() / a, primitive))
    }
}

/// Decompose `expr` into `scale · sin(arg)^sin_pow · cos(arg)^cos_pow`: a single shared `arg`, integer
/// trig powers (a `Div` or a negative `Pow` gives a negative power), and a rational `scale`. `None`
/// if any var-dependent factor is not such a sin/cos power, or two trig arguments disagree.
pub(super) fn extract_sin_cos_power_monomial(
    ctx: &Context,
    expr: ExprId,
    _var: &str,
) -> Option<(BigRational, ExprId, i64, i64)> {
    fn collect(
        ctx: &Context,
        e: ExprId,
        mult: i64,
        scale: &mut BigRational,
        sin_pow: &mut i64,
        cos_pow: &mut i64,
        arg: &mut Option<ExprId>,
    ) -> Option<()> {
        match ctx.get(e) {
            Expr::Mul(a, b) => {
                collect(ctx, *a, mult, scale, sin_pow, cos_pow, arg)?;
                collect(ctx, *b, mult, scale, sin_pow, cos_pow, arg)
            }
            Expr::Div(a, b) => {
                collect(ctx, *a, mult, scale, sin_pow, cos_pow, arg)?;
                collect(ctx, *b, -mult, scale, sin_pow, cos_pow, arg)
            }
            Expr::Neg(inner) => {
                if mult % 2 != 0 {
                    *scale *= -BigRational::one();
                }
                collect(ctx, *inner, mult, scale, sin_pow, cos_pow, arg)
            }
            Expr::Pow(base, exp) => {
                let e_rat = rational_constant_value(ctx, *exp)?;
                if !e_rat.is_integer() {
                    return None;
                }
                let e_int: i64 = e_rat.to_integer().try_into().ok()?;
                collect(ctx, *base, mult * e_int, scale, sin_pow, cos_pow, arg)
            }
            Expr::Function(fn_id, args) if args.len() == 1 => {
                let is_cos = match ctx.builtin_of(*fn_id) {
                    Some(BuiltinFn::Sin) => false,
                    Some(BuiltinFn::Cos) => true,
                    _ => return None,
                };
                match arg {
                    None => *arg = Some(args[0]),
                    Some(existing) => {
                        if compare_expr(ctx, *existing, args[0]) != Ordering::Equal {
                            return None;
                        }
                    }
                }
                if is_cos {
                    *cos_pow += mult;
                } else {
                    *sin_pow += mult;
                }
                Some(())
            }
            _ => {
                let r = rational_constant_value(ctx, e)?;
                if r.is_zero() {
                    return None;
                }
                *scale *= int_pow_rational(&r, mult);
                Some(())
            }
        }
    }
    let mut scale = BigRational::one();
    let mut sin_pow = 0i64;
    let mut cos_pow = 0i64;
    let mut arg = None;
    collect(
        ctx,
        expr,
        1,
        &mut scale,
        &mut sin_pow,
        &mut cos_pow,
        &mut arg,
    )?;
    arg.map(|a| (scale, a, sin_pow, cos_pow))
}

fn trig_cube_primitive(ctx: &mut Context, builtin: BuiltinFn, companion: ExprId) -> Option<ExprId> {
    let three = ctx.num(3);
    let companion_cubed = ctx.add(Expr::Pow(companion, three));
    let cubic_term =
        scale_rational_term(ctx, BigRational::new(1.into(), 3.into()), companion_cubed);

    match builtin {
        BuiltinFn::Sin => {
            let neg_companion = ctx.add(Expr::Neg(companion));
            Some(ctx.add(Expr::Add(neg_companion, cubic_term)))
        }
        BuiltinFn::Cos => Some(ctx.add(Expr::Sub(companion, cubic_term))),
        _ => None,
    }
}

fn trig_fifth_primitive(
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
        BuiltinFn::Sin => {
            let neg_companion = ctx.add(Expr::Neg(companion));
            let neg_fifth = ctx.add(Expr::Neg(fifth_term));
            let first_two = ctx.add(Expr::Add(neg_companion, cubic_term));
            Some(ctx.add(Expr::Add(first_two, neg_fifth)))
        }
        BuiltinFn::Cos => {
            let neg_cubic = ctx.add(Expr::Neg(cubic_term));
            let first_two = ctx.add(Expr::Add(companion, neg_cubic));
            Some(ctx.add(Expr::Add(first_two, fifth_term)))
        }
        _ => None,
    }
}

fn trig_seventh_primitive(
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
        BuiltinFn::Sin => {
            let neg_companion = ctx.add(Expr::Neg(companion));
            let neg_fifth = ctx.add(Expr::Neg(fifth_term));
            let first_two = ctx.add(Expr::Add(neg_companion, companion_cubed));
            let first_three = ctx.add(Expr::Add(first_two, neg_fifth));
            Some(ctx.add(Expr::Add(first_three, seventh_term)))
        }
        BuiltinFn::Cos => {
            let neg_cubed = ctx.add(Expr::Neg(companion_cubed));
            let neg_seventh = ctx.add(Expr::Neg(seventh_term));
            let first_two = ctx.add(Expr::Add(companion, neg_cubed));
            let first_three = ctx.add(Expr::Add(first_two, fifth_term));
            Some(ctx.add(Expr::Add(first_three, neg_seventh)))
        }
        _ => None,
    }
}

pub(super) fn polynomial_reciprocal_trig_square_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let numerator = Polynomial::from_expr(ctx, num, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&numerator, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    let integral = match builtin {
        BuiltinFn::Cos => ctx.call_builtin(BuiltinFn::Tan, vec![arg]),
        BuiltinFn::Sin => {
            let cot_arg = ctx.call_builtin(BuiltinFn::Cot, vec![arg]);
            ctx.add(Expr::Neg(cot_arg))
        }
        _ => return None,
    };

    if matches!(builtin, BuiltinFn::Cos | BuiltinFn::Sin) {
        Some(scale_reciprocal_integration_result_preserving_presentation(
            ctx, scale, integral,
        ))
    } else {
        Some(scale_reciprocal_integration_result(ctx, scale, integral))
    }
}

pub(super) fn trig_log_derivative_ratio_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (_, _, scaled) = trig_log_derivative_ratio_scale(ctx, num, den, var)?;
    let log_arg = cas_ast::hold::wrap_hold(ctx, den);
    let log_abs = ln_abs(ctx, log_arg);
    Some(scale_expr_reciprocal_integration_result(
        ctx, scaled, log_abs,
    ))
}

pub(super) fn trig_log_derivative_ratio_scale(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<(BuiltinFn, ExprId, ExprId)> {
    let (den_builtin, arg) = match ctx.get(den) {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(*fn_id)?, args[0]),
        _ => return None,
    };
    let numerator_builtin = trig_log_derivative_numerator_builtin(den_builtin)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let cofactor = trig_log_derivative_ratio_cofactor(ctx, num, numerator_builtin, arg)?;
    let scale = symbolic_linear_cofactor_scale_expr(ctx, cofactor, arg, var)?;

    let scaled = match den_builtin {
        BuiltinFn::Cos => negate_scalar_expr(ctx, scale),
        BuiltinFn::Sin => scale,
        _ => return None,
    };
    Some((den_builtin, arg, scaled))
}

fn trig_log_derivative_ratio_cofactor(
    ctx: &mut Context,
    num: ExprId,
    numerator_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    additive_cofactor_from_term_cofactors(ctx, num, |ctx, term| {
        trig_log_derivative_ratio_term_cofactor(ctx, term, numerator_builtin, arg)
    })
}

fn trig_log_derivative_ratio_term_cofactor(
    ctx: &mut Context,
    term: ExprId,
    numerator_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    product_cofactor_excluding_unary_builtin_arg(
        ctx,
        term,
        numerator_builtin,
        |ctx, numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal,
    )
}

pub(super) fn trig_log_derivative_ratio_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    let (den_builtin, arg, _) = trig_log_derivative_ratio_scale(ctx, num, den, var)?;
    build_reciprocal_trig_denominator_nonzero_condition(ctx, den_builtin, arg)
}

fn nested_trig_log_derivative_log_factor(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, ExprId)> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    if args.len() != 1 || ctx.builtin_of(*fn_id) != Some(BuiltinFn::Ln) {
        return None;
    }

    let log_arg = args[0];
    match ctx.get(log_arg) {
        Expr::Function(inner_fn_id, inner_args) if inner_args.len() == 1 => {
            let builtin = ctx.builtin_of(*inner_fn_id)?;
            match builtin {
                BuiltinFn::Tan | BuiltinFn::Cot => Some((builtin, inner_args[0], expr)),
                _ => None,
            }
        }
        Expr::Div(num, den) => {
            if let (Some(sin_arg), Some(cos_arg)) = (
                unary_builtin_arg(ctx, *num, BuiltinFn::Sin),
                unary_builtin_arg(ctx, *den, BuiltinFn::Cos),
            ) {
                if compare_expr(ctx, sin_arg, cos_arg) == Ordering::Equal {
                    return Some((BuiltinFn::Tan, sin_arg, expr));
                }
            }

            if let (Some(cos_arg), Some(sin_arg)) = (
                unary_builtin_arg(ctx, *num, BuiltinFn::Cos),
                unary_builtin_arg(ctx, *den, BuiltinFn::Sin),
            ) {
                if compare_expr(ctx, cos_arg, sin_arg) == Ordering::Equal {
                    return Some((BuiltinFn::Cot, cos_arg, expr));
                }
            }

            None
        }
        _ => None,
    }
}

fn indexed_nested_trig_log_derivative_log_factor(
    ctx: &Context,
    factors: &[ExprId],
) -> Option<(usize, BuiltinFn, ExprId, ExprId)> {
    factors.iter().enumerate().find_map(|(idx, factor)| {
        nested_trig_log_derivative_log_factor(ctx, *factor)
            .map(|(builtin, arg, log_arg)| (idx, builtin, arg, log_arg))
    })
}

pub(super) fn nested_trig_log_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator_factors = mul_leaves(ctx, num);
    let denominator_factors = mul_leaves(ctx, den);
    let (log_index, log_builtin, arg, log_arg) =
        indexed_nested_trig_log_derivative_log_factor(ctx, &denominator_factors)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let sin_index = indexed_matching_unary_builtin_factor(
        ctx,
        &denominator_factors,
        BuiltinFn::Sin,
        arg,
        &[log_index],
    )?;
    let cos_index = indexed_matching_unary_builtin_factor(
        ctx,
        &denominator_factors,
        BuiltinFn::Cos,
        arg,
        &[log_index, sin_index],
    )?;

    let remaining_denominator: Vec<_> = denominator_factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| {
            (![log_index, sin_index, cos_index].contains(&idx)).then_some(*factor)
        })
        .collect();

    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let mut scale = quotient_scale_against_polynomial(
        ctx,
        &numerator_factors,
        &remaining_denominator,
        &derivative,
        var,
    )?;
    if scale.is_zero() {
        return None;
    }
    if log_builtin == BuiltinFn::Cot {
        scale = -scale;
    }

    let log_abs = ln_abs(ctx, log_arg);
    Some(scale_rational_term(ctx, scale, log_abs))
}

pub fn integrate_symbolic_is_affine_trig_seventh_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return false;
    };
    is_number(ctx, exp, 7) && trig_odd_power_affine_antiderivative(ctx, base, exp, var).is_some()
}

pub fn integrate_symbolic_is_tan_fourth_affine_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => {
            trig_tan_fourth_affine_antiderivative(ctx, base, exp, var).is_some()
        }
        Expr::Div(num, den) => {
            trig_tan_fourth_quotient_antiderivative(ctx, num, den, var).is_some()
        }
        _ => false,
    }
}

pub fn integrate_symbolic_is_cot_fourth_affine_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => {
            trig_cot_fourth_affine_antiderivative(ctx, base, exp, var).is_some()
        }
        Expr::Div(num, den) => {
            trig_cot_fourth_quotient_antiderivative(ctx, num, den, var).is_some()
        }
        _ => false,
    }
}

pub fn integrate_symbolic_is_tan_sixth_affine_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => trig_tan_sixth_affine_antiderivative(ctx, base, exp, var).is_some(),
        Expr::Div(num, den) => trig_tan_sixth_quotient_antiderivative(ctx, num, den, var).is_some(),
        _ => false,
    }
}

pub fn integrate_symbolic_is_cot_sixth_affine_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => trig_cot_sixth_affine_antiderivative(ctx, base, exp, var).is_some(),
        Expr::Div(num, den) => trig_cot_sixth_quotient_antiderivative(ctx, num, den, var).is_some(),
        _ => false,
    }
}

pub fn integrate_symbolic_is_tan_eighth_affine_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => {
            trig_tan_eighth_affine_antiderivative(ctx, base, exp, var).is_some()
        }
        Expr::Div(num, den) => {
            trig_tan_eighth_quotient_antiderivative(ctx, num, den, var).is_some()
        }
        _ => false,
    }
}

pub fn integrate_symbolic_is_cot_eighth_affine_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => {
            trig_cot_eighth_affine_antiderivative(ctx, base, exp, var).is_some()
        }
        Expr::Div(num, den) => {
            trig_cot_eighth_quotient_antiderivative(ctx, num, den, var).is_some()
        }
        _ => false,
    }
}

pub fn integrate_symbolic_is_sec_fourth_affine_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => {
            trig_sec_fourth_affine_antiderivative(ctx, base, exp, var).is_some()
        }
        Expr::Div(num, den) => {
            trig_sec_fourth_quotient_antiderivative(ctx, num, den, var).is_some()
        }
        _ => false,
    }
}

pub fn integrate_symbolic_is_csc_fourth_affine_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => {
            trig_csc_fourth_affine_antiderivative(ctx, base, exp, var).is_some()
        }
        Expr::Div(num, den) => {
            trig_csc_fourth_quotient_antiderivative(ctx, num, den, var).is_some()
        }
        _ => false,
    }
}

pub fn integrate_symbolic_is_sec_sixth_affine_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => trig_sec_sixth_affine_antiderivative(ctx, base, exp, var).is_some(),
        Expr::Div(num, den) => trig_sec_sixth_quotient_antiderivative(ctx, num, den, var).is_some(),
        _ => false,
    }
}

pub fn integrate_symbolic_is_csc_sixth_affine_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => trig_csc_sixth_affine_antiderivative(ctx, base, exp, var).is_some(),
        Expr::Div(num, den) => trig_csc_sixth_quotient_antiderivative(ctx, num, den, var).is_some(),
        _ => false,
    }
}

pub fn integrate_symbolic_is_sec_eighth_affine_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => {
            trig_sec_eighth_affine_antiderivative(ctx, base, exp, var).is_some()
        }
        Expr::Div(num, den) => {
            trig_sec_eighth_quotient_antiderivative(ctx, num, den, var).is_some()
        }
        _ => false,
    }
}

pub fn integrate_symbolic_is_csc_eighth_affine_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Pow(base, exp) => {
            trig_csc_eighth_affine_antiderivative(ctx, base, exp, var).is_some()
        }
        Expr::Div(num, den) => {
            trig_csc_eighth_quotient_antiderivative(ctx, num, den, var).is_some()
        }
        _ => false,
    }
}

pub(super) fn reciprocal_trig_square_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        _ => return None,
    };
    if !is_number(ctx, num, 1) {
        return None;
    }

    let (builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    get_linear_coeffs(ctx, arg, var)?;
    build_reciprocal_trig_denominator_nonzero_condition(ctx, builtin, arg)
}

pub(super) fn sec_fourth_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) if is_number(ctx, num, 1) => {
            let arg = reciprocal_trig_power_quotient_arg(ctx, num, den, var, BuiltinFn::Cos, 4)?;
            Some(ctx.call_builtin(BuiltinFn::Cos, vec![arg]))
        }
        Expr::Pow(base, exp) if is_number(ctx, exp, 4) => {
            let arg = reciprocal_trig_power_affine_arg(ctx, base, var, BuiltinFn::Sec)?;
            Some(ctx.call_builtin(BuiltinFn::Cos, vec![arg]))
        }
        _ => None,
    }
}

pub(super) fn csc_fourth_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) if is_number(ctx, num, 1) => {
            let arg = reciprocal_trig_power_quotient_arg(ctx, num, den, var, BuiltinFn::Sin, 4)?;
            Some(ctx.call_builtin(BuiltinFn::Sin, vec![arg]))
        }
        Expr::Pow(base, exp) if is_number(ctx, exp, 4) => {
            let arg = reciprocal_trig_power_affine_arg(ctx, base, var, BuiltinFn::Csc)?;
            Some(ctx.call_builtin(BuiltinFn::Sin, vec![arg]))
        }
        _ => None,
    }
}

pub(super) fn sec_sixth_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) if is_number(ctx, num, 1) => {
            let arg = reciprocal_trig_power_quotient_arg(ctx, num, den, var, BuiltinFn::Cos, 6)?;
            Some(ctx.call_builtin(BuiltinFn::Cos, vec![arg]))
        }
        Expr::Pow(base, exp) if is_number(ctx, exp, 6) => {
            let arg = reciprocal_trig_power_affine_arg(ctx, base, var, BuiltinFn::Sec)?;
            Some(ctx.call_builtin(BuiltinFn::Cos, vec![arg]))
        }
        _ => None,
    }
}

pub(super) fn csc_sixth_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) if is_number(ctx, num, 1) => {
            let arg = reciprocal_trig_power_quotient_arg(ctx, num, den, var, BuiltinFn::Sin, 6)?;
            Some(ctx.call_builtin(BuiltinFn::Sin, vec![arg]))
        }
        Expr::Pow(base, exp) if is_number(ctx, exp, 6) => {
            let arg = reciprocal_trig_power_affine_arg(ctx, base, var, BuiltinFn::Csc)?;
            Some(ctx.call_builtin(BuiltinFn::Sin, vec![arg]))
        }
        _ => None,
    }
}

pub(super) fn sec_eighth_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) if is_number(ctx, num, 1) => {
            let arg = reciprocal_trig_power_quotient_arg(ctx, num, den, var, BuiltinFn::Cos, 8)?;
            Some(ctx.call_builtin(BuiltinFn::Cos, vec![arg]))
        }
        Expr::Pow(base, exp) if is_number(ctx, exp, 8) => {
            let arg = reciprocal_trig_power_affine_arg(ctx, base, var, BuiltinFn::Sec)?;
            Some(ctx.call_builtin(BuiltinFn::Cos, vec![arg]))
        }
        _ => None,
    }
}

pub(super) fn csc_eighth_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) if is_number(ctx, num, 1) => {
            let arg = reciprocal_trig_power_quotient_arg(ctx, num, den, var, BuiltinFn::Sin, 8)?;
            Some(ctx.call_builtin(BuiltinFn::Sin, vec![arg]))
        }
        Expr::Pow(base, exp) if is_number(ctx, exp, 8) => {
            let arg = reciprocal_trig_power_affine_arg(ctx, base, var, BuiltinFn::Csc)?;
            Some(ctx.call_builtin(BuiltinFn::Sin, vec![arg]))
        }
        _ => None,
    }
}

pub(super) fn polynomial_reciprocal_trig_square_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            polynomial_reciprocal_trig_square_required_nonzero_from_parts(ctx, num, den, var)
        }
        Expr::Mul(left, right) => {
            if let Expr::Div(num, den) = ctx.get(right).clone() {
                let combined_num = mul2_raw(ctx, left, num);
                if let Some(condition) =
                    polynomial_reciprocal_trig_square_required_nonzero_from_parts(
                        ctx,
                        combined_num,
                        den,
                        var,
                    )
                {
                    return Some(condition);
                }
            }

            if let Expr::Div(num, den) = ctx.get(left).clone() {
                let combined_num = mul2_raw(ctx, right, num);
                return polynomial_reciprocal_trig_square_required_nonzero_from_parts(
                    ctx,
                    combined_num,
                    den,
                    var,
                );
            }

            None
        }
        _ => None,
    }
}

pub(super) fn trig_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let policy = reciprocal_trig_derivative_policy(den_builtin)?;
    let numerator_arg = unary_builtin_arg(ctx, num, policy.numerator_builtin())?;
    if compare_expr(ctx, numerator_arg, arg) != Ordering::Equal {
        return None;
    }

    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let integral = build_reciprocal_trig_derivative_integral(ctx, policy, arg);
    let result = divide_by_coeff_unless_one_preserving_presentation(ctx, integral, a);
    Some(result)
}

fn sqrt_compact_reciprocal_trig_antiderivative(
    ctx: &mut Context,
    den_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    let radicand = sqrt_like_radicand(ctx, arg)?;
    let sqrt_arg = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    reciprocal_trig_derivative_base_antiderivative(ctx, den_builtin, sqrt_arg)
}

fn polynomial_trig_reciprocal_derivative_term_cofactor(
    ctx: &mut Context,
    term: ExprId,
    numerator_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    unique_product_cofactor_excluding_unary_builtin_arg(
        ctx,
        term,
        numerator_builtin,
        |ctx, numerator_arg| compare_expr(ctx, numerator_arg, arg) == Ordering::Equal,
    )
}

pub(super) fn polynomial_trig_reciprocal_derivative_cofactor(
    ctx: &mut Context,
    num: ExprId,
    numerator_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    additive_cofactor_from_term_cofactors(ctx, num, |ctx, term| {
        polynomial_trig_reciprocal_derivative_term_cofactor(ctx, term, numerator_builtin, arg)
    })
}

pub(super) fn trig_reciprocal_derivative_cofactor_scale(
    ctx: &mut Context,
    cofactor: ExprId,
    arg: ExprId,
    var: &str,
) -> Option<(BigRational, bool)> {
    polynomial_or_symbolic_linear_cofactor_scale(ctx, cofactor, arg, var)
}

pub(super) fn trig_reciprocal_derivative_base_integral(
    ctx: &mut Context,
    den_builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    if let Some(compact) = sqrt_compact_reciprocal_trig_antiderivative(ctx, den_builtin, arg) {
        return Some(compact);
    }

    reciprocal_trig_derivative_base_antiderivative(ctx, den_builtin, arg)
}

pub(super) fn trig_reciprocal_derivative_cofactor_is_nonzero(
    ctx: &mut Context,
    cofactor: ExprId,
    arg: ExprId,
    var: &str,
) -> Option<()> {
    if let Some((scale, _)) = trig_reciprocal_derivative_cofactor_scale(ctx, cofactor, arg, var) {
        return (!scale.is_zero()).then_some(());
    }

    let scale = symbolic_linear_cofactor_scale_expr(ctx, cofactor, arg, var)?;
    (!is_number(ctx, scale, 0)).then_some(())
}

pub(super) fn polynomial_trig_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, arg, cofactor) =
        polynomial_trig_reciprocal_derivative_parts(ctx, num, den, var)?;
    polynomial_trig_reciprocal_derivative_antiderivative_from_parts(
        ctx,
        den_builtin,
        arg,
        cofactor,
        var,
    )
}

fn polynomial_trig_reciprocal_derivative_antiderivative_with_required_nonzero(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    let (den_builtin, arg, cofactor) =
        polynomial_trig_reciprocal_derivative_parts(ctx, num, den, var)?;
    let integral = polynomial_trig_reciprocal_derivative_antiderivative_from_parts(
        ctx,
        den_builtin,
        arg,
        cofactor,
        var,
    )?;
    trig_reciprocal_derivative_cofactor_is_nonzero(ctx, cofactor, arg, var)?;
    let required_nonzero =
        build_reciprocal_trig_denominator_nonzero_condition(ctx, den_builtin, arg)?;
    Some((integral, required_nonzero))
}

pub fn integrate_symbolic_polynomial_trig_reciprocal_derivative_root_gate(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(ExprId, ExprId)> {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            polynomial_trig_reciprocal_derivative_antiderivative_with_required_nonzero(
                ctx, num, den, var,
            )
        }
        Expr::Mul(left, right) => {
            if let Expr::Div(num, den) = ctx.get(right).clone() {
                let combined_num = mul2_raw(ctx, left, num);
                if let Some(out) =
                    polynomial_trig_reciprocal_derivative_antiderivative_with_required_nonzero(
                        ctx,
                        combined_num,
                        den,
                        var,
                    )
                {
                    return Some(out);
                }
            }

            if let Expr::Div(num, den) = ctx.get(left).clone() {
                let combined_num = mul2_raw(ctx, right, num);
                return polynomial_trig_reciprocal_derivative_antiderivative_with_required_nonzero(
                    ctx,
                    combined_num,
                    den,
                    var,
                );
            }

            None
        }
        _ => None,
    }
}

pub(super) fn sqrt_trig_reciprocal_derivative_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_trig_reciprocal_derivative_parts(ctx, expr, var)?;
    let preserve_symbolic_scale_presentation = !matches!(ctx.get(parts.scale), Expr::Number(_))
        && sqrt_like_radicand(ctx, parts.arg).is_some();
    let integral =
        reciprocal_trig_derivative_base_antiderivative(ctx, parts.denominator_builtin, parts.arg)?;
    Some(
        scale_expr_reciprocal_integration_result_preserving_presentation(
            ctx,
            parts.scale,
            integral,
            preserve_symbolic_scale_presentation,
        ),
    )
}

pub fn integrate_symbolic_is_sqrt_trig_reciprocal_derivative_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    sqrt_trig_reciprocal_derivative_parts(ctx, expr, var).is_some()
}

pub(super) fn sqrt_trig_log_derivative_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_trig_log_derivative_parts(ctx, expr, var)?;
    let den_arg = ctx.call_builtin(parts.denominator_builtin, vec![parts.arg]);
    let log_abs_den = ln_abs(ctx, den_arg);
    let integral = match parts.denominator_builtin {
        BuiltinFn::Cos => ctx.add(Expr::Neg(log_abs_den)),
        BuiltinFn::Sin => log_abs_den,
        _ => return None,
    };
    Some(scale_rational_term(ctx, parts.scale, integral))
}

pub fn integrate_symbolic_is_sqrt_trig_log_derivative_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    sqrt_trig_log_derivative_parts(ctx, expr, var).is_some()
}

pub(super) fn sqrt_reciprocal_trig_log_derivative_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_reciprocal_trig_log_derivative_parts(ctx, expr, var)?;
    let primitive_builtin = match parts.denominator_builtin {
        BuiltinFn::Cos => BuiltinFn::Sec,
        BuiltinFn::Sin => BuiltinFn::Csc,
        _ => return None,
    };
    let one = ctx.num(1);
    let primitive = sec_csc_log_antiderivative(ctx, primitive_builtin, parts.arg, one)?;
    Some(scale_rational_term(ctx, parts.scale, primitive))
}

pub fn integrate_symbolic_is_sqrt_reciprocal_trig_log_derivative_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    sqrt_reciprocal_trig_log_derivative_parts(ctx, expr, var).is_some()
}

pub(super) fn polynomial_trig_reciprocal_factor_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (den_builtin, arg, scale) =
        polynomial_trig_reciprocal_factor_derivative_parts(ctx, num, den, var)?;
    let integral = trig_reciprocal_derivative_base_integral(ctx, den_builtin, arg)?;
    if scale.is_one() {
        return Some(integral);
    }
    if scale == -BigRational::one() {
        return Some(negate_integration_result(ctx, integral));
    }

    Some(scale_reciprocal_integration_result_preserving_presentation(
        ctx, scale, integral,
    ))
}

pub(super) fn constant_scaled_trig_reciprocal_derivative_antiderivative(
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
    polynomial_trig_reciprocal_derivative_antiderivative(ctx, scaled_num, den, var)
}

pub(super) fn trig_reciprocal_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    let (den_builtin, arg) = reciprocal_trig_square_parts(ctx, den)?;
    let policy = reciprocal_trig_derivative_policy(den_builtin)?;
    let numerator_arg = unary_builtin_arg(ctx, num, policy.numerator_builtin())?;
    if compare_expr(ctx, numerator_arg, arg) != Ordering::Equal {
        return None;
    }
    get_linear_coeffs(ctx, arg, var)?;
    build_reciprocal_trig_denominator_nonzero_condition(ctx, den_builtin, arg)
}

pub(super) fn polynomial_trig_reciprocal_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    polynomial_trig_reciprocal_derivative_required_nonzero_from_parts(ctx, num, den, var)
}

pub(super) fn sqrt_trig_reciprocal_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_trig_reciprocal_derivative_parts(ctx, expr, var)?;
    build_reciprocal_trig_denominator_nonzero_condition(ctx, parts.denominator_builtin, parts.arg)
}

pub(super) fn sqrt_trig_log_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_trig_log_derivative_parts(ctx, expr, var)?;
    build_reciprocal_trig_denominator_nonzero_condition(ctx, parts.denominator_builtin, parts.arg)
}

pub(super) fn sqrt_reciprocal_trig_log_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_reciprocal_trig_log_derivative_parts(ctx, expr, var)?;
    let arg = ctx.call_builtin(BuiltinFn::Sqrt, vec![parts.radicand]);
    build_reciprocal_trig_denominator_nonzero_condition(ctx, parts.denominator_builtin, arg)
}

pub(super) fn polynomial_trig_reciprocal_factor_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    let (den_builtin, arg, _) =
        polynomial_trig_reciprocal_factor_derivative_parts(ctx, num, den, var)?;

    build_reciprocal_trig_denominator_nonzero_condition(ctx, den_builtin, arg)
}

pub(super) fn constant_scaled_trig_reciprocal_derivative_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (constant, expr) = match ctx.get(expr).clone() {
        Expr::Mul(l, r) if !contains_named_var(ctx, l, var) => (l, r),
        Expr::Mul(l, r) if !contains_named_var(ctx, r, var) => (r, l),
        _ => return None,
    };
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };
    let scaled_num = mul2_raw(ctx, constant, num);
    polynomial_trig_reciprocal_derivative_required_nonzero_from_parts(ctx, scaled_num, den, var)
}

pub(super) fn trig_ratio_power_reciprocal_square_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let div_parts = match ctx.get(expr).clone() {
        Expr::Div(num, den) => Some((num, den)),
        Expr::Mul(l, r) if !contains_named_var(ctx, l, var) => match ctx.get(r).clone() {
            Expr::Div(num, den) => Some((num, den)),
            _ => None,
        },
        Expr::Mul(l, r) if !contains_named_var(ctx, r, var) => match ctx.get(l).clone() {
            Expr::Div(num, den) => Some((num, den)),
            _ => None,
        },
        _ => None,
    };
    if let Some((num, den)) = div_parts {
        let (_, primitive_builtin, arg, _, _) =
            trig_ratio_power_reciprocal_square_parts(ctx, num, den)?;
        let den_builtin = match primitive_builtin {
            BuiltinFn::Tan => BuiltinFn::Cos,
            BuiltinFn::Cot => BuiltinFn::Sin,
            _ => return None,
        };
        get_linear_coeffs(ctx, arg, var)?;
        return build_reciprocal_trig_denominator_nonzero_condition(ctx, den_builtin, arg);
    }

    let factors = mul_leaves(ctx, expr);
    let mut reciprocal_square: Option<(BuiltinFn, ExprId)> = None;
    let mut ratio_power: Option<(BuiltinFn, ExprId)> = None;

    for factor in factors {
        match ctx.get(factor).clone() {
            Expr::Pow(base, exp) => {
                let (fn_id, args) = match ctx.get(base).clone() {
                    Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
                    _ => {
                        rational_constant_value(ctx, factor)?;
                        continue;
                    }
                };
                match ctx.builtin_of(fn_id)? {
                    BuiltinFn::Sec if is_number(ctx, exp, 2) => {
                        if reciprocal_square
                            .replace((BuiltinFn::Cos, args[0]))
                            .is_some()
                        {
                            return None;
                        }
                    }
                    BuiltinFn::Csc if is_number(ctx, exp, 2) => {
                        if reciprocal_square
                            .replace((BuiltinFn::Sin, args[0]))
                            .is_some()
                        {
                            return None;
                        }
                    }
                    BuiltinFn::Tan | BuiltinFn::Cot => {
                        bounded_positive_integer_power(ctx, exp, 1, 5)?;
                        let ratio_builtin = ctx.builtin_of(fn_id)?;
                        if ratio_power.replace((ratio_builtin, args[0])).is_some() {
                            return None;
                        }
                    }
                    _ => {
                        rational_constant_value(ctx, factor)?;
                    }
                }
            }
            Expr::Function(fn_id, args) if args.len() == 1 => match ctx.builtin_of(fn_id)? {
                BuiltinFn::Tan | BuiltinFn::Cot => {
                    let ratio_builtin = ctx.builtin_of(fn_id)?;
                    if ratio_power.replace((ratio_builtin, args[0])).is_some() {
                        return None;
                    }
                }
                _ => {
                    rational_constant_value(ctx, factor)?;
                }
            },
            _ => {
                rational_constant_value(ctx, factor)?;
            }
        }
    }

    let (den_builtin, den_arg) = reciprocal_square?;
    let (ratio_builtin, ratio_arg) = ratio_power?;
    match (den_builtin, ratio_builtin) {
        (BuiltinFn::Cos, BuiltinFn::Tan) | (BuiltinFn::Sin, BuiltinFn::Cot) => {}
        _ => return None,
    }
    if compare_expr(ctx, den_arg, ratio_arg) != Ordering::Equal {
        return None;
    }
    get_linear_coeffs(ctx, den_arg, var)?;
    build_reciprocal_trig_denominator_nonzero_condition(ctx, den_builtin, den_arg)
}

pub(super) fn polynomial_sec_csc_square_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let mut reciprocal_square: Option<(BuiltinFn, ExprId)> = None;
    let mut cofactor_factors = Vec::new();

    for factor in factors {
        let reciprocal_square_part = match ctx.get(factor).clone() {
            Expr::Pow(base, exp) if is_number(ctx, exp, 2) => {
                let (fn_id, args) = match ctx.get(base).clone() {
                    Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
                    _ => {
                        cofactor_factors.push(factor);
                        continue;
                    }
                };
                match ctx.builtin_of(fn_id)? {
                    BuiltinFn::Sec => Some((BuiltinFn::Cos, args[0])),
                    BuiltinFn::Csc => Some((BuiltinFn::Sin, args[0])),
                    _ => None,
                }
            }
            _ => None,
        };

        if let Some(part) = reciprocal_square_part {
            if reciprocal_square.replace(part).is_some() {
                return None;
            }
        } else {
            cofactor_factors.push(factor);
        }
    }

    let (required_builtin, arg) = reciprocal_square?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let cofactor = if cofactor_factors.is_empty() {
        ctx.num(1)
    } else {
        build_balanced_mul(ctx, &cofactor_factors)
    };
    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    build_reciprocal_trig_denominator_nonzero_condition(ctx, required_builtin, arg)
}

pub(super) fn trig_ratio_square_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        if let Some(parts) = trig_ratio_square_quotient_parts(ctx, num, den, var) {
            return build_trig_pole_nonzero_condition(ctx, parts.builtin, parts.arg);
        }

        if let Some(parts) = trig_tan_fourth_quotient_parts(ctx, num, den, var) {
            return build_trig_pole_nonzero_condition(ctx, BuiltinFn::Tan, parts.arg);
        }

        if let Some(parts) = trig_cot_fourth_quotient_parts(ctx, num, den, var) {
            return build_trig_pole_nonzero_condition(ctx, BuiltinFn::Cot, parts.arg);
        }

        if let Some(parts) = trig_tan_sixth_quotient_parts(ctx, num, den, var) {
            return build_trig_pole_nonzero_condition(ctx, BuiltinFn::Tan, parts.arg);
        }

        if let Some(parts) = trig_cot_sixth_quotient_parts(ctx, num, den, var) {
            return build_trig_pole_nonzero_condition(ctx, BuiltinFn::Cot, parts.arg);
        }

        if let Some(parts) = trig_tan_eighth_quotient_parts(ctx, num, den, var) {
            return build_trig_pole_nonzero_condition(ctx, BuiltinFn::Tan, parts.arg);
        }

        let parts = trig_cot_eighth_quotient_parts(ctx, num, den, var)?;
        return build_trig_pole_nonzero_condition(ctx, BuiltinFn::Cot, parts.arg);
    }

    let Expr::Pow(base, exp) = ctx.get(expr).clone() else {
        return None;
    };

    let (fn_id, args) = match ctx.get(base).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (fn_id, args),
        _ => return None,
    };
    let builtin = ctx.builtin_of(fn_id)?;
    match builtin {
        BuiltinFn::Tan | BuiltinFn::Cot
            if is_number(ctx, exp, 2)
                || is_number(ctx, exp, 4)
                || is_number(ctx, exp, 6)
                || is_number(ctx, exp, 8) => {}
        _ => return None,
    }
    let arg = args[0];
    get_linear_coeffs(ctx, arg, var)?;
    build_trig_pole_nonzero_condition(ctx, builtin, arg)
}

pub(super) fn trig_log_antiderivative(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    let log_arg_builtin = match builtin {
        BuiltinFn::Tan => BuiltinFn::Cos,
        BuiltinFn::Cot => BuiltinFn::Sin,
        BuiltinFn::Sec | BuiltinFn::Csc => {
            return sec_csc_log_antiderivative(ctx, builtin, arg, a);
        }
        _ => return None,
    };
    let log_arg = ctx.call_builtin(log_arg_builtin, vec![arg]);
    let log_abs = ln_abs(ctx, log_arg);
    let integral = match builtin {
        BuiltinFn::Tan => ctx.add(Expr::Neg(log_abs)),
        BuiltinFn::Cot => log_abs,
        _ => return None,
    };

    let is_a_one = if let Expr::Number(n) = ctx.get(a) {
        n.is_one()
    } else {
        false
    };
    if is_a_one {
        Some(integral)
    } else {
        Some(ctx.add(Expr::Div(integral, a)))
    }
}

pub(super) fn sec_csc_log_antiderivative(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
    coeff: ExprId,
) -> Option<ExprId> {
    let log_arg = build_reciprocal_trig_log_argument(ctx, builtin, arg)?;
    let log_arg = cas_ast::hold::wrap_hold(ctx, log_arg);
    let log_abs = ln_abs(ctx, log_arg);
    if let Some(coeff) = rational_constant_value(ctx, coeff) {
        if !coeff.is_zero() {
            return Some(scale_reciprocal_integration_result_preserving_presentation(
                ctx,
                BigRational::one() / coeff,
                log_abs,
            ));
        }
    }
    Some(divide_by_coeff_unless_one_preserving_presentation(
        ctx, log_abs, coeff,
    ))
}

pub(super) fn reciprocal_trig_log_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let numerator_scale = rational_constant_value(ctx, num)?;
    if numerator_scale.is_zero() {
        return None;
    }

    let (denominator_scale, builtin, arg) = scaled_reciprocal_trig_log_denominator(ctx, den)?;
    let (a, _) = get_linear_coeffs(ctx, arg, var)?;
    if let Some(slope) = rational_constant_value(ctx, a) {
        let effective_slope = slope * denominator_scale.clone() / numerator_scale.clone();
        if !effective_slope.is_zero() {
            let effective_slope = ctx.add(Expr::Number(effective_slope));
            return sec_csc_log_antiderivative(ctx, builtin, arg, effective_slope);
        }
    }
    let primitive = sec_csc_log_antiderivative(ctx, builtin, arg, a)?;
    Some(scale_reciprocal_integration_result_preserving_presentation(
        ctx,
        numerator_scale / denominator_scale,
        primitive,
    ))
}

pub(super) fn polynomial_reciprocal_trig_log_antiderivative(
    ctx: &mut Context,
    num: ExprId,
    den: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (builtin, arg) = reciprocal_trig_reciprocal_parts_from_denominator(ctx, den)?;
    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let scale = symbolic_linear_cofactor_scale_expr(ctx, num, arg, var)?;
    if is_number(ctx, scale, 0) {
        return None;
    }

    let one = ctx.num(1);
    let primitive = sec_csc_log_antiderivative(ctx, builtin, arg, one)?;
    Some(
        scale_expr_reciprocal_integration_result_preserving_presentation(
            ctx, scale, primitive, false,
        ),
    )
}

fn scaled_reciprocal_trig_log_denominator(
    ctx: &Context,
    den: ExprId,
) -> Option<(BigRational, BuiltinFn, ExprId)> {
    if let Some((builtin, arg)) = reciprocal_trig_reciprocal_parts_from_denominator(ctx, den) {
        return Some((BigRational::one(), builtin, arg));
    }

    let factors = mul_leaves(ctx, den);
    if factors.len() < 2 {
        return None;
    }

    let mut denominator_scale = BigRational::one();
    let mut matched_denominator: Option<(BuiltinFn, ExprId)> = None;
    for factor in factors {
        if let Some(value) = rational_constant_value(ctx, factor) {
            denominator_scale *= value;
            continue;
        }
        let parts = reciprocal_trig_reciprocal_parts_from_denominator(ctx, factor)?;
        if matched_denominator.replace(parts).is_some() {
            return None;
        }
    }

    if denominator_scale.is_zero() {
        return None;
    }
    let (builtin, arg) = matched_denominator?;
    Some((denominator_scale, builtin, arg))
}

fn scaled_reciprocal_trig_log_denominator_call(
    ctx: &Context,
    den: ExprId,
) -> Option<(BigRational, BuiltinFn, ExprId)> {
    if let Some((builtin, arg)) = reciprocal_trig_denominator_call(ctx, den) {
        return Some((BigRational::one(), builtin, arg));
    }

    let factors = mul_leaves(ctx, den);
    if factors.len() < 2 {
        return None;
    }

    let mut denominator_scale = BigRational::one();
    let mut matched_denominator: Option<(BuiltinFn, ExprId)> = None;
    for factor in factors {
        if let Some(value) = rational_constant_value(ctx, factor) {
            denominator_scale *= value;
            continue;
        }
        let parts = reciprocal_trig_denominator_call(ctx, factor)?;
        if matched_denominator.replace(parts).is_some() {
            return None;
        }
    }

    if denominator_scale.is_zero() {
        return None;
    }
    let (builtin, arg) = matched_denominator?;
    Some((denominator_scale, builtin, arg))
}

pub(super) fn trig_log_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    if let Some(inner) = constant_scaled_integrand_inner(ctx, expr, var) {
        return trig_log_required_nonzero(ctx, inner, var);
    }

    if matches!(ctx.get(expr), Expr::Div(_, _)) {
        return reciprocal_trig_log_required_nonzero(ctx, expr, var);
    }

    let (builtin, arg) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => (ctx.builtin_of(fn_id)?, args[0]),
        _ => return None,
    };
    if !contains_named_var(ctx, arg, var) {
        return None;
    }
    build_trig_pole_nonzero_condition(ctx, builtin, arg)
}

pub(super) fn residual_trig_pole_required_nonzero_conditions(
    ctx: &mut Context,
    expr: ExprId,
) -> Vec<ExprId> {
    let mut conditions = Vec::new();
    let mut stack = vec![(expr, 0usize)];

    while let Some((current, depth)) = stack.pop() {
        if depth > SYMBOLIC_INTEGRATION_DOMAIN_SCAN_DEPTH {
            continue;
        }

        match ctx.get(current).clone() {
            Expr::Add(left, right)
            | Expr::Sub(left, right)
            | Expr::Mul(left, right)
            | Expr::Div(left, right)
            | Expr::Pow(left, right) => {
                stack.push((left, depth + 1));
                stack.push((right, depth + 1));
            }
            Expr::Neg(inner) | Expr::Hold(inner) => stack.push((inner, depth + 1)),
            Expr::Function(fn_id, args) => {
                if args.len() == 1 {
                    if let Some(nonzero_builtin) =
                        ctx.builtin_of(fn_id).and_then(trig_pole_nonzero_builtin)
                    {
                        let arg = args[0];
                        if contains_variable(ctx, arg) {
                            conditions.push(ctx.call_builtin(nonzero_builtin, vec![arg]));
                        }
                    }
                }
                stack.extend(args.into_iter().map(|arg| (arg, depth + 1)));
            }
            Expr::Matrix { data, .. } => {
                stack.extend(data.into_iter().map(|entry| (entry, depth + 1)));
            }
            Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => {}
        }
    }

    conditions
}

pub(super) fn polynomial_trig_log_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    let (trig_index, builtin, arg) = indexed_trig_pole_builtin_factor(ctx, &factors)?;

    if has_trig_pole_builtin_factor_except(ctx, &factors, trig_index) {
        return None;
    }

    if !contains_named_var(ctx, arg, var) {
        return None;
    }

    let cofactor = factor_product_excluding_index(ctx, &factors, trig_index);
    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    let derivative = arg_poly.derivative();
    let scale = constant_polynomial_ratio(&cofactor_poly, &derivative)?;
    if scale.is_zero() {
        return None;
    }

    build_trig_pole_nonzero_condition(ctx, builtin, arg)
}

pub(super) fn reciprocal_trig_log_required_nonzero(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (num, den) = match ctx.get(expr).clone() {
        Expr::Div(num, den) => (num, den),
        _ => return None,
    };

    let (denominator_scale, den_builtin, arg) =
        scaled_reciprocal_trig_log_denominator_call(ctx, den)?;

    let scale = symbolic_linear_cofactor_scale_expr(ctx, num, arg, var)?;
    if is_number(ctx, scale, 0) || denominator_scale.is_zero() {
        return None;
    }

    build_reciprocal_trig_denominator_nonzero_condition(ctx, den_builtin, arg)
}

/// p(x) * sqrt(q) for quadratic q: rewrite as (p*q)/sqrt(q) and
/// delegate to the quotient owners (reduction families, Hermite
/// split). Covers the whole radical-numerator chapter - sqrt(1-x^2),
/// x^2 sqrt(1-x^2), sqrt(x^2 +/- a^2), completed squares - in one
/// recognizer. Built INTERNALLY (Mul with Pow(q,-1/2)) because the
/// public pre-simplifier cancels q/sqrt(q) back to sqrt(q).
/// Rational functions of e^x: substitute u = e^(c x) with c the gcd of
/// all exponent slopes, build the SINGLE flattened quotient
/// num(u)/(den(u) * c * u) via Polynomial arithmetic (the rational
/// owners dispatch on raw AST and do not flatten nested Divs), then
/// integrate in u and back-substitute. Covers 1/(1+e^x),
/// e^x/(1+e^(2x)) -> arctan(e^x), e^(2x)/(1+e^x), (e^x-1)/(e^x+1)...
/// Symbolic coefficients decline (rational constants only). Ordered
/// AFTER the linear-exponential owners so their displays survive.
/// `integral x^(2k+1) f(c x^2) dx` for f in {exp, sin, cos} and odd power
/// `2k+1 >= 3`: substituting `u = x^2` (so `x dx = du/2` and
/// `x^(2k+1) = x (x^2)^k = x u^k`) turns it into `(1/2) integral u^k f(c u) du`,
/// delegated to the existing polynomial*{exp,trig} by-parts owner, then
/// back-substituting `u = x^2`. The `k=0` case `x f(x^2)` is already owned by
/// the derivative-substitution rule, so this only fires for `2k+1 >= 3`.
/// `integral p(x) sin(u)^2 dx` and `integral p(x) cos(u)^2 dx` for a polynomial
/// `p` (degree >= 1) and affine `u`: apply the power reduction
/// `sin^2(u) = (1 - cos(2u))/2`, `cos^2(u) = (1 + cos(2u))/2`, so the integrand
/// becomes `p/2 -/+ (p/2) cos(2u)`, delegated to the existing power-rule and
/// polynomial*trig by-parts owners. The bare (constant-cofactor) case is already
/// owned by trig_square_affine_antiderivative.
pub(super) fn polynomial_times_trig_square_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }
    // Find exactly one sin(u)^2 / cos(u)^2 factor with an affine argument.
    let mut trig: Option<(usize, BuiltinFn, ExprId)> = None;
    for (i, factor) in factors.iter().enumerate() {
        let Expr::Pow(base, exp) = ctx.get(*factor).clone() else {
            continue;
        };
        if !is_number(ctx, exp, 2) {
            continue;
        }
        let Expr::Function(fn_id, args) = ctx.get(base).clone() else {
            continue;
        };
        if args.len() != 1 {
            continue;
        }
        let builtin = match ctx.builtin_of(fn_id) {
            Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos)) => b,
            _ => continue,
        };
        // affine argument with a nonzero rational slope
        let Some((slope, _)) = get_linear_coeffs(ctx, args[0], var) else {
            continue;
        };
        let Some(slope) = rational_constant_value(ctx, slope) else {
            continue;
        };
        if slope.is_zero() {
            continue;
        }
        if trig.is_some() {
            return None; // more than one trig-square factor is out of scope
        }
        trig = Some((i, builtin, args[0]));
    }
    let (trig_idx, builtin, u) = trig?;

    // The remaining factors must be a non-constant polynomial cofactor.
    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != trig_idx)
        .map(|(_, f)| *f)
        .collect();
    let cofactor = build_balanced_mul(ctx, &cofactor_factors);
    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    if cofactor_poly.degree() == 0 {
        return None; // a constant cofactor is the bare power-reduction case
    }

    // Power reduction: sin^2(u) = (1 - cos(2u))/2, cos^2(u) = (1 + cos(2u))/2.
    // Build the DISTRIBUTED form p/2 -/+ (p/2) cos(2u) -- the integrator handles
    // p/2 (power rule) and p*cos(2u)/2 (polynomial*trig) but does NOT distribute
    // a p*(sum) product itself, so the two terms must be split here.
    let two = ctx.num(2);
    let two_u = mul2_raw(ctx, two, u);
    let cos_2u = ctx.call_builtin(BuiltinFn::Cos, vec![two_u]);
    let half = ctx.add(Expr::Number(BigRational::new(1.into(), 2.into())));
    let half_cofactor = mul2_raw(ctx, half, cofactor);
    let cofactor_cos = mul2_raw(ctx, cofactor, cos_2u);
    let half_cofactor_cos = mul2_raw(ctx, half, cofactor_cos);
    let rewritten = match builtin {
        BuiltinFn::Sin => ctx.add(Expr::Sub(half_cofactor, half_cofactor_cos)),
        BuiltinFn::Cos => ctx.add(Expr::Add(half_cofactor, half_cofactor_cos)),
        _ => return None,
    };
    integrate_symbolic_expr(ctx, rewritten, var)
}

/// `p(x) * sin(ax+b)^n` / `p(x) * cos(ax+b)^n` for an EVEN power `n` in `4..=8`
/// and a non-constant polynomial cofactor `p`. Generalizes the `n == 2`
/// power-reduction owner: the even power reduces to a cosine sum
/// `sin^(2m)(u) = C(2m,m)/4^m + (2/4^m) Σ_{j=1}^{m} (-1)^j C(2m, m-j) cos(2j u)`
/// (the cosine variant drops the `(-1)^j`). Multiplying by `p` and DISTRIBUTING
/// gives `p·C(2m,m)/4^m + Σ_j (coeff·p) cos(2j u)`, each term already owned (the
/// bare polynomial integrator and the polynomial-times-cos(affine) by-parts). As
/// for `n == 2`, the integrator does not distribute a `p·(sum)` product itself,
/// so the distributed sum is built here. Runs after the `n == 2` owner, so it
/// only adds even `n >= 4`.
pub(super) fn polynomial_times_higher_even_trig_power_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }
    // Find exactly one sin(u)^n / cos(u)^n factor with even n in 4..=8 and an
    // affine argument with nonzero rational slope.
    let mut trig: Option<(usize, BuiltinFn, ExprId, u32)> = None;
    for (i, factor) in factors.iter().enumerate() {
        let Expr::Pow(base, exp) = ctx.get(*factor).clone() else {
            continue;
        };
        let Expr::Number(exp_value) = ctx.get(exp).clone() else {
            continue;
        };
        if !exp_value.is_integer() {
            continue;
        }
        let Some(n) = exp_value.to_integer().to_u32() else {
            continue;
        };
        if !(4..=8).contains(&n) || !n.is_multiple_of(2) {
            continue;
        }
        let Expr::Function(fn_id, args) = ctx.get(base).clone() else {
            continue;
        };
        if args.len() != 1 {
            continue;
        }
        let builtin = match ctx.builtin_of(fn_id) {
            Some(b @ (BuiltinFn::Sin | BuiltinFn::Cos)) => b,
            _ => continue,
        };
        let Some((slope, _)) = get_linear_coeffs(ctx, args[0], var) else {
            continue;
        };
        let Some(slope) = rational_constant_value(ctx, slope) else {
            continue;
        };
        if slope.is_zero() {
            continue;
        }
        if trig.is_some() {
            return None; // more than one trig-power factor is out of scope
        }
        trig = Some((i, builtin, args[0], n));
    }
    let (trig_idx, builtin, u, n) = trig?;

    // The remaining factors must be a non-constant polynomial cofactor.
    let cofactor_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != trig_idx)
        .map(|(_, f)| *f)
        .collect();
    let cofactor = build_balanced_mul(ctx, &cofactor_factors);
    let cofactor_poly = Polynomial::from_expr(ctx, cofactor, var).ok()?;
    if cofactor_poly.degree() == 0 {
        return None; // a constant cofactor is the bare power-reduction case
    }

    let m = n / 2;
    let mut four_m = num_bigint::BigInt::from(1);
    for _ in 0..m {
        four_m *= 4;
    }

    // Constant term: p · C(2m,m)/4^m.
    let c0 = BigRational::new(
        num_bigint::BigInt::from(crate::combinatorics::binomial_coeff(n, m)),
        four_m.clone(),
    );
    let mut acc = scale_rational_term(ctx, c0, cofactor);

    // Cosine terms: (coeff · p) cos(2j u), j = 1..=m.
    for j in 1..=m {
        let magnitude = BigRational::new(
            num_bigint::BigInt::from(2 * crate::combinatorics::binomial_coeff(n, m - j)),
            four_m.clone(),
        );
        let coeff = match builtin {
            // sin^(2m) carries (-1)^j; cos^(2m) is all positive.
            BuiltinFn::Sin if !j.is_multiple_of(2) => -magnitude,
            BuiltinFn::Sin | BuiltinFn::Cos => magnitude,
            _ => return None,
        };
        let two_j = ctx.num((2 * j) as i64);
        let arg = mul2_raw(ctx, two_j, u);
        let cos_term = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
        let cofactor_cos = mul2_raw(ctx, cofactor, cos_term);
        let term = scale_rational_term(ctx, coeff, cofactor_cos);
        acc = ctx.add(Expr::Add(acc, term));
    }
    integrate_symbolic_expr(ctx, acc, var)
}

/// Accumulate sin/cos powers from a product of sin(k var)^p / cos(k var)^p
/// factors sharing one rational nonzero slope. Returns false on any
/// foreign factor or argument mismatch.
pub(super) fn collect_mixed_trig_powers(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
    sin_power: &mut i64,
    cos_power: &mut i64,
    slope: &mut Option<BigRational>,
) -> bool {
    match ctx.get(expr).clone() {
        Expr::Mul(l, r) => {
            collect_mixed_trig_powers(ctx, l, var, sin_power, cos_power, slope)
                && collect_mixed_trig_powers(ctx, r, var, sin_power, cos_power, slope)
        }
        Expr::Pow(base, exponent) => {
            let Some(value) = crate::numeric_eval::as_rational_const(ctx, exponent) else {
                return false;
            };
            if !value.is_integer() || value <= BigRational::zero() {
                return false;
            }
            let Ok(power) = i64::try_from(value.numer()) else {
                return false;
            };
            let Some((atom_slope, is_sine)) = weierstrass_trig_atom(ctx, base, var) else {
                return false;
            };
            if !mixed_trig_slope_consistent(slope, atom_slope) {
                return false;
            }
            if is_sine {
                *sin_power += power;
            } else {
                *cos_power += power;
            }
            true
        }
        _ => {
            let Some((atom_slope, is_sine)) = weierstrass_trig_atom(ctx, expr, var) else {
                return false;
            };
            if !mixed_trig_slope_consistent(slope, atom_slope) {
                return false;
            }
            if is_sine {
                *sin_power += 1;
            } else {
                *cos_power += 1;
            }
            true
        }
    }
}

fn mixed_trig_slope_consistent(slope: &mut Option<BigRational>, atom_slope: BigRational) -> bool {
    match slope {
        Some(existing) => *existing == atom_slope,
        None => {
            *slope = Some(atom_slope);
            true
        }
    }
}

/// sin(k*var) or cos(k*var) with rational nonzero k and zero offset.
/// Returns (k, is_sine).
pub(super) fn weierstrass_trig_atom(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(BigRational, bool)> {
    let (arg, is_sine) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => match ctx.builtin_of(fn_id) {
            Some(BuiltinFn::Sin) => (args[0], true),
            Some(BuiltinFn::Cos) => (args[0], false),
            _ => return None,
        },
        _ => return None,
    };
    let (slope_expr, offset) = get_linear_coeffs(ctx, arg, var)?;
    if !is_number(ctx, offset, 0) {
        return None;
    }
    let slope = rational_constant_value(ctx, slope_expr)?;
    (!slope.is_zero()).then_some((slope, is_sine))
}

/// Product-to-sum for distinct-frequency trig products with pure k*x
/// arguments: sin(ax)cos(bx), sin(ax)sin(bx), cos(ax)cos(bx) with
/// rational a != +-b integrate via the closed forms of the three
/// product-to-sum identities, built DIRECTLY (delegating sin(kx) to the
/// public route can produce power-expansion forms). Equal frequencies
/// keep their existing owners.
pub(super) fn trig_product_to_sum_antiderivative(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (left_is_sin, a) = pure_linear_trig_factor(ctx, left, var)?;
    let (right_is_sin, b) = pure_linear_trig_factor(ctx, right, var)?;
    if a == b || a == -b.clone() {
        return None;
    }

    let sum = &a + &b;
    let difference = &a - &b;
    let half = BigRational::new(1.into(), 2.into());

    // Each term: trig((a +- b) x) scaled; integral of sin(kx) is
    // -cos(kx)/k and of cos(kx) is sin(kx)/k.
    let term =
        |ctx: &mut Context, freq: &BigRational, is_sin: bool, scale: BigRational| -> ExprId {
            let freq_expr = ctx.add(Expr::Number(freq.clone()));
            let var_expr = ctx.var(var);
            let arg = mul2_raw(ctx, freq_expr, var_expr);
            if is_sin {
                // scale * integral(sin) = scale * (-cos(freq x)/freq)
                let cosine = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
                let coefficient = ctx.add(Expr::Number(-(scale / freq)));
                mul2_raw(ctx, coefficient, cosine)
            } else {
                let sine = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
                let coefficient = ctx.add(Expr::Number(scale / freq));
                mul2_raw(ctx, coefficient, sine)
            }
        };

    let (first, second) = match (left_is_sin, right_is_sin) {
        // sin(ax)cos(bx) = 1/2 [ sin((a+b)x) + sin((a-b)x) ]
        (true, false) => (
            term(ctx, &sum, true, half.clone()),
            term(ctx, &difference, true, half),
        ),
        // cos(ax)sin(bx): swap roles
        (false, true) => {
            let swapped_difference = &b - &a;
            (
                term(ctx, &sum, true, half.clone()),
                term(ctx, &swapped_difference, true, half),
            )
        }
        // sin sin = 1/2 [ cos((a-b)x) - cos((a+b)x) ]
        (true, true) => (
            term(ctx, &difference, false, half.clone()),
            term(ctx, &sum, false, -half),
        ),
        // cos cos = 1/2 [ cos((a-b)x) + cos((a+b)x) ]
        (false, false) => (
            term(ctx, &difference, false, half.clone()),
            term(ctx, &sum, false, half),
        ),
    };
    Some(ctx.add(Expr::Add(first, second)))
}

/// A sin/cos factor with a pure rational-multiple argument k*x; returns
/// (is_sin, k).
fn pure_linear_trig_factor(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<(bool, BigRational)> {
    let (fn_id, arg) = match ctx.get(expr) {
        Expr::Function(fn_id, args) if args.len() == 1 => (*fn_id, args[0]),
        _ => return None,
    };
    let is_sin = match ctx.builtin_of(fn_id) {
        Some(BuiltinFn::Sin) => true,
        Some(BuiltinFn::Cos) => false,
        _ => return None,
    };
    let poly = Polynomial::from_expr(ctx, arg, var).ok()?;
    if poly.degree() != 1 || !poly.coeffs[0].is_zero() {
        return None;
    }
    Some((is_sin, poly.coeffs[1].clone()))
}

fn scaled_exp_trig_antiderivative(
    ctx: &mut Context,
    exp_factor: ExprId,
    inner: ExprId,
    scale: BigRational,
    cofactor: ExprId,
    has_nontrivial_cofactor: bool,
) -> ExprId {
    let exp_inner = mul2_raw(ctx, exp_factor, inner);
    if let Some(cofactor_scale) = rational_constant_value(ctx, cofactor) {
        return scale_rational_term(ctx, scale * cofactor_scale, exp_inner);
    }

    let integral = scale_rational_term(ctx, scale, exp_inner);
    if has_nontrivial_cofactor {
        mul2_raw(ctx, cofactor, integral)
    } else {
        integral
    }
}

fn exp_trig_distinct_linear_inner(
    ctx: &mut Context,
    builtin: BuiltinFn,
    trig_arg: ExprId,
    exp_slope: &BigRational,
    trig_slope: &BigRational,
) -> Option<ExprId> {
    let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![trig_arg]);
    let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![trig_arg]);
    match builtin {
        BuiltinFn::Sin => {
            let sin_term = scale_rational_term(ctx, exp_slope.clone(), sin_arg);
            let cos_term = scale_rational_term(ctx, trig_slope.clone(), cos_arg);
            Some(ctx.add(Expr::Sub(sin_term, cos_term)))
        }
        BuiltinFn::Cos => {
            let cos_term = scale_rational_term(ctx, exp_slope.clone(), cos_arg);
            let sin_term = scale_rational_term(ctx, trig_slope.clone(), sin_arg);
            Some(ctx.add(Expr::Add(cos_term, sin_term)))
        }
        _ => None,
    }
}

fn exp_trig_same_linear_inner(
    ctx: &mut Context,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![arg]);
    let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![arg]);
    match builtin {
        BuiltinFn::Sin => Some(ctx.add(Expr::Sub(sin_arg, cos_arg))),
        BuiltinFn::Cos => Some(ctx.add(Expr::Add(sin_arg, cos_arg))),
        _ => None,
    }
}

pub(super) fn exp_trig_same_linear_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let factors = mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (exp_index, exp_factor) in factors.iter().enumerate() {
        let Some((exp_arg, arg_slope)) = linear_exp_factor_parts(ctx, *exp_factor, var).ok()?
        else {
            continue;
        };

        for (trig_index, trig_factor) in factors.iter().enumerate() {
            if trig_index == exp_index {
                continue;
            }

            let Some((builtin, trig_arg, trig_slope)) =
                linear_trig_factor_parts(ctx, *trig_factor, var).ok()?
            else {
                continue;
            };

            let cofactor_factors = factors_excluding_two_indices(&factors, exp_index, trig_index);
            let cofactor = factor_product_or_one(ctx, &cofactor_factors);
            if contains_named_var(ctx, cofactor, var) {
                continue;
            }

            if compare_expr(ctx, exp_arg, trig_arg) != Ordering::Equal {
                let Some(inner) =
                    exp_trig_distinct_linear_inner(ctx, builtin, trig_arg, &arg_slope, &trig_slope)
                else {
                    continue;
                };
                let denominator =
                    arg_slope.clone() * arg_slope.clone() + trig_slope.clone() * trig_slope.clone();
                if denominator.is_zero() {
                    continue;
                }
                let scale = BigRational::one() / denominator;
                return Some(scaled_exp_trig_antiderivative(
                    ctx,
                    *exp_factor,
                    inner,
                    scale,
                    cofactor,
                    !cofactor_factors.is_empty(),
                ));
            }

            let Some(inner) = exp_trig_same_linear_inner(ctx, builtin, exp_arg) else {
                continue;
            };
            let scale = BigRational::one() / (BigRational::from_integer(2.into()) * arg_slope);
            return Some(scaled_exp_trig_antiderivative(
                ctx,
                *exp_factor,
                inner,
                scale,
                cofactor,
                !cofactor_factors.is_empty(),
            ));
        }
    }

    None
}

pub fn integrate_symbolic_is_exp_trig_same_linear_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    exp_trig_same_linear_antiderivative(ctx, expr, var).is_some()
}

/// `integral cos(ln(u)) du = u/2 (cos(ln u) + sin(ln u))` and
/// `integral sin(ln(u)) du = u/2 (sin(ln u) - cos(ln u))` for an affine inner
/// `u` in `x`. Substituting `t = ln u` turns the integrand into the cyclic
/// `cos(t) e^t` / `sin(t) e^t` form; the `1/u'` affine cofactor scales it. Only
/// fires on a bare `cos(ln(affine))` / `sin(ln(affine))`, so a non-logarithmic
/// or non-affine inner stays an honest residual.
pub(super) fn trig_of_log_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (builtin, ln_arg) = match ctx.get(expr).clone() {
        Expr::Function(fn_id, args) if args.len() == 1 => match ctx.builtin_of(fn_id) {
            Some(b @ (BuiltinFn::Cos | BuiltinFn::Sin)) => (b, args[0]),
            _ => return None,
        },
        _ => return None,
    };
    let inner = unary_builtin_arg(ctx, ln_arg, BuiltinFn::Ln)?;
    let (inner, slope) = nonzero_linear_arg_and_slope(ctx, inner, var)?;
    let slope = rational_constant_value(ctx, slope)?;
    // scale = 1 / (2 u'); the affine cofactor of the inner substitution.
    let scale = BigRational::one() / (BigRational::from_integer(2.into()) * slope);

    let cos_ln = ctx.call_builtin(BuiltinFn::Cos, vec![ln_arg]);
    let sin_ln = ctx.call_builtin(BuiltinFn::Sin, vec![ln_arg]);
    let combo = match builtin {
        BuiltinFn::Cos => ctx.add(Expr::Add(cos_ln, sin_ln)),
        BuiltinFn::Sin => ctx.add(Expr::Sub(sin_ln, cos_ln)),
        _ => unreachable!("trig_of_log only matches cos/sin"),
    };
    let inner_combo = mul2_raw(ctx, inner, combo);
    let scale_expr = ctx.add(Expr::Number(scale));
    Some(mul2_raw(ctx, scale_expr, inner_combo))
}

pub fn integrate_symbolic_is_polynomial_times_trig_linear_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    if is_polynomial_times_linear_function_target(
        ctx,
        expr,
        var,
        2,
        MAX_TRIG_POLYNOMIAL_BY_PARTS_DEGREE,
        signed_trig_like_factor,
    ) {
        return true;
    }

    additive_polynomial_times_trig_linear_antiderivative(ctx, expr, var).is_some()
}

fn polynomial_trig_term(
    ctx: &mut Context,
    poly: &Polynomial,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    if poly.is_zero() {
        return None;
    }

    let trig = ctx.call_builtin(builtin, vec![arg]);
    let poly_expr = poly.to_expr(ctx);
    if poly.degree() == 0 && poly.coeffs.first().is_some_and(BigRational::is_one) {
        Some(trig)
    } else {
        Some(mul2_raw(ctx, poly_expr, trig))
    }
}

pub(super) fn polynomial_times_trig_linear_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (outer_sign, factors) = signed_mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (trig_index, factor) in factors.iter().enumerate() {
        let Some(trig_parts) =
            signed_linear_function_factor_parts(ctx, *factor, var, signed_trig_like_factor)
        else {
            continue;
        };

        let effective_sign = combine_factor_signs(outer_sign, trig_parts.sign);
        let cofactor =
            signed_factor_product_excluding_index(ctx, &factors, trig_index, effective_sign);
        let Ok(cofactor_poly) = Polynomial::from_expr(ctx, cofactor, var) else {
            continue;
        };
        if !(2..=MAX_TRIG_POLYNOMIAL_BY_PARTS_DEGREE).contains(&cofactor_poly.degree()) {
            continue;
        }

        let (sin_poly, cos_poly) = polynomial_trig_by_parts_polys(
            &cofactor_poly,
            trig_parts.builtin,
            &trig_parts.arg_slope,
        )?;

        return match trig_parts.builtin {
            BuiltinFn::Sin | BuiltinFn::Cos => {
                let terms: Vec<ExprId> = [
                    polynomial_trig_term(ctx, &sin_poly, BuiltinFn::Sin, trig_parts.arg),
                    polynomial_trig_term(ctx, &cos_poly, BuiltinFn::Cos, trig_parts.arg),
                ]
                .into_iter()
                .flatten()
                .collect();
                Some(build_balanced_add(ctx, &terms))
            }
            _ => None,
        };
    }

    None
}

pub(super) fn additive_polynomial_times_trig_linear_antiderivative(
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
            polynomial_trig_linear_term_parts(ctx, term, var)?;
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

    if !(2..=MAX_TRIG_POLYNOMIAL_BY_PARTS_DEGREE).contains(&cofactor_sum.degree()) {
        return None;
    }

    let cofactor = cofactor_sum.to_expr(ctx);
    let combined_expr = mul2_raw(ctx, cofactor, common_factor?);
    polynomial_times_trig_linear_antiderivative(ctx, combined_expr, var)
}

pub(super) fn linear_times_trig_linear_antiderivative(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let (outer_sign, factors) = signed_mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    for (trig_index, factor) in factors.iter().enumerate() {
        let Some(trig_parts) =
            signed_linear_function_factor_parts(ctx, *factor, var, signed_trig_like_factor)
        else {
            continue;
        };

        let raw_cofactor = factor_product_excluding_index(ctx, &factors, trig_index);
        let effective_sign = combine_factor_signs(outer_sign, trig_parts.sign);
        let cofactor = match effective_sign {
            Sign::Pos => raw_cofactor,
            Sign::Neg => ctx.add(Expr::Neg(raw_cofactor)),
        };
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

        if effective_sign == Sign::Neg && matches!(trig_parts.builtin, BuiltinFn::Sin) {
            let Ok(raw_cofactor_poly) = Polynomial::from_expr(ctx, raw_cofactor, var) else {
                continue;
            };
            let raw_cofactor_slope = raw_cofactor_poly
                .coeffs
                .get(1)
                .cloned()
                .unwrap_or_else(BigRational::zero);
            if raw_cofactor_slope.is_zero() {
                continue;
            }

            let raw_quotient =
                trig_by_parts_quotient_by_slope(ctx, raw_cofactor, &trig_parts.arg_slope);
            let raw_correction =
                raw_cofactor_slope / (trig_parts.arg_slope.clone() * trig_parts.arg_slope.clone());
            let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![trig_parts.arg]);
            let quotient_cos = mul2_raw(ctx, cos_arg, raw_quotient);
            let correction_sin = scale_factor(ctx, raw_correction, trig_parts.factor);
            let result = ctx.add(Expr::Sub(quotient_cos, correction_sin));
            return Some(cas_ast::hold::wrap_hold(ctx, result));
        }

        let quotient = trig_by_parts_quotient_by_slope(ctx, cofactor, &trig_parts.arg_slope);
        let correction =
            cofactor_slope / (trig_parts.arg_slope.clone() * trig_parts.arg_slope.clone());

        return match trig_parts.builtin {
            BuiltinFn::Sin => {
                let cos_arg = ctx.call_builtin(BuiltinFn::Cos, vec![trig_parts.arg]);
                let correction_sin = scale_factor(ctx, correction, trig_parts.factor);
                if trig_parts.arg_slope.is_negative() {
                    let positive_slope = -trig_parts.arg_slope.clone();
                    let positive_quotient =
                        trig_by_parts_quotient_by_slope(ctx, cofactor, &positive_slope);
                    let quotient_cos = mul2_raw(ctx, positive_quotient, cos_arg);
                    Some(ctx.add(Expr::Add(correction_sin, quotient_cos)))
                } else {
                    let quotient_cos = mul2_raw(ctx, quotient, cos_arg);
                    Some(ctx.add(Expr::Sub(correction_sin, quotient_cos)))
                }
            }
            BuiltinFn::Cos => {
                let sin_arg = ctx.call_builtin(BuiltinFn::Sin, vec![trig_parts.arg]);
                let correction_cos = scale_factor(ctx, correction, trig_parts.factor);
                if trig_parts.arg_slope.is_negative() {
                    let positive_slope = -trig_parts.arg_slope.clone();
                    let positive_quotient =
                        trig_by_parts_quotient_by_slope(ctx, cofactor, &positive_slope);
                    let quotient_sin = mul2_raw(ctx, positive_quotient, sin_arg);
                    Some(ctx.add(Expr::Sub(correction_cos, quotient_sin)))
                } else {
                    let quotient_sin = mul2_raw(ctx, quotient, sin_arg);
                    Some(ctx.add(Expr::Add(quotient_sin, correction_cos)))
                }
            }
            _ => None,
        };
    }

    None
}

pub fn integrate_symbolic_is_linear_times_trig_linear_target(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> bool {
    linear_times_trig_linear_antiderivative(ctx, expr, var).is_some()
}

pub(super) fn sqrt_trig_reciprocal_derivative_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_trig_reciprocal_derivative_parts(ctx, expr, var)?;
    Some(parts.radicand)
}

pub(super) fn sqrt_trig_log_derivative_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_trig_log_derivative_parts(ctx, expr, var)?;
    Some(parts.radicand)
}

pub(super) fn sqrt_reciprocal_trig_log_derivative_radicand(
    ctx: &mut Context,
    expr: ExprId,
    var: &str,
) -> Option<ExprId> {
    let parts = sqrt_reciprocal_trig_log_derivative_parts(ctx, expr, var)?;
    Some(parts.radicand)
}
