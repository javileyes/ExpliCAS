use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_domain::exprs_equivalent;
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Zero};

fn expr_eq(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal
}

fn one_expr(ctx: &mut Context) -> ExprId {
    ctx.num(1)
}

fn expr_is_one(ctx: &mut Context, expr: ExprId) -> bool {
    let one = one_expr(ctx);
    expr_eq(ctx, expr, one)
}

fn unary_builtin_arg(ctx: &Context, expr: ExprId, builtin: BuiltinFn) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(expr) else {
        return None;
    };
    (ctx.builtin_of(*fn_id) == Some(builtin) && args.len() == 1).then_some(args[0])
}

fn same_arg_unary_pair(
    ctx: &Context,
    left: ExprId,
    first_builtin: BuiltinFn,
    right: ExprId,
    second_builtin: BuiltinFn,
) -> Option<ExprId> {
    let first_arg = unary_builtin_arg(ctx, left, first_builtin)?;
    let second_arg = unary_builtin_arg(ctx, right, second_builtin)?;
    exprs_equivalent(ctx, first_arg, second_arg).then_some(first_arg)
}

fn unordered_same_arg_unary_sum(
    ctx: &Context,
    expr: ExprId,
    first_builtin: BuiltinFn,
    second_builtin: BuiltinFn,
) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    same_arg_unary_pair(ctx, *left, first_builtin, *right, second_builtin)
        .or_else(|| same_arg_unary_pair(ctx, *right, first_builtin, *left, second_builtin))
}

fn csc_minus_cot_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            same_arg_unary_pair(ctx, *left, BuiltinFn::Csc, *right, BuiltinFn::Cot)
        }
        Expr::Add(left, right) => {
            let Expr::Neg(negated_right) = ctx.get(*right) else {
                return None;
            };
            same_arg_unary_pair(ctx, *left, BuiltinFn::Csc, *negated_right, BuiltinFn::Cot)
        }
        _ => None,
    }
}

fn add_one_sin_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    if is_one_constant(ctx, *left) {
        unary_builtin_arg(ctx, *right, BuiltinFn::Sin)
    } else if is_one_constant(ctx, *right) {
        unary_builtin_arg(ctx, *left, BuiltinFn::Sin)
    } else {
        None
    }
}

fn plus_or_minus_one_cos_arg(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Sub(left, right) if is_one_constant(ctx, *left) => {
            unary_builtin_arg(ctx, *right, BuiltinFn::Cos)
        }
        Expr::Sub(left, right) if is_one_constant(ctx, *right) => {
            unary_builtin_arg(ctx, *left, BuiltinFn::Cos)
        }
        _ => None,
    }
}

fn is_one_constant(ctx: &Context, expr: ExprId) -> bool {
    cas_math::numeric_eval::as_rational_const(ctx, expr).is_some_and(|value| value.is_one())
}

fn is_zero_constant(ctx: &Context, expr: ExprId) -> bool {
    cas_math::numeric_eval::as_rational_const(ctx, expr).is_some_and(|value| value.is_zero())
}

fn is_nonzero_constant(ctx: &Context, expr: ExprId) -> bool {
    cas_math::numeric_eval::as_rational_const(ctx, expr).is_some_and(|value| !value.is_zero())
}

fn same_arg_quotient_over_builtin_denominator(
    ctx: &Context,
    expr: ExprId,
    numerator_arg: impl Fn(&Context, ExprId) -> Option<ExprId>,
    denominator_builtin: BuiltinFn,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num_arg = numerator_arg(ctx, *num)?;
    let den_arg = unary_builtin_arg(ctx, *den, denominator_builtin)?;
    exprs_equivalent(ctx, num_arg, den_arg).then_some(num_arg)
}

fn reciprocal_trig_log_abs_primitive(
    ctx: &Context,
    inner_expr: ExprId,
) -> Option<(BuiltinFn, ExprId)> {
    if let Some(arg) = unordered_same_arg_unary_sum(ctx, inner_expr, BuiltinFn::Sec, BuiltinFn::Tan)
    {
        return Some((BuiltinFn::Sec, arg));
    }

    if let Some(arg) = csc_minus_cot_arg(ctx, inner_expr) {
        return Some((BuiltinFn::Csc, arg));
    }

    if let Some(arg) =
        same_arg_quotient_over_builtin_denominator(ctx, inner_expr, add_one_sin_arg, BuiltinFn::Cos)
    {
        return Some((BuiltinFn::Sec, arg));
    }

    if let Some(arg) = same_arg_quotient_over_builtin_denominator(
        ctx,
        inner_expr,
        plus_or_minus_one_cos_arg,
        BuiltinFn::Sin,
    ) {
        return Some((BuiltinFn::Csc, arg));
    }

    None
}

fn ln_abs_inner(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let ln_arg = unary_builtin_arg(ctx, expr, BuiltinFn::Ln)?;
    unary_builtin_arg(ctx, ln_arg, BuiltinFn::Abs)
}

fn scaled_ln_abs_inner(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    if let Some(inner) = ln_abs_inner(ctx, expr) {
        return Some((BigRational::one(), inner));
    }

    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let (scale, inner) = scaled_ln_abs_inner(ctx, *inner)?;
            Some((-scale, inner))
        }
        Expr::Div(num, den) => {
            let den_scale = cas_math::numeric_eval::as_rational_const(ctx, *den)?;
            if den_scale.is_zero() {
                return None;
            }
            let (scale, inner) = scaled_ln_abs_inner(ctx, *num)?;
            Some((scale / den_scale, inner))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut matched_inner = None;

            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(factor_scale) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    scale *= factor_scale;
                    continue;
                }

                let (factor_scale, inner) = scaled_ln_abs_inner(ctx, factor)?;
                if matched_inner.replace(inner).is_some() {
                    return None;
                }
                scale *= factor_scale;
            }

            Some((scale, matched_inner?))
        }
        _ => None,
    }
}

fn reciprocal_trig_denominator_builtin(builtin: BuiltinFn) -> Option<BuiltinFn> {
    match builtin {
        BuiltinFn::Sec => Some(BuiltinFn::Cos),
        BuiltinFn::Csc => Some(BuiltinFn::Sin),
        _ => None,
    }
}

fn reciprocal_trig_target_coefficient(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<BigRational> {
    if let Some(target_arg) = unary_builtin_arg(ctx, expr, builtin) {
        if exprs_equivalent(ctx, target_arg, arg) {
            return Some(BigRational::one());
        }
    }

    match ctx.get(expr) {
        Expr::Neg(inner) => {
            reciprocal_trig_target_coefficient(ctx, *inner, builtin, arg).map(|coeff| -coeff)
        }
        Expr::Div(num, den) => {
            let denominator_builtin = reciprocal_trig_denominator_builtin(builtin)?;
            if let Some(den_arg) = unary_builtin_arg(ctx, *den, denominator_builtin) {
                if exprs_equivalent(ctx, den_arg, arg) {
                    return cas_math::numeric_eval::as_rational_const(ctx, *num);
                }
            }

            let den_scale = cas_math::numeric_eval::as_rational_const(ctx, *den)?;
            if den_scale.is_zero() {
                return None;
            }
            let num_coeff = reciprocal_trig_target_coefficient(ctx, *num, builtin, arg)?;
            Some(num_coeff / den_scale)
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut matched_target = false;

            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(factor_scale) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    scale *= factor_scale;
                    continue;
                }

                if matched_target {
                    return None;
                }
                scale *= reciprocal_trig_target_coefficient(ctx, factor, builtin, arg)?;
                matched_target = true;
            }

            matched_target.then_some(scale)
        }
        _ => None,
    }
}

fn scaled_unary_builtin_rational_target(
    ctx: &Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<(BigRational, ExprId)> {
    if let Some(arg) = unary_builtin_arg(ctx, expr, builtin) {
        return Some((BigRational::one(), arg));
    }

    match ctx.get(expr) {
        Expr::Neg(inner) => {
            let (scale, arg) = scaled_unary_builtin_rational_target(ctx, *inner, builtin)?;
            Some((-scale, arg))
        }
        Expr::Div(num, den) => {
            let den_scale = cas_math::numeric_eval::as_rational_const(ctx, *den)?;
            if den_scale.is_zero() {
                return None;
            }
            let (scale, arg) = scaled_unary_builtin_rational_target(ctx, *num, builtin)?;
            Some((scale / den_scale, arg))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut matched_arg = None;

            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(factor_scale) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    scale *= factor_scale;
                    continue;
                }

                let (target_scale, arg) =
                    scaled_unary_builtin_rational_target(ctx, factor, builtin)?;
                if matched_arg.replace(arg).is_some() {
                    return None;
                }
                scale *= target_scale;
            }

            Some((scale, matched_arg?))
        }
        _ => None,
    }
}

fn scaled_reciprocal_builtin_rational_target(
    ctx: &Context,
    expr: ExprId,
    denominator_builtin: BuiltinFn,
) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr) {
        Expr::Div(num, den) => {
            let numerator_scale = cas_math::numeric_eval::as_rational_const(ctx, *num)?;
            let arg = unary_builtin_arg(ctx, *den, denominator_builtin)?;
            Some((numerator_scale, arg))
        }
        Expr::Neg(inner) => {
            let (scale, arg) =
                scaled_reciprocal_builtin_rational_target(ctx, *inner, denominator_builtin)?;
            Some((-scale, arg))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut matched_arg = None;

            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(factor_scale) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    scale *= factor_scale;
                    continue;
                }

                let (target_scale, arg) =
                    scaled_reciprocal_builtin_rational_target(ctx, factor, denominator_builtin)?;
                if matched_arg.replace(arg).is_some() {
                    return None;
                }
                scale *= target_scale;
            }

            Some((scale, matched_arg?))
        }
        _ => None,
    }
}

fn reciprocal_trig_derivative_primitive(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, BigRational)> {
    if let Some((scale, arg)) = scaled_unary_builtin_rational_target(ctx, expr, BuiltinFn::Sec) {
        return Some((BuiltinFn::Sec, arg, scale));
    }

    if let Some((scale, arg)) = scaled_unary_builtin_rational_target(ctx, expr, BuiltinFn::Csc) {
        return Some((BuiltinFn::Csc, arg, scale));
    }

    if let Some((scale, arg)) = scaled_reciprocal_builtin_rational_target(ctx, expr, BuiltinFn::Cos)
    {
        return Some((BuiltinFn::Sec, arg, scale));
    }

    if let Some((scale, arg)) = scaled_reciprocal_builtin_rational_target(ctx, expr, BuiltinFn::Sin)
    {
        return Some((BuiltinFn::Csc, arg, scale));
    }

    None
}

fn reciprocal_trig_derivative_pair(builtin: BuiltinFn) -> Option<(BuiltinFn, BuiltinFn)> {
    match builtin {
        BuiltinFn::Sec => Some((BuiltinFn::Sec, BuiltinFn::Tan)),
        BuiltinFn::Csc => Some((BuiltinFn::Csc, BuiltinFn::Cot)),
        _ => None,
    }
}

fn reciprocal_trig_derivative_sign(builtin: BuiltinFn) -> Option<BigRational> {
    match builtin {
        BuiltinFn::Sec => Some(BigRational::one()),
        BuiltinFn::Csc => Some(-BigRational::one()),
        _ => None,
    }
}

fn reciprocal_trig_derivative_target_scale_expr(
    ctx: &mut Context,
    expr: ExprId,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let scale = reciprocal_trig_derivative_target_scale_expr(ctx, inner, builtin, arg)?;
        return Some(ctx.add(Expr::Neg(scale)));
    }

    let (first_builtin, second_builtin) = reciprocal_trig_derivative_pair(builtin)?;
    let mut first_seen = false;
    let mut second_seen = false;
    let mut scale_factors = Vec::new();

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if !first_seen
            && unary_builtin_arg(ctx, factor, first_builtin)
                .is_some_and(|factor_arg| exprs_equivalent(ctx, factor_arg, arg))
        {
            first_seen = true;
            continue;
        }

        if !second_seen
            && unary_builtin_arg(ctx, factor, second_builtin)
                .is_some_and(|factor_arg| exprs_equivalent(ctx, factor_arg, arg))
        {
            second_seen = true;
            continue;
        }

        scale_factors.push(factor);
    }

    if !first_seen || !second_seen {
        return None;
    }

    if scale_factors.is_empty() {
        Some(ctx.num(1))
    } else {
        Some(cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors))
    }
}

fn reciprocal_trig_derivative_integrand_quotient(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let quotient = reciprocal_trig_derivative_integrand_quotient(ctx, inner)?;
        return Some(ctx.add(Expr::Neg(quotient)));
    }

    reciprocal_trig_derivative_integrand_quotient_for_builtin(ctx, expr, BuiltinFn::Sec).or_else(
        || reciprocal_trig_derivative_integrand_quotient_for_builtin(ctx, expr, BuiltinFn::Csc),
    )
}

fn reciprocal_trig_derivative_integrand_quotient_for_builtin(
    ctx: &mut Context,
    expr: ExprId,
    builtin: BuiltinFn,
) -> Option<ExprId> {
    let (ratio_builtin, denominator_builtin) = match builtin {
        BuiltinFn::Sec => (BuiltinFn::Tan, BuiltinFn::Cos),
        BuiltinFn::Csc => (BuiltinFn::Cot, BuiltinFn::Sin),
        _ => return None,
    };

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut reciprocal_index = None;
    let mut ratio_index = None;
    let mut matched_arg = None;

    for (idx, factor) in factors.iter().enumerate() {
        let Some(arg) = unary_builtin_arg(ctx, *factor, builtin) else {
            continue;
        };
        if reciprocal_index.is_some() {
            return None;
        }
        reciprocal_index = Some(idx);
        matched_arg = Some(arg);
    }

    let arg = matched_arg?;
    for (idx, factor) in factors.iter().enumerate() {
        let Some(ratio_arg) = unary_builtin_arg(ctx, *factor, ratio_builtin) else {
            continue;
        };
        if !exprs_equivalent(ctx, ratio_arg, arg) {
            continue;
        }
        if ratio_index.is_some() {
            return None;
        }
        ratio_index = Some(idx);
    }

    let reciprocal_index = reciprocal_index?;
    ratio_index?;

    let numerator_factors: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != reciprocal_index).then_some(*factor))
        .collect();
    let numerator = cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors);
    let denominator = ctx.call_builtin(denominator_builtin, vec![arg]);
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

fn plain_trig_log_abs_primitive(
    ctx: &Context,
    inner_expr: ExprId,
) -> Option<(BuiltinFn, ExprId, BigRational)> {
    if let Some(arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Sin) {
        return Some((BuiltinFn::Cot, arg, BigRational::one()));
    }

    if let Some(arg) = unary_builtin_arg(ctx, inner_expr, BuiltinFn::Cos) {
        return Some((BuiltinFn::Tan, arg, -BigRational::one()));
    }

    None
}

fn plain_trig_quotient_builtins(builtin: BuiltinFn) -> Option<(BuiltinFn, BuiltinFn)> {
    match builtin {
        BuiltinFn::Tan => Some((BuiltinFn::Sin, BuiltinFn::Cos)),
        BuiltinFn::Cot => Some((BuiltinFn::Cos, BuiltinFn::Sin)),
        _ => None,
    }
}

fn scaled_unary_builtin_scale_expr(
    ctx: &mut Context,
    expr: ExprId,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    if let Some(target_arg) = unary_builtin_arg(ctx, expr, builtin) {
        if exprs_equivalent(ctx, target_arg, arg) {
            return Some(ctx.num(1));
        }
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let scale = scaled_unary_builtin_scale_expr(ctx, inner, builtin, arg)?;
            Some(ctx.add(Expr::Neg(scale)))
        }
        Expr::Div(num, den) => {
            let scale = scaled_unary_builtin_scale_expr(ctx, num, builtin, arg)?;
            if expr_is_one(ctx, den) {
                return Some(scale);
            }
            Some(ctx.add(Expr::Div(scale, den)))
        }
        Expr::Mul(_, _) => {
            let mut scale_factors = Vec::new();
            let mut matched_target_scale = None;

            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(target_scale) =
                    scaled_unary_builtin_scale_expr(ctx, factor, builtin, arg)
                {
                    if matched_target_scale.replace(target_scale).is_some() {
                        return None;
                    }
                    continue;
                }

                scale_factors.push(factor);
            }

            let target_scale = matched_target_scale?;
            if !expr_is_one(ctx, target_scale) {
                scale_factors.push(target_scale);
            }

            if scale_factors.is_empty() {
                Some(ctx.num(1))
            } else {
                Some(cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors))
            }
        }
        _ => None,
    }
}

fn plain_trig_target_scale_expr(
    ctx: &mut Context,
    expr: ExprId,
    builtin: BuiltinFn,
    arg: ExprId,
) -> Option<ExprId> {
    if let Some(scale) = scaled_unary_builtin_scale_expr(ctx, expr, builtin, arg) {
        return Some(scale);
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            let scale = plain_trig_target_scale_expr(ctx, inner, builtin, arg)?;
            Some(ctx.add(Expr::Neg(scale)))
        }
        Expr::Div(num, den) => {
            let (num_builtin, den_builtin) = plain_trig_quotient_builtins(builtin)?;
            let den_arg = unary_builtin_arg(ctx, den, den_builtin)?;
            if !exprs_equivalent(ctx, den_arg, arg) {
                return None;
            }
            scaled_unary_builtin_scale_expr(ctx, num, num_builtin, arg)
        }
        Expr::Mul(_, _) => {
            let mut scale_factors = Vec::new();
            let mut matched_target_scale = None;

            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(target_scale) = plain_trig_target_scale_expr(ctx, factor, builtin, arg)
                {
                    if matched_target_scale.replace(target_scale).is_some() {
                        return None;
                    }
                    continue;
                }

                scale_factors.push(factor);
            }

            let target_scale = matched_target_scale?;
            if !expr_is_one(ctx, target_scale) {
                scale_factors.push(target_scale);
            }

            if scale_factors.is_empty() {
                Some(ctx.num(1))
            } else {
                Some(cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors))
            }
        }
        _ => None,
    }
}

fn scale_expr_by_rational(ctx: &mut Context, expr: ExprId, scale: BigRational) -> ExprId {
    if scale.is_one() {
        return expr;
    }

    if scale == -BigRational::one() {
        return ctx.add(Expr::Neg(expr));
    }

    let scale_expr = ctx.add(Expr::Number(scale));
    cas_math::expr_nary::build_balanced_mul(ctx, &[scale_expr, expr])
}

fn scale_polynomial(poly: &Polynomial, scale: &BigRational) -> Polynomial {
    Polynomial::new(
        poly.coeffs.iter().map(|coeff| coeff * scale).collect(),
        poly.var.clone(),
    )
}

fn polynomial_derivative_scale_matches(
    ctx: &Context,
    actual_scale: ExprId,
    trig_arg: ExprId,
    var_name: &str,
    expected_scale: &BigRational,
) -> Option<bool> {
    let actual_poly = Polynomial::from_expr(ctx, actual_scale, var_name).ok()?;
    let arg_poly = Polynomial::from_expr(ctx, trig_arg, var_name).ok()?;
    let expected_poly = scale_polynomial(&arg_poly.derivative(), expected_scale);
    Some(actual_poly == expected_poly)
}

fn diff_call_with_optional_divisor(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    if crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, expr).is_some() {
        let one = one_expr(ctx);
        return Some((expr, one));
    }

    let Expr::Div(num, den) = ctx.get(expr) else {
        return None;
    };
    let num = *num;
    let den = *den;
    crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, num)?;
    Some((num, den))
}

fn is_constant_scaled_hyperbolic_reciprocal_target(ctx: &Context, expr: ExprId) -> bool {
    if let Expr::Neg(inner) = ctx.get(expr) {
        return is_constant_scaled_hyperbolic_reciprocal_target(ctx, *inner);
    }

    let Expr::Div(num, den) = ctx.get(expr) else {
        return false;
    };
    if cas_ast::views::as_rational_const(ctx, *num, 4).is_none() {
        return false;
    }

    let mut matched_hyperbolic = false;
    for factor in cas_math::expr_nary::mul_leaves(ctx, *den) {
        if unary_builtin_arg(ctx, factor, BuiltinFn::Sinh).is_some()
            || unary_builtin_arg(ctx, factor, BuiltinFn::Cosh).is_some()
        {
            if matched_hyperbolic {
                return false;
            }
            matched_hyperbolic = true;
            continue;
        }

        let Some(factor_scale) = cas_ast::views::as_rational_const(ctx, factor, 4) else {
            return false;
        };
        if factor_scale.is_zero() {
            return false;
        }
    }

    matched_hyperbolic
}

fn scaled_expected_matches(
    ctx: &mut Context,
    scale: ExprId,
    builtin: BuiltinFn,
    hyperbolic_arg: ExprId,
    right: ExprId,
) -> bool {
    let tanh = ctx.call_builtin(BuiltinFn::Tanh, vec![hyperbolic_arg]);
    match builtin {
        BuiltinFn::Sinh if expr_is_one(ctx, scale) => {
            let one = one_expr(ctx);
            let reciprocal_tanh = ctx.add(Expr::Div(one, tanh));
            if expr_eq(ctx, reciprocal_tanh, right) {
                return true;
            }

            let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![hyperbolic_arg]);
            let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![hyperbolic_arg]);
            let coth_quotient = ctx.add(Expr::Div(cosh, sinh));
            expr_eq(ctx, coth_quotient, right)
        }
        BuiltinFn::Sinh => {
            let expected = ctx.add(Expr::Div(scale, tanh));
            expr_eq(ctx, expected, right)
        }
        BuiltinFn::Cosh if expr_is_one(ctx, scale) => {
            if expr_eq(ctx, tanh, right) {
                return true;
            }

            let sinh = ctx.call_builtin(BuiltinFn::Sinh, vec![hyperbolic_arg]);
            let cosh = ctx.call_builtin(BuiltinFn::Cosh, vec![hyperbolic_arg]);
            let tanh_quotient = ctx.add(Expr::Div(sinh, cosh));
            expr_eq(ctx, tanh_quotient, right)
        }
        BuiltinFn::Cosh => {
            let expected = ctx.add(Expr::Mul(scale, tanh));
            expr_eq(ctx, expected, right)
        }
        _ => false,
    }
}

fn log_abs_hyperbolic_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<bool> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let ln_arg = unary_builtin_arg(ctx, call.target, BuiltinFn::Ln)?;
    let abs_arg = unary_builtin_arg(ctx, ln_arg, BuiltinFn::Abs)?;

    let (builtin, hyperbolic_arg) =
        if let Some(arg) = unary_builtin_arg(ctx, abs_arg, BuiltinFn::Sinh) {
            (BuiltinFn::Sinh, arg)
        } else {
            (
                BuiltinFn::Cosh,
                unary_builtin_arg(ctx, abs_arg, BuiltinFn::Cosh)?,
            )
        };

    let derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        ctx,
        hyperbolic_arg,
        &call.var_name,
    )?;
    let scale = if expr_is_one(ctx, divisor) {
        derivative
    } else if expr_eq(ctx, derivative, divisor) {
        one_expr(ctx)
    } else {
        ctx.add(Expr::Div(derivative, divisor))
    };

    Some(scaled_expected_matches(
        ctx,
        scale,
        builtin,
        hyperbolic_arg,
        right,
    ))
}

pub(crate) fn try_diff_log_abs_hyperbolic_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    log_abs_hyperbolic_diff_matches(ctx, diff_expr, divisor, right)
        .filter(|matched| *matched)
        .map(|_| ctx.num(0))
}

fn log_abs_reciprocal_trig_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<bool> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    log_abs_reciprocal_trig_target_diff_matches(ctx, call.target, &call.var_name, divisor, right)
}

fn log_abs_reciprocal_trig_target_diff_matches(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    divisor: ExprId,
    right: ExprId,
) -> Option<bool> {
    let (target_scale, inner_expr) = scaled_ln_abs_inner(ctx, target)?;
    let (builtin, trig_arg) = reciprocal_trig_log_abs_primitive(ctx, inner_expr)?;

    let arg_derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        ctx, trig_arg, var_name,
    )?;
    let arg_scale = cas_math::numeric_eval::as_rational_const(ctx, arg_derivative)?;
    let divisor_scale = cas_math::numeric_eval::as_rational_const(ctx, divisor)?;
    if divisor_scale.is_zero() {
        return None;
    }

    let expected_scale = target_scale * arg_scale / divisor_scale;
    let actual_scale = reciprocal_trig_target_coefficient(ctx, right, builtin, trig_arg)?;
    Some(actual_scale == expected_scale)
}

fn integrated_log_abs_reciprocal_trig_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    let antiderivative = cas_math::symbolic_integration_support::integrate_symbolic_expr(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    )?;
    let matched = log_abs_reciprocal_trig_target_diff_matches(
        ctx,
        antiderivative,
        &diff_call.var_name,
        divisor,
        right,
    )?;
    if !matched {
        return None;
    }

    let required_nonzero =
        cas_math::symbolic_integration_support::integrate_symbolic_required_nonzero_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )
        .into_iter()
        .map(crate::ImplicitCondition::NonZero);
    let required_positive =
        cas_math::symbolic_integration_support::integrate_symbolic_required_positive_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )
        .into_iter()
        .map(crate::ImplicitCondition::Positive);

    Some(required_nonzero.chain(required_positive).collect())
}

fn reciprocal_trig_derivative_target_diff_matches(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    divisor: ExprId,
    right: ExprId,
) -> Option<bool> {
    let (builtin, trig_arg, target_scale) = reciprocal_trig_derivative_primitive(ctx, target)?;
    let divisor_scale = cas_math::numeric_eval::as_rational_const(ctx, divisor)?;
    if divisor_scale.is_zero() {
        return None;
    }

    let derivative_sign = reciprocal_trig_derivative_sign(builtin)?;
    let expected_rational_scale = target_scale * derivative_sign / divisor_scale;
    let actual_scale = reciprocal_trig_derivative_target_scale_expr(ctx, right, builtin, trig_arg)?;
    polynomial_derivative_scale_matches(
        ctx,
        actual_scale,
        trig_arg,
        var_name,
        &expected_rational_scale,
    )
}

fn integrated_reciprocal_trig_derivative_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    let mut condition_target = integrate_call.target;
    let antiderivative = if let Some(antiderivative) =
        cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        ) {
        antiderivative
    } else {
        let normalized = reciprocal_trig_derivative_integrand_quotient(ctx, integrate_call.target)?;
        condition_target = normalized;
        cas_math::symbolic_integration_support::integrate_symbolic_expr(
            ctx,
            normalized,
            &integrate_call.var_name,
        )?
    };
    let matched = reciprocal_trig_derivative_target_diff_matches(
        ctx,
        antiderivative,
        &diff_call.var_name,
        divisor,
        right,
    )?;
    if !matched {
        return None;
    }

    let required_nonzero =
        cas_math::symbolic_integration_support::integrate_symbolic_required_nonzero_conditions(
            ctx,
            condition_target,
            &integrate_call.var_name,
        )
        .into_iter()
        .map(crate::ImplicitCondition::NonZero);
    let required_positive =
        cas_math::symbolic_integration_support::integrate_symbolic_required_positive_conditions(
            ctx,
            condition_target,
            &integrate_call.var_name,
        )
        .into_iter()
        .map(crate::ImplicitCondition::Positive);

    Some(required_nonzero.chain(required_positive).collect())
}

fn log_abs_plain_trig_target_diff_matches(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    divisor: ExprId,
    right: ExprId,
) -> Option<bool> {
    let (target_scale, inner_expr) = scaled_ln_abs_inner(ctx, target)?;
    let (builtin, trig_arg, primitive_sign) = plain_trig_log_abs_primitive(ctx, inner_expr)?;

    let divisor_scale = cas_math::numeric_eval::as_rational_const(ctx, divisor)?;
    if divisor_scale.is_zero() {
        return None;
    }

    let expected_rational_scale = target_scale * primitive_sign / divisor_scale;
    let actual_scale = plain_trig_target_scale_expr(ctx, right, builtin, trig_arg)?;
    if let Some(matched) = polynomial_derivative_scale_matches(
        ctx,
        actual_scale,
        trig_arg,
        var_name,
        &expected_rational_scale,
    ) {
        return Some(matched);
    }

    let arg_derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        ctx, trig_arg, var_name,
    )?;
    let expected_scale = scale_expr_by_rational(ctx, arg_derivative, expected_rational_scale);
    Some(exprs_equivalent(ctx, actual_scale, expected_scale))
}

fn integrated_log_abs_plain_trig_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    let antiderivative = cas_math::symbolic_integration_support::integrate_symbolic_expr(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    )?;
    let matched = log_abs_plain_trig_target_diff_matches(
        ctx,
        antiderivative,
        &diff_call.var_name,
        divisor,
        right,
    )?;
    if !matched {
        return None;
    }

    let required_nonzero =
        cas_math::symbolic_integration_support::integrate_symbolic_required_nonzero_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )
        .into_iter()
        .map(crate::ImplicitCondition::NonZero);
    let required_positive =
        cas_math::symbolic_integration_support::integrate_symbolic_required_positive_conditions(
            ctx,
            integrate_call.target,
            &integrate_call.var_name,
        )
        .into_iter()
        .map(crate::ImplicitCondition::Positive);

    Some(required_nonzero.chain(required_positive).collect())
}

pub(crate) fn try_diff_reciprocal_trig_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    log_abs_reciprocal_trig_diff_matches(ctx, diff_expr, divisor, right)
        .filter(|matched| *matched)
        .map(|_| ctx.num(0))
}

pub(crate) fn try_diff_integral_reciprocal_trig_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions =
        integrated_log_abs_reciprocal_trig_diff_matches(ctx, diff_expr, divisor, right).or_else(
            || integrated_reciprocal_trig_derivative_diff_matches(ctx, diff_expr, divisor, right),
        )?;
    Some((ctx.num(0), required_conditions))
}

pub(crate) fn try_diff_integral_plain_trig_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions =
        integrated_log_abs_plain_trig_diff_matches(ctx, diff_expr, divisor, right)?;
    Some((ctx.num(0), required_conditions))
}

fn constant_scaled_hyperbolic_reciprocal_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<bool> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    if !is_constant_scaled_hyperbolic_reciprocal_target(ctx, call.target) {
        return None;
    }

    let derivative = cas_math::symbolic_differentiation_support::differentiate_symbolic_expr(
        ctx,
        call.target,
        &call.var_name,
    )?;
    let expected = if expr_is_one(ctx, divisor) {
        derivative
    } else {
        ctx.add(Expr::Div(derivative, divisor))
    };

    Some(expr_eq(ctx, expected, right))
}

pub(crate) fn try_diff_hyperbolic_reciprocal_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    constant_scaled_hyperbolic_reciprocal_diff_matches(ctx, diff_expr, divisor, right)
        .filter(|matched| *matched)
        .map(|_| ctx.num(0))
}

pub(crate) fn try_diff_hyperbolic_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    try_diff_log_abs_hyperbolic_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_hyperbolic_reciprocal_residual_zero_preorder(ctx, left, right))
}

pub(crate) fn try_diff_hyperbolic_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_diff_hyperbolic_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_hyperbolic_residual_zero_preorder(ctx, right, left))
}

pub(crate) fn try_diff_reciprocal_trig_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<ExprId> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_diff_reciprocal_trig_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_reciprocal_trig_residual_zero_preorder(ctx, right, left))
}

pub(crate) fn try_diff_integral_reciprocal_trig_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_wrapped_root_zero(
        ctx,
        expr,
        3,
        try_diff_integral_reciprocal_trig_residual_direct_root_zero,
    )
}

fn try_diff_integral_reciprocal_trig_residual_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_diff_integral_reciprocal_trig_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_integral_reciprocal_trig_residual_zero_preorder(ctx, right, left))
}

type IntegralResidualRootResult = Option<(ExprId, Vec<crate::ImplicitCondition>)>;
type IntegralResidualRootMatcher = fn(&mut Context, ExprId) -> IntegralResidualRootResult;

fn try_diff_integral_residual_wrapped_root_zero(
    ctx: &mut Context,
    expr: ExprId,
    depth: u8,
    direct_root_zero: IntegralResidualRootMatcher,
) -> IntegralResidualRootResult {
    if let Some(result) = direct_root_zero(ctx, expr) {
        return Some(result);
    }

    if depth == 0 {
        return None;
    }

    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            if is_zero_constant(ctx, left) {
                try_diff_integral_residual_wrapped_root_zero(
                    ctx,
                    right,
                    depth - 1,
                    direct_root_zero,
                )
            } else if is_zero_constant(ctx, right) {
                try_diff_integral_residual_wrapped_root_zero(ctx, left, depth - 1, direct_root_zero)
            } else {
                None
            }
        }
        Expr::Sub(left, right) => {
            if is_zero_constant(ctx, right) {
                try_diff_integral_residual_wrapped_root_zero(ctx, left, depth - 1, direct_root_zero)
            } else if is_zero_constant(ctx, left) {
                try_diff_integral_residual_wrapped_root_zero(
                    ctx,
                    right,
                    depth - 1,
                    direct_root_zero,
                )
            } else {
                None
            }
        }
        Expr::Neg(inner) => {
            try_diff_integral_residual_wrapped_root_zero(ctx, inner, depth - 1, direct_root_zero)
        }
        Expr::Mul(left, right) => {
            if is_nonzero_constant(ctx, left) {
                try_diff_integral_residual_wrapped_root_zero(
                    ctx,
                    right,
                    depth - 1,
                    direct_root_zero,
                )
            } else if is_nonzero_constant(ctx, right) {
                try_diff_integral_residual_wrapped_root_zero(ctx, left, depth - 1, direct_root_zero)
            } else {
                None
            }
        }
        Expr::Div(num, den) => {
            if cas_math::numeric_eval::as_rational_const(ctx, den)
                .is_some_and(|value| value.is_zero())
            {
                return None;
            }
            let (zero, mut required_conditions) = try_diff_integral_residual_wrapped_root_zero(
                ctx,
                num,
                depth - 1,
                direct_root_zero,
            )?;
            if cas_math::numeric_eval::as_rational_const(ctx, den).is_none() {
                required_conditions.push(crate::ImplicitCondition::NonZero(den));
            }
            Some((zero, required_conditions))
        }
        _ => None,
    }
}

pub(crate) fn try_diff_integral_plain_trig_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_wrapped_root_zero(
        ctx,
        expr,
        3,
        try_diff_integral_plain_trig_residual_direct_root_zero,
    )
}

fn try_diff_integral_plain_trig_residual_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_diff_integral_plain_trig_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_integral_plain_trig_residual_zero_preorder(ctx, right, left))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn render(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    fn root_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_hyperbolic_residual_root_zero(&mut ctx, expr).map(|result| render(&ctx, result))
    }

    fn reciprocal_trig_root_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_reciprocal_trig_residual_root_zero(&mut ctx, expr)
            .map(|result| render(&ctx, result))
    }

    fn integral_reciprocal_trig_root_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_reciprocal_trig_residual_root_zero(&mut ctx, expr)
            .map(|(result, _required_conditions)| render(&ctx, result))
    }

    fn integral_plain_trig_root_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_plain_trig_residual_root_zero(&mut ctx, expr)
            .map(|(result, _required_conditions)| render(&ctx, result))
    }

    fn simplify_text(input: &str) -> String {
        let mut simplifier = crate::engine::Simplifier::new();
        let expr = parse(input, &mut simplifier.context)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        let (result, _steps) = simplifier.simplify(expr);
        render(&simplifier.context, result)
    }

    #[test]
    fn diff_log_abs_hyperbolic_residual_root_cancels_compact_forms() {
        let cases = [
            "diff(ln(abs(sinh(2*x+1))), x)/2 - 1/tanh(2*x+1)",
            "diff(ln(abs(sinh(2*x+1))), x)/2 - cosh(2*x+1)/sinh(2*x+1)",
            "diff(ln(abs(cosh(2*x+1))), x)/2 - tanh(2*x+1)",
            "diff(ln(abs(cosh(2*x+1))), x)/2 - sinh(2*x+1)/cosh(2*x+1)",
        ];

        for input in cases {
            assert_eq!(
                root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_log_abs_reciprocal_trig_residual_root_cancels_compact_forms() {
        let cases = [
            "diff(ln(abs((sin(2*x+1)+1)/cos(2*x+1))), x)/2 - sec(2*x+1)",
            "diff(ln(abs((sin(2*x+1)+1)/cos(2*x+1))), x)/2 - 1/cos(2*x+1)",
            "diff(ln(abs((cos(2*x+1)-1)/sin(2*x+1))), x)/2 - csc(2*x+1)",
            "diff(ln(abs((cos(2*x+1)-1)/sin(2*x+1))), x)/2 - 1/sin(2*x+1)",
            "diff(-2/3*ln(abs((cos((2-3*x)/2)-1)/sin((2-3*x)/2))), x) - csc((2-3*x)/2)",
            "diff(-2/3*ln(abs((cos((2-3*x)/2)-1)/sin((2-3*x)/2))), x) - 1/sin((2-3*x)/2)",
        ];

        for input in cases {
            assert_eq!(
                reciprocal_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_reciprocal_trig_residual_root_verifies_antiderivative_first() {
        let cases = [
            "diff(integrate(sec((3*x+2)/2), x), x) - sec((3*x+2)/2)",
            "diff(integrate(1/cos((3*x+2)/2), x), x) - 1/cos((3*x+2)/2)",
            "diff(integrate(csc((2-3*x)/2), x), x) - csc((2-3*x)/2)",
            "diff(integrate(csc((2-3*x)/2), x), x) - 1/sin((2-3*x)/2)",
            "diff(integrate(1/sin((2-3*x)/2), x), x) - 1/sin((2-3*x)/2)",
            "diff(integrate(2*x*sec(x^2)*tan(x^2), x), x) - 2*x*sec(x^2)*tan(x^2)",
            "diff(integrate(2*x*csc(x^2)*cot(x^2), x), x) - 2*x*csc(x^2)*cot(x^2)",
            "diff(integrate((4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2)",
            "diff(integrate((4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2)",
        ];

        for input in cases {
            assert_eq!(
                integral_reciprocal_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }

        assert_eq!(
            integral_reciprocal_trig_root_residual_result(
                "diff(integrate(csc((2-3*x)/2), x), y) - csc((2-3*x)/2)"
            ),
            None
        );
    }

    #[test]
    fn diff_integral_reciprocal_trig_residual_root_cancels_algebraic_wrappers() {
        let sec_residual = "diff(integrate((4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*sec(x^4-x^2)*tan(x^4-x^2)";
        let csc_residual = "diff(integrate((4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*csc(x^4-x^2)*cot(x^4-x^2)";
        let cases = [
            format!("({sec_residual}) + 0"),
            format!("0 - ({sec_residual})"),
            format!("2*({sec_residual})"),
            format!("({sec_residual})/(x+1)"),
            format!("-({csc_residual})"),
            format!("(3/2)*(({csc_residual}) + 0)"),
        ];

        for input in cases {
            assert_eq!(
                integral_reciprocal_trig_root_residual_result(&input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(&input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_cancels_algebraic_wrappers() {
        let tan_residual =
            "diff(integrate((4*x^3-2*x)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*tan(x^4-x^2)";
        let cot_residual =
            "diff(integrate((4*x^3-2*x)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*cot(x^4-x^2)";
        let cases = [
            format!("({tan_residual}) + 0"),
            format!("0 - ({tan_residual})"),
            format!("2*({tan_residual})"),
            format!("({tan_residual})/(x+1)"),
            format!("-({cot_residual})"),
            format!("(3/2)*(({cot_residual}) + 0)"),
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(&input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(&input), "0", "{input}");
        }
    }

    #[test]
    fn diff_integral_plain_trig_residual_root_verifies_antiderivative_first() {
        let cases = [
            "diff(integrate(tan(2*x+1), x), x) - tan(2*x+1)",
            "diff(integrate(sin(2*x+1)/cos(2*x+1), x), x) - sin(2*x+1)/cos(2*x+1)",
            "diff(integrate(cot(2*x+1), x), x) - cot(2*x+1)",
            "diff(integrate(cos(2*x+1)/sin(2*x+1), x), x) - cos(2*x+1)/sin(2*x+1)",
            "diff(integrate(2*x*tan(x^2), x), x) - 2*x*tan(x^2)",
            "diff(integrate(3*x^2*cot(x^3), x), x) - 3*x^2*cot(x^3)",
            "diff(integrate((4*x^3-2*x)*tan(x^4-x^2), x), x) - (4*x^3-2*x)*tan(x^4-x^2)",
            "diff(integrate((4*x^3-2*x)*cot(x^4-x^2), x), x) - (4*x^3-2*x)*cot(x^4-x^2)",
        ];

        for input in cases {
            assert_eq!(
                integral_plain_trig_root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }

        assert_eq!(
            integral_plain_trig_root_residual_result(
                "diff(integrate(cot(2*x+1), x), y) - cot(2*x+1)"
            ),
            None
        );
    }

    #[test]
    fn diff_hyperbolic_reciprocal_residual_root_cancels_compact_forms() {
        let cases = [
            "diff(-1/(2*cosh(2*x+1)), x) - sinh(2*x+1)/cosh(2*x+1)^2",
            "diff(-1/(2*sinh(2*x+1)), x) - cosh(2*x+1)/sinh(2*x+1)^2",
            "sinh(2*x+1)/cosh(2*x+1)^2 - diff(-1/(2*cosh(2*x+1)), x)",
            "cosh(2*x+1)/sinh(2*x+1)^2 - diff(-1/(2*sinh(2*x+1)), x)",
        ];

        for input in cases {
            assert_eq!(
                root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
        }
    }
}
