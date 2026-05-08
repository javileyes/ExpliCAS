use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::expr_domain::exprs_equivalent;
use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Zero};

fn expr_eq(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    cas_ast::ordering::compare_expr(ctx, left, right) == std::cmp::Ordering::Equal
}

fn exprs_match(ctx: &Context, left: ExprId, right: ExprId) -> bool {
    if expr_eq(ctx, left, right) || exprs_equivalent(ctx, left, right) {
        return true;
    }

    let Some(left_base) = reciprocal_sqrt_like_base(ctx, left) else {
        return false;
    };
    let Some(right_base) = reciprocal_sqrt_like_base(ctx, right) else {
        return false;
    };

    expr_eq(ctx, left_base, right_base) || exprs_equivalent(ctx, left_base, right_base)
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

fn is_positive_half(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(value) => *value == BigRational::new(1.into(), 2.into()),
        _ => false,
    }
}

fn is_negative_half(ctx: &Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Number(value) => *value == BigRational::new((-1).into(), 2.into()),
        Expr::Neg(inner) => is_positive_half(ctx, *inner),
        _ => false,
    }
}

fn sqrt_like_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Sqrt) =>
        {
            Some(args[0])
        }
        Expr::Pow(base, exp) if is_positive_half(ctx, *exp) => Some(*base),
        _ => None,
    }
}

fn reciprocal_sqrt_like_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Pow(base, exp) if is_negative_half(ctx, *exp) => Some(*base),
        Expr::Div(num, den) if is_one_constant(ctx, *num) => sqrt_like_base(ctx, *den),
        _ => None,
    }
}

fn scaled_reciprocal_sqrt_coeff(
    ctx: &mut Context,
    expr: ExprId,
    expected_base: ExprId,
) -> Option<BigRational> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    if let Some(base) = reciprocal_sqrt_like_base(ctx, expr) {
        return exprs_match(ctx, base, expected_base).then_some(BigRational::one());
    }

    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            scaled_reciprocal_sqrt_coeff(ctx, inner, expected_base).map(|coeff| -coeff)
        }
        Expr::Div(num, den) => {
            let den = cas_ast::hold::strip_all_holds(ctx, den);
            if let Some(base) = sqrt_like_base(ctx, den) {
                if exprs_match(ctx, base, expected_base) {
                    return cas_math::numeric_eval::as_rational_const(ctx, num);
                }
            }

            let numerator_coeff = cas_math::numeric_eval::as_rational_const(ctx, num)?;
            let mut denominator_coeff = BigRational::one();
            let mut matched_sqrt = false;
            for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
                if let Some(factor_coeff) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    denominator_coeff *= factor_coeff;
                    continue;
                }

                if !matched_sqrt {
                    if let Some(base) = sqrt_like_base(ctx, factor) {
                        if exprs_match(ctx, base, expected_base) {
                            matched_sqrt = true;
                            continue;
                        }
                    }
                }

                return None;
            }

            (matched_sqrt && !denominator_coeff.is_zero())
                .then_some(numerator_coeff / denominator_coeff)
        }
        Expr::Mul(_, _) => {
            let mut coeff = BigRational::one();
            let mut matched_reciprocal = false;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(factor_coeff) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    coeff *= factor_coeff;
                    continue;
                }

                if !matched_reciprocal {
                    if let Some(base) = reciprocal_sqrt_like_base(ctx, factor) {
                        if exprs_match(ctx, base, expected_base) {
                            matched_reciprocal = true;
                            continue;
                        }
                    }
                }

                return None;
            }

            matched_reciprocal.then_some(coeff)
        }
        _ => None,
    }
}

fn sqrt_chain_derivative_scale_matches(
    ctx: &mut Context,
    actual_scale: ExprId,
    sqrt_arg: ExprId,
    var_name: &str,
) -> Option<bool> {
    let sqrt_arg = cas_ast::hold::strip_all_holds(ctx, sqrt_arg);
    let base = sqrt_like_base(ctx, sqrt_arg)?;
    let base_poly = Polynomial::from_expr(ctx, base, var_name).ok()?;
    let derivative = base_poly.derivative();
    if derivative.is_zero() || derivative.coeffs.len() != 1 {
        return None;
    }

    let expected_coeff = derivative.coeffs[0].clone() / BigRational::from_integer(2.into());
    let actual_coeff = scaled_reciprocal_sqrt_coeff(ctx, actual_scale, base)?;
    Some(actual_coeff == expected_coeff)
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

fn square_expr(ctx: &mut Context, expr: ExprId) -> ExprId {
    let two = ctx.num(2);
    ctx.add(Expr::Pow(expr, two))
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

fn scaled_inverse_reciprocal_trig_target(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BuiltinFn, ExprId, BigRational)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args) => {
            let builtin = match ctx.builtin_of(*fn_id) {
                Some(
                    builtin @ (BuiltinFn::Arcsec
                    | BuiltinFn::Asec
                    | BuiltinFn::Arccsc
                    | BuiltinFn::Acsc),
                ) => builtin,
                _ => return None,
            };
            (args.len() == 1).then_some((builtin, args[0], BigRational::one()))
        }
        Expr::Neg(inner) => {
            let (builtin, arg, scale) = scaled_inverse_reciprocal_trig_target(ctx, *inner)?;
            Some((builtin, arg, -scale))
        }
        Expr::Div(num, den) => {
            let den_scale = cas_math::numeric_eval::as_rational_const(ctx, *den)?;
            if den_scale.is_zero() {
                return None;
            }
            let (builtin, arg, scale) = scaled_inverse_reciprocal_trig_target(ctx, *num)?;
            Some((builtin, arg, scale / den_scale))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut matched = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    scale *= value;
                    continue;
                }

                let (builtin, arg, factor_scale) =
                    scaled_inverse_reciprocal_trig_target(ctx, factor)?;
                if matched.replace((builtin, arg)).is_some() {
                    return None;
                }
                scale *= factor_scale;
            }

            let (builtin, arg) = matched?;
            Some((builtin, arg, scale))
        }
        _ => None,
    }
}

fn inverse_reciprocal_trig_derivative_sign(builtin: BuiltinFn) -> Option<BigRational> {
    match builtin {
        BuiltinFn::Arcsec | BuiltinFn::Asec => Some(BigRational::one()),
        BuiltinFn::Arccsc | BuiltinFn::Acsc => Some(-BigRational::one()),
        _ => None,
    }
}

fn abs_arg_matches(ctx: &Context, expr: ExprId, arg: ExprId) -> bool {
    unary_builtin_arg(ctx, expr, BuiltinFn::Abs)
        .is_some_and(|abs_arg| exprs_equivalent(ctx, abs_arg, arg))
}

fn inverse_reciprocal_trig_abs_sqrt_coeff(
    ctx: &mut Context,
    expr: ExprId,
    arg: ExprId,
    gap: ExprId,
) -> Option<BigRational> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => {
            inverse_reciprocal_trig_abs_sqrt_coeff(ctx, inner, arg, gap).map(|coeff| -coeff)
        }
        Expr::Div(num, den) => {
            let numerator_coeff = cas_math::numeric_eval::as_rational_const(ctx, num)?;
            let mut denominator_coeff = BigRational::one();
            let mut matched_abs = false;
            let mut matched_sqrt = false;

            for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
                if let Some(factor_coeff) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    denominator_coeff *= factor_coeff;
                    continue;
                }

                if !matched_abs && abs_arg_matches(ctx, factor, arg) {
                    matched_abs = true;
                    continue;
                }

                if !matched_sqrt {
                    if let Some(base) = sqrt_like_base(ctx, factor) {
                        if exprs_match(ctx, base, gap) {
                            matched_sqrt = true;
                            continue;
                        }
                    }
                }

                return None;
            }

            (matched_abs && matched_sqrt && !denominator_coeff.is_zero())
                .then_some(numerator_coeff / denominator_coeff)
        }
        _ => None,
    }
}

fn inverse_reciprocal_trig_positive_arg_sqrt_coeff(
    ctx: &mut Context,
    expr: ExprId,
    arg: ExprId,
    gap: ExprId,
    arg_derivative: &Polynomial,
    var_name: &str,
) -> Option<BigRational> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => inverse_reciprocal_trig_positive_arg_sqrt_coeff(
            ctx,
            inner,
            arg,
            gap,
            arg_derivative,
            var_name,
        )
        .map(|coeff| -coeff),
        Expr::Div(num, den) => {
            let actual_num_poly = Polynomial::from_expr(ctx, num, var_name).ok()?;
            let mut denominator_coeff = BigRational::one();
            let mut matched_arg = false;
            let mut matched_sqrt = false;

            for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
                if let Some(factor_coeff) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
                    denominator_coeff *= factor_coeff;
                    continue;
                }

                if !matched_arg && exprs_match(ctx, factor, arg) {
                    matched_arg = true;
                    continue;
                }

                if !matched_sqrt {
                    if let Some(base) = sqrt_like_base(ctx, factor) {
                        if exprs_match(ctx, base, gap) {
                            matched_sqrt = true;
                            continue;
                        }
                    }
                }

                return None;
            }

            if !matched_arg || !matched_sqrt || denominator_coeff.is_zero() {
                return None;
            }

            let coeff = polynomial_scale_factor(arg_derivative, &actual_num_poly)?;
            Some(coeff / denominator_coeff)
        }
        _ => None,
    }
}

fn polynomial_scale_factor(expected: &Polynomial, actual: &Polynomial) -> Option<BigRational> {
    if expected.is_zero() {
        return actual.is_zero().then_some(BigRational::one());
    }
    if expected.degree() != actual.degree() {
        return None;
    }

    let mut scale = None;
    for (expected_coeff, actual_coeff) in expected.coeffs.iter().zip(actual.coeffs.iter()) {
        if expected_coeff.is_zero() {
            if !actual_coeff.is_zero() {
                return None;
            }
            continue;
        }
        let candidate = actual_coeff / expected_coeff;
        if scale.as_ref().is_some_and(|scale| scale != &candidate) {
            return None;
        }
        scale = Some(candidate);
    }

    scale
}

fn quadratic_polynomial_is_strictly_greater_than_one(poly: &Polynomial) -> bool {
    if poly.degree() != 2 {
        return false;
    }

    let a = poly
        .coeffs
        .get(2)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    if a <= BigRational::zero() {
        return false;
    }
    let b = poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let c = poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let four = BigRational::from_integer(4.into());
    let minimum = c - (b.clone() * b) / (four * a);
    minimum > BigRational::one()
}

fn inverse_reciprocal_trig_positive_quadratic_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (builtin, arg, target_scale) = scaled_inverse_reciprocal_trig_target(ctx, call.target)?;
    let arg_poly = Polynomial::from_expr(ctx, arg, &call.var_name).ok()?;
    if !quadratic_polynomial_is_strictly_greater_than_one(&arg_poly) {
        return None;
    }

    let derivative = arg_poly.derivative();
    if derivative.is_zero() {
        return None;
    }

    let divisor_scale = cas_math::numeric_eval::as_rational_const(ctx, divisor)?;
    if divisor_scale.is_zero() {
        return None;
    }

    let sign = inverse_reciprocal_trig_derivative_sign(builtin)?;
    let expected_coeff = sign * target_scale / divisor_scale;
    let one = ctx.num(1);
    let arg_sq = square_expr(ctx, arg);
    let raw_gap = ctx.add(Expr::Sub(arg_sq, one));
    let gap = cas_math::expr_normalization::normalize_condition_expr(ctx, raw_gap);
    let actual_coeff = inverse_reciprocal_trig_positive_arg_sqrt_coeff(
        ctx,
        right,
        arg,
        gap,
        &derivative,
        &call.var_name,
    )?;
    (actual_coeff == expected_coeff).then_some(Vec::new())
}

fn inverse_reciprocal_trig_sqrt_polynomial_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (builtin, arg, target_scale) = scaled_inverse_reciprocal_trig_target(ctx, call.target)?;
    let base = sqrt_like_base(ctx, arg)?;
    let base_poly = Polynomial::from_expr(ctx, base, &call.var_name).ok()?;
    let derivative = base_poly.derivative();
    if derivative.is_zero() {
        return None;
    }

    let divisor_scale = cas_math::numeric_eval::as_rational_const(ctx, divisor)?;
    if divisor_scale.is_zero() {
        return None;
    }

    let sign = inverse_reciprocal_trig_derivative_sign(builtin)?;
    let two = BigRational::from_integer(2.into());
    let expected_coeff = sign * target_scale / (two * divisor_scale);
    let gap_poly = base_poly.sub(&Polynomial::one(call.var_name.clone()));
    let gap = gap_poly.to_expr(ctx);
    let actual_coeff = inverse_reciprocal_trig_positive_arg_sqrt_coeff(
        ctx,
        right,
        base,
        gap,
        &derivative,
        &call.var_name,
    )?;
    (actual_coeff == expected_coeff).then_some(vec![crate::ImplicitCondition::Positive(gap)])
}

fn inverse_reciprocal_trig_affine_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let (builtin, arg, target_scale) = scaled_inverse_reciprocal_trig_target(ctx, call.target)?;
    let arg_poly = Polynomial::from_expr(ctx, arg, &call.var_name).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }

    let derivative = arg_poly.derivative();
    let derivative_scale = derivative.coeffs.first().cloned()?;
    if derivative_scale.is_zero() {
        return None;
    }

    let divisor_scale = cas_math::numeric_eval::as_rational_const(ctx, divisor)?;
    if divisor_scale.is_zero() {
        return None;
    }

    let sign = inverse_reciprocal_trig_derivative_sign(builtin)?;
    let expected_coeff = sign * target_scale * derivative_scale / divisor_scale;
    let one = ctx.num(1);
    let arg_sq = square_expr(ctx, arg);
    let raw_gap = ctx.add(Expr::Sub(arg_sq, one));
    let gap = cas_math::expr_normalization::normalize_condition_expr(ctx, raw_gap);
    let actual_coeff = inverse_reciprocal_trig_abs_sqrt_coeff(ctx, right, arg, gap)?;
    (actual_coeff == expected_coeff).then_some(vec![crate::ImplicitCondition::Positive(gap)])
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
            if let Some((target_arg, target_scale)) = tanh_scaled_target(ctx, right) {
                let target_arg = cas_ast::hold::strip_all_holds(ctx, target_arg);
                let target_scale = cas_ast::hold::strip_all_holds(ctx, target_scale);
                let hyperbolic_arg = cas_ast::hold::strip_all_holds(ctx, hyperbolic_arg);
                let scale = cas_ast::hold::strip_all_holds(ctx, scale);
                if exprs_match(ctx, target_arg, hyperbolic_arg)
                    && exprs_match(ctx, target_scale, scale)
                {
                    return true;
                }
            }

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

    let (builtin, hyperbolic_arg) =
        if let Some(abs_arg) = unary_builtin_arg(ctx, ln_arg, BuiltinFn::Abs) {
            if let Some(arg) = unary_builtin_arg(ctx, abs_arg, BuiltinFn::Sinh) {
                (BuiltinFn::Sinh, arg)
            } else {
                (
                    BuiltinFn::Cosh,
                    unary_builtin_arg(ctx, abs_arg, BuiltinFn::Cosh)?,
                )
            }
        } else {
            (
                BuiltinFn::Cosh,
                unary_builtin_arg(ctx, ln_arg, BuiltinFn::Cosh)?,
            )
        };

    if builtin == BuiltinFn::Cosh {
        if let Some((target_arg, target_scale)) = tanh_scaled_target(ctx, right) {
            if exprs_match(ctx, target_arg, hyperbolic_arg)
                && sqrt_chain_derivative_scale_matches(
                    ctx,
                    target_scale,
                    hyperbolic_arg,
                    &call.var_name,
                )
                .is_some_and(|matched| matched)
            {
                return Some(true);
            }
        }
    }

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

fn sinh_over_cosh_scaled_target(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);

    if let Expr::Mul(_, _) = ctx.get(expr) {
        let mut matched = None;
        let mut scale_factors = Vec::new();
        for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
            let factor = cas_ast::hold::strip_all_holds(ctx, factor);
            if matched.is_none() {
                if let Some((arg, scale)) = sinh_over_cosh_scaled_target(ctx, factor) {
                    matched = Some(arg);
                    if !expr_is_one(ctx, scale) {
                        scale_factors.push(scale);
                    }
                    continue;
                }
            }

            scale_factors.push(factor);
        }

        let arg = matched?;
        let scale = if scale_factors.is_empty() {
            ctx.num(1)
        } else {
            cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors)
        };
        return Some((arg, scale));
    }

    let Expr::Div(num, den) = ctx.get(expr).clone() else {
        return None;
    };
    let den = cas_ast::hold::strip_all_holds(ctx, den);
    let hyperbolic_arg = unary_builtin_arg(ctx, den, BuiltinFn::Cosh)?;

    let mut matched_sinh = false;
    let mut scale_factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, num) {
        let factor = cas_ast::hold::strip_all_holds(ctx, factor);
        if !matched_sinh {
            if let Some(sinh_arg) = unary_builtin_arg(ctx, factor, BuiltinFn::Sinh) {
                if exprs_match(ctx, sinh_arg, hyperbolic_arg) {
                    matched_sinh = true;
                    continue;
                }
            }
        }

        scale_factors.push(factor);
    }

    if !matched_sinh {
        return None;
    }

    let scale = if scale_factors.is_empty() {
        ctx.num(1)
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors)
    };
    Some((hyperbolic_arg, scale))
}

fn tanh_scaled_target(ctx: &mut Context, expr: ExprId) -> Option<(ExprId, ExprId)> {
    let expr = cas_ast::hold::strip_all_holds(ctx, expr);
    if let Some(arg) = unary_builtin_arg(ctx, expr, BuiltinFn::Tanh) {
        return Some((arg, ctx.num(1)));
    }

    if let Expr::Div(num, den) = ctx.get(expr).clone() {
        let (arg, num_scale) = tanh_scaled_target(ctx, num)?;
        let den = cas_ast::hold::strip_all_holds(ctx, den);
        if expr_is_one(ctx, den) {
            return Some((arg, num_scale));
        }
        let scale = if expr_is_one(ctx, num_scale) {
            let one = ctx.num(1);
            ctx.add(Expr::Div(one, den))
        } else {
            ctx.add(Expr::Div(num_scale, den))
        };
        return Some((arg, scale));
    }

    let mut matched_arg = None;
    let mut scale_factors = Vec::new();
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        let factor = cas_ast::hold::strip_all_holds(ctx, factor);
        if matched_arg.is_none() {
            if let Some(arg) = unary_builtin_arg(ctx, factor, BuiltinFn::Tanh) {
                matched_arg = Some(arg);
                continue;
            }
        }

        scale_factors.push(factor);
    }

    let arg = matched_arg?;
    let scale = if scale_factors.is_empty() {
        ctx.num(1)
    } else {
        cas_math::expr_nary::build_balanced_mul(ctx, &scale_factors)
    };
    Some((arg, scale))
}

fn hyperbolic_tanh_common_factor_residual_matches(
    ctx: &mut Context,
    quotient_expr: ExprId,
    target_expr: ExprId,
) -> Option<bool> {
    let (quotient_arg, quotient_scale) = sinh_over_cosh_scaled_target(ctx, quotient_expr)?;
    let (target_arg, target_scale) = tanh_scaled_target(ctx, target_expr)?;
    let quotient_arg = cas_ast::hold::strip_all_holds(ctx, quotient_arg);
    let quotient_scale = cas_ast::hold::strip_all_holds(ctx, quotient_scale);
    let target_arg = cas_ast::hold::strip_all_holds(ctx, target_arg);
    let target_scale = cas_ast::hold::strip_all_holds(ctx, target_scale);

    Some(
        exprs_match(ctx, quotient_arg, target_arg)
            && exprs_match(ctx, quotient_scale, target_scale),
    )
}

fn try_hyperbolic_tanh_common_factor_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<ExprId> {
    hyperbolic_tanh_common_factor_residual_matches(ctx, left, right)
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

fn integrated_quadratic_exp_linear_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    divisor: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if !expr_is_one(ctx, divisor) {
        return None;
    }

    let diff_call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let integrate_call =
        crate::symbolic_calculus_call_support::try_extract_integrate_call(ctx, diff_call.target)?;
    if diff_call.var_name != integrate_call.var_name {
        return None;
    }

    if !cas_math::symbolic_integration_support::integrate_symbolic_is_quadratic_times_exp_linear_target(
        ctx,
        integrate_call.target,
        &integrate_call.var_name,
    ) {
        return None;
    }

    if !exprs_match(ctx, integrate_call.target, right) {
        return None;
    }

    Some(Vec::new())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConstantPassthroughOrientation {
    Add,
    LeadingSub,
}

struct ConstantPassthrough {
    constant: BigRational,
    core: ExprId,
    orientation: ConstantPassthroughOrientation,
}

fn negated_expr_core(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    match ctx.get(expr) {
        Expr::Neg(inner) => Some(*inner),
        Expr::Mul(left, right) => {
            let negative_one = -BigRational::one();
            if cas_math::numeric_eval::as_rational_const(ctx, *left)
                .is_some_and(|value| value == negative_one)
            {
                Some(*right)
            } else if cas_math::numeric_eval::as_rational_const(ctx, *right)
                .is_some_and(|value| value == negative_one)
            {
                Some(*left)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn strip_constant_passthrough(ctx: &Context, expr: ExprId) -> Option<ConstantPassthrough> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            if let Some(constant) = cas_math::numeric_eval::as_rational_const(ctx, *left) {
                if let Some(inner) = negated_expr_core(ctx, *right) {
                    return Some(ConstantPassthrough {
                        constant,
                        core: inner,
                        orientation: ConstantPassthroughOrientation::LeadingSub,
                    });
                }
                Some(ConstantPassthrough {
                    constant,
                    core: *right,
                    orientation: ConstantPassthroughOrientation::Add,
                })
            } else if let Some(constant) = cas_math::numeric_eval::as_rational_const(ctx, *right) {
                if let Some(inner) = negated_expr_core(ctx, *left) {
                    return Some(ConstantPassthrough {
                        constant,
                        core: inner,
                        orientation: ConstantPassthroughOrientation::LeadingSub,
                    });
                }
                Some(ConstantPassthrough {
                    constant,
                    core: *left,
                    orientation: ConstantPassthroughOrientation::Add,
                })
            } else {
                None
            }
        }
        Expr::Sub(left, right) => {
            let constant = cas_math::numeric_eval::as_rational_const(ctx, *left)?;
            Some(ConstantPassthrough {
                constant,
                core: *right,
                orientation: ConstantPassthroughOrientation::LeadingSub,
            })
        }
        _ => None,
    }
}

fn rational_expr(ctx: &mut Context, value: &BigRational) -> ExprId {
    ctx.add(Expr::Number(value.clone()))
}

fn build_constant_passthrough_expr(
    ctx: &mut Context,
    constant: &BigRational,
    core: ExprId,
    orientation: ConstantPassthroughOrientation,
) -> ExprId {
    let constant = rational_expr(ctx, constant);
    match orientation {
        ConstantPassthroughOrientation::Add => ctx.add(Expr::Add(constant, core)),
        ConstantPassthroughOrientation::LeadingSub => ctx.add(Expr::Sub(constant, core)),
    }
}

fn diff_inverse_reciprocal_trig_core_difference_conditions(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    try_diff_inverse_reciprocal_trig_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_inverse_reciprocal_trig_residual_zero_preorder(ctx, right, left))
        .map(|(_zero, required_conditions)| required_conditions)
}

fn diff_inverse_reciprocal_trig_core_sum_conditions(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let neg_right = ctx.add(Expr::Neg(right));
    diff_inverse_reciprocal_trig_core_difference_conditions(ctx, left, neg_right).or_else(|| {
        let neg_left = ctx.add(Expr::Neg(left));
        diff_inverse_reciprocal_trig_core_difference_conditions(ctx, right, neg_left)
    })
}

fn arctan_sqrt_positive_polynomial_quotient_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    right: ExprId,
) -> Option<()> {
    let compact =
        crate::rules::calculus::arctan_sqrt_positive_polynomial_quotient_derivative_for_diff_call(
            ctx, diff_expr,
        )?;
    let compact = cas_ast::hold::strip_all_holds(ctx, compact);
    (exprs_match(ctx, compact, right)
        || quotient_matches_with_unordered_products(ctx, compact, right))
    .then_some(())
}

fn arctan_sqrt_derivative_presentation_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let target = cas_ast::hold::strip_all_holds(ctx, call.target);
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    let compact =
        crate::rules::calculus::try_post_calculus_presentation(ctx, diff_expr, diff_expr)?;
    let compact = cas_ast::hold::strip_all_holds(ctx, compact);
    if !(exprs_match(ctx, compact, right)
        || quotient_matches_with_unordered_products(ctx, compact, right)
        || quotient_matches_with_polynomial_content_denominators(
            ctx,
            compact,
            right,
            &call.var_name,
        ))
    {
        return None;
    }

    Some(vec![crate::ImplicitCondition::Positive(radicand)])
}

fn unit_interval_bounded_inverse_trig_derivative_presentation_diff_matches(
    ctx: &mut Context,
    diff_expr: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let required_conditions =
        unit_interval_bounded_inverse_trig_derivative_presentation_conditions(ctx, diff_expr)?;
    let compact =
        crate::rules::calculus::try_post_calculus_presentation(ctx, diff_expr, diff_expr)?;
    let compact = cas_ast::hold::strip_all_holds(ctx, compact);
    if !(exprs_match(ctx, compact, right)
        || quotient_matches_with_unordered_products(ctx, compact, right))
    {
        return None;
    }

    Some(required_conditions)
}

fn unit_interval_bounded_inverse_trig_derivative_presentation_conditions(
    ctx: &mut Context,
    diff_expr: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let call = crate::symbolic_calculus_call_support::try_extract_diff_call(ctx, diff_expr)?;
    let mut target = cas_ast::hold::strip_all_holds(ctx, call.target);
    if let Expr::Div(numerator, denominator) = ctx.get(target).clone() {
        let denominator = cas_ast::views::as_rational_const(ctx, denominator, 8)?;
        if denominator.is_zero() {
            return None;
        }
        target = numerator;
    }

    let mut inverse_trig_arg = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, target) {
        if cas_ast::views::as_rational_const(ctx, factor, 8).is_some() {
            continue;
        }

        let Expr::Function(fn_id, args) = ctx.get(factor).clone() else {
            return None;
        };
        if args.len() != 1
            || !matches!(
                ctx.builtin_of(fn_id),
                Some(BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arccos | BuiltinFn::Acos)
            )
            || inverse_trig_arg.replace(args[0]).is_some()
        {
            return None;
        }
    }

    let arg = inverse_trig_arg?;
    let arg_poly = Polynomial::from_expr(ctx, arg, &call.var_name).ok()?;
    if arg_poly.degree() != 1 {
        return None;
    }
    let offset = arg_poly
        .coeffs
        .first()
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let slope = arg_poly
        .coeffs
        .get(1)
        .cloned()
        .unwrap_or_else(BigRational::zero);
    let two = BigRational::from_integer(2.into());
    let unit_interval_arg = (offset == -BigRational::one() && slope == two)
        || (offset == BigRational::one() && slope == -two);
    if !unit_interval_arg {
        return None;
    }

    let var = ctx.var(&call.var_name);
    let one = ctx.num(1);
    let one_minus_var = ctx.add(Expr::Sub(one, var));
    Some(vec![
        crate::ImplicitCondition::Positive(var),
        crate::ImplicitCondition::Positive(one_minus_var),
    ])
}

fn arctan_sqrt_radicand_arg(ctx: &Context, target: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan | BuiltinFn::Acot | BuiltinFn::Arccot)
        )
    {
        return None;
    }
    sqrt_like_base(ctx, args[0])
}

fn quotient_matches_with_unordered_products(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> bool {
    let (left_num, left_den, right_num, right_den) = match (ctx.get(left), ctx.get(right)) {
        (Expr::Div(left_num, left_den), Expr::Div(right_num, right_den)) => {
            (*left_num, *left_den, *right_num, *right_den)
        }
        _ => return false,
    };

    exprs_match(ctx, left_num, right_num)
        && (unordered_product_factors_match(ctx, left_den, right_den)
            || compact_sqrt_one_plus_denominator_matches_expanded_sum(ctx, left_den, right_den)
            || compact_sqrt_one_plus_denominator_matches_expanded_sum(ctx, right_den, left_den))
}

fn unordered_product_factors_match(ctx: &mut Context, left: ExprId, right: ExprId) -> bool {
    let left_factors = cas_math::expr_nary::mul_leaves(ctx, left);
    let mut right_factors = cas_math::expr_nary::mul_leaves(ctx, right);
    if left_factors.len() != right_factors.len() {
        return false;
    }

    for left_factor in left_factors {
        let Some(pos) = right_factors
            .iter()
            .position(|right_factor| exprs_match(ctx, left_factor, *right_factor))
        else {
            return false;
        };
        right_factors.remove(pos);
    }
    true
}

fn quotient_matches_with_polynomial_content_denominators(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    let (left_num, left_den, right_num, right_den) = match (ctx.get(left), ctx.get(right)) {
        (Expr::Div(left_num, left_den), Expr::Div(right_num, right_den)) => {
            (*left_num, *left_den, *right_num, *right_den)
        }
        _ => return false,
    };

    if exprs_match(ctx, left_num, right_num)
        && unordered_product_factors_match_after_polynomial_content(
            ctx, left_den, right_den, var_name,
        )
    {
        return true;
    }

    let Some((left_coeff, left_factors)) =
        rational_quotient_parts_after_polynomial_content(ctx, left_num, left_den, var_name)
    else {
        return false;
    };
    let Some((right_coeff, right_factors)) =
        rational_quotient_parts_after_polynomial_content(ctx, right_num, right_den, var_name)
    else {
        return false;
    };
    left_coeff == right_coeff && unordered_factor_lists_match(ctx, left_factors, right_factors)
}

fn rational_quotient_parts_after_polynomial_content(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    var_name: &str,
) -> Option<(BigRational, Vec<ExprId>)> {
    let numerator = rational_const_for_matching(ctx, numerator)?;
    let (denominator_scale, denominator_factors) =
        product_factors_after_polynomial_content(ctx, denominator, var_name)?;
    if denominator_scale.is_zero() {
        return None;
    }
    Some((numerator / denominator_scale, denominator_factors))
}

fn rational_const_for_matching(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    if let Some(value) = cas_math::numeric_eval::as_rational_const(ctx, expr) {
        return Some(value);
    }
    match ctx.get(expr) {
        Expr::Neg(inner) => rational_const_for_matching(ctx, *inner).map(|value| -value),
        _ => None,
    }
}

fn unordered_product_factors_match_after_polynomial_content(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
    var_name: &str,
) -> bool {
    let Some((left_scale, left_factors)) =
        product_factors_after_polynomial_content(ctx, left, var_name)
    else {
        return false;
    };
    let Some((right_scale, right_factors)) =
        product_factors_after_polynomial_content(ctx, right, var_name)
    else {
        return false;
    };
    if left_scale != right_scale || left_factors.len() != right_factors.len() {
        return false;
    }

    unordered_factor_lists_match(ctx, left_factors, right_factors)
}

fn unordered_factor_lists_match(
    ctx: &mut Context,
    left_factors: Vec<ExprId>,
    right_factors: Vec<ExprId>,
) -> bool {
    if left_factors.len() != right_factors.len() {
        return false;
    }
    let mut unmatched_right = right_factors;
    for left_factor in left_factors {
        let Some(pos) = unmatched_right
            .iter()
            .position(|right_factor| exprs_match(ctx, left_factor, *right_factor))
        else {
            return false;
        };
        unmatched_right.remove(pos);
    }
    true
}

fn product_factors_after_polynomial_content(
    ctx: &mut Context,
    expr: ExprId,
    _var_name: &str,
) -> Option<(BigRational, Vec<ExprId>)> {
    let mut scale = BigRational::one();
    let mut factors = Vec::new();

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = rational_const_for_matching(ctx, factor) {
            scale *= value;
            continue;
        }

        let budget = PolyBudget {
            max_terms: 8,
            max_total_degree: 4,
            max_pow_exp: 4,
        };
        if let Ok(poly) = multipoly_from_expr(ctx, factor, &budget) {
            let (content, primitive) = poly.primitive_part();
            if content.is_zero() {
                return None;
            }
            if content.is_one() {
                factors.push(factor);
            } else {
                scale *= content;
                factors.push(multipoly_to_expr(&primitive, ctx));
            }
            continue;
        }

        factors.push(factor);
    }

    Some((scale, factors))
}

fn compact_sqrt_one_plus_denominator_matches_expanded_sum(
    ctx: &mut Context,
    compact: ExprId,
    expanded: ExprId,
) -> bool {
    let Some((compact_scale, compact_base)) = compact_sqrt_one_plus_denominator(ctx, compact)
    else {
        return false;
    };
    let Some((expanded_scale, expanded_base)) = expanded_sqrt_one_plus_denominator(ctx, expanded)
    else {
        return false;
    };

    compact_scale == expanded_scale && exprs_match(ctx, compact_base, expanded_base)
}

fn compact_sqrt_one_plus_denominator(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let mut scale = BigRational::one();
    let mut sqrt_base = None;
    let mut add_one_base = None;

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
            scale *= value;
            continue;
        }
        if let Some(base) = sqrt_like_base(ctx, factor) {
            if sqrt_base.replace(base).is_some() {
                return None;
            }
            continue;
        }
        if let Some(base) = plus_one_base(ctx, factor) {
            if add_one_base.replace(base).is_some() {
                return None;
            }
            continue;
        }
        return None;
    }

    let sqrt_base = sqrt_base?;
    let add_one_base = add_one_base?;
    exprs_match(ctx, sqrt_base, add_one_base).then_some((scale, sqrt_base))
}

fn plus_one_base(ctx: &Context, expr: ExprId) -> Option<ExprId> {
    let Expr::Add(left, right) = ctx.get(expr) else {
        return None;
    };
    if is_one_constant(ctx, *left) {
        Some(*right)
    } else if is_one_constant(ctx, *right) {
        Some(*left)
    } else {
        None
    }
}

fn expanded_sqrt_one_plus_denominator(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Add(left, right) => (*left, *right),
        _ => return None,
    };
    let left_term = scaled_sqrt_power_term(ctx, left)?;
    let right_term = scaled_sqrt_power_term(ctx, right)?;

    let (half_term, three_half_term) = if left_term.2 == BigRational::new(1.into(), 2.into())
        && right_term.2 == BigRational::new(3.into(), 2.into())
    {
        (left_term, right_term)
    } else if left_term.2 == BigRational::new(3.into(), 2.into())
        && right_term.2 == BigRational::new(1.into(), 2.into())
    {
        (right_term, left_term)
    } else {
        return None;
    };

    (half_term.0 == three_half_term.0 && exprs_match(ctx, half_term.1, three_half_term.1))
        .then_some((half_term.0, half_term.1))
}

fn scaled_sqrt_power_term(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId, BigRational)> {
    let mut scale = BigRational::one();
    let mut power = None;

    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_math::numeric_eval::as_rational_const(ctx, factor) {
            scale *= value;
            continue;
        }
        let (base, exponent) = sqrt_power_factor(ctx, factor)?;
        if power.replace((base, exponent)).is_some() {
            return None;
        }
    }

    let (base, exponent) = power?;
    Some((scale, base, exponent))
}

fn sqrt_power_factor(ctx: &Context, expr: ExprId) -> Option<(ExprId, BigRational)> {
    if let Some(base) = sqrt_like_base(ctx, expr) {
        return Some((base, BigRational::new(1.into(), 2.into())));
    }
    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    let Expr::Number(exponent) = ctx.get(*exp) else {
        return None;
    };
    if *exponent == BigRational::new(1.into(), 2.into())
        || *exponent == BigRational::new(3.into(), 2.into())
    {
        Some((*base, exponent.clone()))
    } else {
        None
    }
}

fn arctan_sqrt_positive_polynomial_quotient_shifted_pair_conditions(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    if arctan_sqrt_positive_polynomial_quotient_diff_matches(ctx, left, right).is_some()
        || arctan_sqrt_positive_polynomial_quotient_diff_matches(ctx, right, left).is_some()
        || quotient_matches_with_unordered_products(ctx, left, right)
    {
        return Some(Vec::new());
    }
    arctan_sqrt_derivative_presentation_diff_matches(ctx, left, right)
        .or_else(|| arctan_sqrt_derivative_presentation_diff_matches(ctx, right, left))
        .or_else(|| {
            unit_interval_bounded_inverse_trig_derivative_presentation_diff_matches(
                ctx, left, right,
            )
        })
        .or_else(|| {
            unit_interval_bounded_inverse_trig_derivative_presentation_diff_matches(
                ctx, right, left,
            )
        })
}

fn arctan_sqrt_positive_polynomial_quotient_shifted_core_sum_conditions(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<Vec<crate::ImplicitCondition>> {
    let neg_right = negated_quotient_for_matching(ctx, right);
    arctan_sqrt_positive_polynomial_quotient_shifted_pair_conditions(ctx, left, neg_right).or_else(
        || {
            let neg_left = negated_quotient_for_matching(ctx, left);
            arctan_sqrt_positive_polynomial_quotient_shifted_pair_conditions(ctx, neg_left, right)
        },
    )
}

fn negated_quotient_for_matching(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Div(num, den) => {
            let neg_num = ctx.add(Expr::Neg(num));
            ctx.add(Expr::Div(neg_num, den))
        }
        _ => ctx.add(Expr::Neg(expr)),
    }
}

fn arctan_sqrt_positive_polynomial_quotient_shifted_passthrough_conditions(
    ctx: &mut Context,
    numerator_passthrough: &ConstantPassthrough,
    denominator_passthrough: &ConstantPassthrough,
) -> Option<Vec<crate::ImplicitCondition>> {
    if numerator_passthrough.orientation == denominator_passthrough.orientation {
        arctan_sqrt_positive_polynomial_quotient_shifted_pair_conditions(
            ctx,
            numerator_passthrough.core,
            denominator_passthrough.core,
        )
    } else {
        arctan_sqrt_positive_polynomial_quotient_shifted_core_sum_conditions(
            ctx,
            numerator_passthrough.core,
            denominator_passthrough.core,
        )
    }
}

pub(crate) fn try_diff_arctan_sqrt_positive_polynomial_quotient_shifted_one_root(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return None,
    };
    let numerator_passthrough = strip_constant_passthrough(ctx, numerator)?;
    let denominator_passthrough = strip_constant_passthrough(ctx, denominator)?;
    if numerator_passthrough.constant != denominator_passthrough.constant {
        return None;
    }
    let mut required_conditions =
        arctan_sqrt_positive_polynomial_quotient_shifted_passthrough_conditions(
            ctx,
            &numerator_passthrough,
            &denominator_passthrough,
        )?;
    required_conditions.push(crate::ImplicitCondition::NonZero(denominator));

    Some((ctx.num(1), required_conditions))
}

pub(crate) fn try_diff_arctan_sqrt_positive_polynomial_quotient_shifted_compact_mismatch(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return None,
    };
    let numerator_passthrough = strip_constant_passthrough(ctx, numerator)?;
    let denominator_passthrough = strip_constant_passthrough(ctx, denominator)?;
    if numerator_passthrough.constant == denominator_passthrough.constant {
        return None;
    }
    let mut required_conditions =
        arctan_sqrt_positive_polynomial_quotient_shifted_passthrough_conditions(
            ctx,
            &numerator_passthrough,
            &denominator_passthrough,
        )?;

    let compact_numerator = build_constant_passthrough_expr(
        ctx,
        &numerator_passthrough.constant,
        denominator_passthrough.core,
        denominator_passthrough.orientation,
    );
    let compact_quotient = ctx.add(Expr::Div(compact_numerator, denominator));

    required_conditions.push(crate::ImplicitCondition::NonZero(denominator));
    Some((compact_quotient, required_conditions))
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

pub(crate) fn try_diff_inverse_reciprocal_trig_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions =
        inverse_reciprocal_trig_affine_diff_matches(ctx, diff_expr, divisor, right)
            .or_else(|| {
                inverse_reciprocal_trig_positive_quadratic_diff_matches(
                    ctx, diff_expr, divisor, right,
                )
            })
            .or_else(|| {
                inverse_reciprocal_trig_sqrt_polynomial_diff_matches(ctx, diff_expr, divisor, right)
            })?;
    Some((ctx.num(0), required_conditions))
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

pub(crate) fn try_diff_integral_quadratic_exp_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    try_diff_integral_residual_wrapped_root_zero(
        ctx,
        expr,
        3,
        try_diff_integral_quadratic_exp_residual_direct_root_zero,
    )
}

fn try_diff_integral_quadratic_exp_residual_direct_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (left, right) = match ctx.get(expr) {
        Expr::Sub(left, right) => (*left, *right),
        _ => return None,
    };
    try_diff_integral_quadratic_exp_residual_zero_preorder(ctx, left, right)
        .or_else(|| try_diff_integral_quadratic_exp_residual_zero_preorder(ctx, right, left))
}

fn try_diff_integral_quadratic_exp_residual_zero_preorder(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (diff_expr, divisor) = diff_call_with_optional_divisor(ctx, left)?;
    let required_conditions =
        integrated_quadratic_exp_linear_diff_matches(ctx, diff_expr, divisor, right)?;
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
        .or_else(|| try_hyperbolic_tanh_common_factor_residual_zero_preorder(ctx, left, right))
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

pub(crate) fn try_diff_inverse_reciprocal_trig_residual_root_zero(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            let left = *left;
            let right = *right;
            try_diff_inverse_reciprocal_trig_residual_zero_preorder(ctx, left, right).or_else(
                || try_diff_inverse_reciprocal_trig_residual_zero_preorder(ctx, right, left),
            )
        }
        Expr::Add(left, right) => {
            let left = *left;
            let right = *right;
            let neg_right = ctx.add(Expr::Neg(right));
            let neg_left = ctx.add(Expr::Neg(left));
            try_diff_inverse_reciprocal_trig_residual_zero_preorder(ctx, left, neg_right).or_else(
                || try_diff_inverse_reciprocal_trig_residual_zero_preorder(ctx, right, neg_left),
            )
        }
        _ => None,
    }
}

pub(crate) fn try_diff_inverse_reciprocal_trig_shifted_quotient_root_one(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return None,
    };

    let numerator_passthrough = strip_constant_passthrough(ctx, numerator)?;
    let denominator_passthrough = strip_constant_passthrough(ctx, denominator)?;
    if numerator_passthrough.constant != denominator_passthrough.constant {
        return None;
    }

    let mut required_conditions =
        if numerator_passthrough.orientation == denominator_passthrough.orientation {
            diff_inverse_reciprocal_trig_core_difference_conditions(
                ctx,
                numerator_passthrough.core,
                denominator_passthrough.core,
            )
        } else {
            diff_inverse_reciprocal_trig_core_sum_conditions(
                ctx,
                numerator_passthrough.core,
                denominator_passthrough.core,
            )
        }?;
    required_conditions.push(crate::ImplicitCondition::NonZero(denominator));

    Some((ctx.num(1), required_conditions))
}

pub(crate) fn try_diff_inverse_reciprocal_trig_shifted_quotient_compact_mismatch(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (numerator, denominator) = match ctx.get(expr) {
        Expr::Div(numerator, denominator) => (*numerator, *denominator),
        _ => return None,
    };

    let numerator_passthrough = strip_constant_passthrough(ctx, numerator)?;
    let denominator_passthrough = strip_constant_passthrough(ctx, denominator)?;
    if numerator_passthrough.constant == denominator_passthrough.constant {
        return None;
    }

    let (target_orientation, required_conditions) =
        if numerator_passthrough.orientation == denominator_passthrough.orientation {
            (
                denominator_passthrough.orientation,
                diff_inverse_reciprocal_trig_core_difference_conditions(
                    ctx,
                    numerator_passthrough.core,
                    denominator_passthrough.core,
                )?,
            )
        } else {
            (
                denominator_passthrough.orientation,
                diff_inverse_reciprocal_trig_core_sum_conditions(
                    ctx,
                    numerator_passthrough.core,
                    denominator_passthrough.core,
                )?,
            )
        };

    let compact_numerator = build_constant_passthrough_expr(
        ctx,
        &numerator_passthrough.constant,
        denominator_passthrough.core,
        target_orientation,
    );
    let compact_quotient = ctx.add(Expr::Div(compact_numerator, denominator));
    let mut required_conditions = required_conditions;
    required_conditions.push(crate::ImplicitCondition::NonZero(denominator));

    Some((compact_quotient, required_conditions))
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

    fn require_match<T>(value: Option<T>) -> T {
        match value {
            Some(value) => value,
            None => panic!("expected residual helper to match"),
        }
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

    fn integral_quadratic_exp_root_residual_result(input: &str) -> Option<String> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_integral_quadratic_exp_residual_root_zero(&mut ctx, expr)
            .map(|(result, _required_conditions)| render(&ctx, result))
    }

    fn diff_arctan_sqrt_positive_quotient_shifted_one_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_arctan_sqrt_positive_polynomial_quotient_shifted_one_root(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|cond| cond.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn diff_arctan_sqrt_positive_quotient_shifted_mismatch_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_arctan_sqrt_positive_polynomial_quotient_shifted_compact_mismatch(&mut ctx, expr)
            .map(|(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|cond| cond.display(&ctx))
                        .collect(),
                )
            })
    }

    fn diff_inverse_reciprocal_trig_shifted_quotient_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_inverse_reciprocal_trig_shifted_quotient_root_one(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|cond| cond.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn diff_inverse_reciprocal_trig_shifted_quotient_mismatch_result(
        input: &str,
    ) -> Option<(String, Vec<String>)> {
        let mut ctx = Context::new();
        let expr = parse(input, &mut ctx)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        try_diff_inverse_reciprocal_trig_shifted_quotient_compact_mismatch(&mut ctx, expr).map(
            |(result, required_conditions)| {
                (
                    render(&ctx, result),
                    required_conditions
                        .into_iter()
                        .map(|cond| cond.display(&ctx))
                        .collect(),
                )
            },
        )
    }

    fn simplify_text(input: &str) -> String {
        let mut simplifier = crate::engine::Simplifier::new();
        let expr = parse(input, &mut simplifier.context)
            .unwrap_or_else(|err| panic!("parse failed for {input}: {err:?}"));
        let (result, _steps) = simplifier.simplify(expr);
        render(&simplifier.context, result)
    }

    fn simplify_text_with_default_rules(input: &str) -> String {
        let mut simplifier = crate::engine::Simplifier::with_default_rules();
        simplifier.disable_rule("Double Angle Identity");
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
            "diff(ln(cosh(sqrt(2*x))), x) - tanh(sqrt(2*x))/sqrt(2*x)",
        ];

        for input in cases {
            assert_eq!(
                root_residual_result(input),
                Some("0".to_string()),
                "{input}"
            );
            assert_eq!(simplify_text(input), "0", "{input}");
            assert_eq!(simplify_text_with_default_rules(input), "0", "{input}");
        }
    }

    #[test]
    fn hyperbolic_tanh_common_factor_residual_root_cancels_scaled_sqrt_forms() {
        let cases = [
            "sinh((2*x)^(1/2)) * (2*x)^(-1/2) / cosh((2*x)^(1/2)) - tanh((2*x)^(1/2)) * (2*x)^(-1/2)",
            "tanh((2*x)^(1/2)) * (2*x)^(-1/2) - sinh((2*x)^(1/2)) * (2*x)^(-1/2) / cosh((2*x)^(1/2))",
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
    fn diff_integral_quadratic_exp_residual_root_cancels_supported_target() {
        let input = "diff(integrate((x^2+x+1)*exp(2*x+1), x), x) - (x^2+x+1)*exp(2*x+1)";
        assert_eq!(
            integral_quadratic_exp_root_residual_result(input),
            Some("0".to_string())
        );
        assert_eq!(simplify_text(input), "0");

        assert_eq!(
            integral_quadratic_exp_root_residual_result(
                "diff(integrate((x^2+x+1)*exp(2*x+1), y), y) - (x^2+x+1)*exp(2*x+1)"
            ),
            None
        );
    }

    #[test]
    fn diff_arctan_sqrt_positive_quotient_shifted_one_root_cancels_supported_pair() {
        let input = "(1 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(1 + (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))";
        let (result, required_conditions) =
            require_match(diff_arctan_sqrt_positive_quotient_shifted_one_result(input));
        assert_eq!(result, "1");
        assert_eq!(
            required_conditions,
            vec![
                "(2 * x + 1) / (2 * sqrt((x^2 + x + 1) / (x^2 + x + 3)) * (x^2 + x + 2) * (x^2 + x + 3)) + 1 ≠ 0"
                    .to_string()
            ]
        );

        assert!(diff_arctan_sqrt_positive_quotient_shifted_one_result(
            "(1 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), y))/(1 + (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))"
        )
        .is_none());
    }

    #[test]
    fn diff_arctan_sqrt_positive_quotient_shifted_one_root_cancels_negative_orientation_pair() {
        let input = "(1 - diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(1 - (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))";
        let (result, required_conditions) =
            require_match(diff_arctan_sqrt_positive_quotient_shifted_one_result(input));
        assert_eq!(result, "1");
        assert_eq!(
            required_conditions,
            vec![
                "1 - (2 * x + 1) / (2 * sqrt((x^2 + x + 1) / (x^2 + x + 3)) * (x^2 + x + 2) * (x^2 + x + 3)) ≠ 0"
                    .to_string()
            ]
        );
    }

    #[test]
    fn diff_arctan_sqrt_positive_quotient_shifted_constant_root_cancels_matching_wrappers() {
        let plus_input = "(2 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(2 + (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))";
        let (plus_result, plus_required_conditions) = require_match(
            diff_arctan_sqrt_positive_quotient_shifted_one_result(plus_input),
        );
        assert_eq!(plus_result, "1");
        assert_eq!(
            plus_required_conditions,
            vec![
                "(2 * x + 1) / (2 * sqrt((x^2 + x + 1) / (x^2 + x + 3)) * (x^2 + x + 2) * (x^2 + x + 3)) + 2 ≠ 0"
                    .to_string()
            ]
        );

        let minus_input = "(2 - diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(2 - (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))";
        let (minus_result, minus_required_conditions) = require_match(
            diff_arctan_sqrt_positive_quotient_shifted_one_result(minus_input),
        );
        assert_eq!(minus_result, "1");
        assert_eq!(
            minus_required_conditions,
            vec![
                "2 - (2 * x + 1) / (2 * sqrt((x^2 + x + 1) / (x^2 + x + 3)) * (x^2 + x + 2) * (x^2 + x + 3)) ≠ 0"
                    .to_string()
            ]
        );

        assert!(diff_arctan_sqrt_positive_quotient_shifted_one_result(
            "(2 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(3 + (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))"
        )
        .is_none());
        assert!(diff_arctan_sqrt_positive_quotient_shifted_one_result(
            "(2 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(2 - (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))"
        )
        .is_none());
    }

    #[test]
    fn diff_arctan_sqrt_positive_quotient_shifted_mismatch_compacts_diff_side() {
        assert_eq!(
            simplify_text("(1 + diff(arctan(sqrt(x)), x))/(2 + 1/(2*sqrt(x)*(x+1)))"),
            "(1 / (2 * sqrt(x) * (x + 1)) + 1) / (1 / (2 * sqrt(x) * (x + 1)) + 2)"
        );
        assert_eq!(
            simplify_text("(1 - diff(arctan(sqrt(x)), x))/(2 - 1/(2*sqrt(x)*(x+1)))"),
            "(1 - 1 / (2 * sqrt(x) * (x + 1))) / (2 - 1 / (2 * sqrt(x) * (x + 1)))"
        );
        assert_eq!(
            simplify_text("(1 + diff(arctan(sqrt(2*x+3)), x))/(2 + 1/(sqrt(2*x+3)*(2*x+4)))"),
            "(1 / (sqrt(2 * x + 3) * (2 * x + 4)) + 1) / (1 / (sqrt(2 * x + 3) * (2 * x + 4)) + 2)"
        );
        assert_eq!(
            simplify_text("(1 - diff(arctan(sqrt(2*x+3)), x))/(2 - 1/(sqrt(2*x+3)*(2*x+4)))"),
            "(1 - 1 / (sqrt(2 * x + 3) * (2 * x + 4))) / (2 - 1 / (sqrt(2 * x + 3) * (2 * x + 4)))"
        );
        assert_eq!(
            simplify_text(
                "(1 + diff(arctan(sqrt(5-3*x)), x))/(2 - 3/(2*sqrt(5-3*x)*(6-3*x)))"
            ),
            "(1 - 3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x))) / (2 - 3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)))"
        );
        assert_eq!(
            simplify_text(
                "(1 - diff(arctan(sqrt(5-3*x)), x))/(2 + 3/(2*sqrt(5-3*x)*(6-3*x)))"
            ),
            "(3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)) + 1) / (3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)) + 2)"
        );
        assert_eq!(
            simplify_text("(1 + diff(arctan(sqrt(5-3*x)), x))/(1 - 3/(2*sqrt(5-3*x)*(6-3*x)))"),
            "1"
        );
        assert_eq!(
            simplify_text(
                "(1 + diff(arccot(sqrt(5-3*x)), x))/(2 + 3/(2*sqrt(5-3*x)*(6-3*x)))"
            ),
            "(3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)) + 1) / (3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)) + 2)"
        );
        assert_eq!(
            simplify_text(
                "(1 - diff(arccot(sqrt(5-3*x)), x))/(2 - 3/(2*sqrt(5-3*x)*(6-3*x)))"
            ),
            "(1 - 3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x))) / (2 - 3 / (2 * sqrt(5 - 3 * x) * (6 - 3 * x)))"
        );
        assert_eq!(
            simplify_text("(1 + diff(arccot(sqrt(5-3*x)), x))/(1 + 3/(2*sqrt(5-3*x)*(6-3*x)))"),
            "1"
        );

        let plus_input = "(1 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(2 + (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))";
        let (plus_result, plus_required_conditions) = require_match(
            diff_arctan_sqrt_positive_quotient_shifted_mismatch_result(plus_input),
        );
        assert_ne!(plus_result, "1");
        assert!(!plus_result.contains("diff"), "{plus_result}");
        assert!(plus_result.contains("+ 1"), "{plus_result}");
        assert!(plus_result.contains("+ 2"), "{plus_result}");
        assert_eq!(
            plus_required_conditions,
            vec![
                "(2 * x + 1) / (2 * sqrt((x^2 + x + 1) / (x^2 + x + 3)) * (x^2 + x + 2) * (x^2 + x + 3)) + 2 ≠ 0"
                    .to_string()
            ]
        );

        let minus_input = "(1 - diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(2 - (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))";
        let (minus_result, minus_required_conditions) = require_match(
            diff_arctan_sqrt_positive_quotient_shifted_mismatch_result(minus_input),
        );
        assert_ne!(minus_result, "1");
        assert!(!minus_result.contains("diff"), "{minus_result}");
        assert!(minus_result.contains("1 -"), "{minus_result}");
        assert!(minus_result.contains("2 -"), "{minus_result}");
        assert_eq!(
            minus_required_conditions,
            vec![
                "2 - (2 * x + 1) / (2 * sqrt((x^2 + x + 1) / (x^2 + x + 3)) * (x^2 + x + 2) * (x^2 + x + 3)) ≠ 0"
                    .to_string()
            ]
        );

        assert!(diff_arctan_sqrt_positive_quotient_shifted_mismatch_result(
            "(1 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(1 + (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))"
        )
        .is_none());
        assert!(diff_arctan_sqrt_positive_quotient_shifted_mismatch_result(
            "(1 + diff(arctan(sqrt((x^2+x+1)/(x^2+x+3))), x))/(2 - (2*x+1)/(2*(x^2+x+2)*(x^2+x+3)*sqrt((x^2+x+1)/(x^2+x+3))))"
        )
        .is_none());
    }

    #[test]
    fn diff_unit_interval_bounded_inverse_trig_shifted_quotient_compacts_contextual_diff() {
        assert_eq!(
            simplify_text("(1 + diff(1/2*arcsin(2*x-1), x))/(1 + 1/(2*sqrt(x)*sqrt(1-x)))"),
            "1"
        );
        assert_eq!(
            simplify_text("(1 + diff(1/2*arccos(2*x-1), x))/(1 - 1/(2*sqrt(x)*sqrt(1-x)))"),
            "1"
        );
        assert_eq!(
            simplify_text("(1 + diff(1/2*arcsin(2*x-1), x))/(2 + 1/(2*sqrt(x)*sqrt(1-x)))"),
            "(1 / (2 * sqrt(x) * sqrt(1 - x)) + 1) / (1 / (2 * sqrt(x) * sqrt(1 - x)) + 2)"
        );
        assert_eq!(
            simplify_text("(1 + diff(1/2*arccos(2*x-1), x))/(2 - 1/(2*sqrt(x)*sqrt(1-x)))"),
            "(1 - 1 / (2 * sqrt(x) * sqrt(1 - x))) / (2 - 1 / (2 * sqrt(x) * sqrt(1 - x)))"
        );
    }

    #[test]
    fn diff_inverse_reciprocal_trig_shifted_quotient_root_cancels_matching_wrappers() {
        let minus_input = "(1 - diff(arcsec(sqrt(x+1)), x))/(1 - 1/(2*(x+1)*sqrt(x)))";
        let (minus_result, minus_required_conditions) = require_match(
            diff_inverse_reciprocal_trig_shifted_quotient_result(minus_input),
        );
        assert_eq!(minus_result, "1");
        assert_eq!(
            minus_required_conditions,
            vec![
                "x > 0".to_string(),
                "1 - 1 / (2 * sqrt(x) * (x + 1)) ≠ 0".to_string()
            ]
        );

        let arccsc_input = "(1 - diff(arccsc(sqrt(x+1)), x))/(1 + 1/(2*(x+1)*sqrt(x)))";
        let (arccsc_result, arccsc_required_conditions) = require_match(
            diff_inverse_reciprocal_trig_shifted_quotient_result(arccsc_input),
        );
        assert_eq!(arccsc_result, "1");
        assert_eq!(
            arccsc_required_conditions,
            vec![
                "x > 0".to_string(),
                "1 / (2 * sqrt(x) * (x + 1)) + 1 ≠ 0".to_string()
            ]
        );

        assert!(diff_inverse_reciprocal_trig_shifted_quotient_result(
            "(1 - diff(arcsec(sqrt(x+1)), x))/(2 - 1/(2*(x+1)*sqrt(x)))"
        )
        .is_none());
        assert!(diff_inverse_reciprocal_trig_shifted_quotient_result(
            "(1 + diff(arcsec(sqrt(x+1)), x))/(1 - 1/(2*(x+1)*sqrt(x)))"
        )
        .is_none());
        assert!(diff_inverse_reciprocal_trig_shifted_quotient_result(
            "(1 - diff(arccsc(sqrt(x+1)), x))/(2 + 1/(2*(x+1)*sqrt(x)))"
        )
        .is_none());
        assert!(diff_inverse_reciprocal_trig_shifted_quotient_result(
            "(1 - diff(arccsc(sqrt(x+1)), x))/(1 - 1/(2*(x+1)*sqrt(x)))"
        )
        .is_none());
    }

    #[test]
    fn diff_inverse_reciprocal_trig_shifted_quotient_mismatch_compacts_diff_side() {
        let arccsc_input = "(1 - diff(arccsc(sqrt(x+1)), x))/(2 + 1/(2*(x+1)*sqrt(x)))";
        let (arccsc_result, arccsc_required_conditions) = require_match(
            diff_inverse_reciprocal_trig_shifted_quotient_mismatch_result(arccsc_input),
        );
        assert_ne!(arccsc_result, "1");
        assert!(!arccsc_result.contains("diff"), "{arccsc_result}");
        assert!(arccsc_result.contains("+ 1"), "{arccsc_result}");
        assert!(arccsc_result.contains("+ 2"), "{arccsc_result}");
        assert_eq!(
            arccsc_required_conditions,
            vec![
                "x > 0".to_string(),
                "1 / (2 * sqrt(x) * (x + 1)) + 2 ≠ 0".to_string()
            ]
        );

        let arcsec_input = "(1 - diff(arcsec(sqrt(x+1)), x))/(2 - 1/(2*(x+1)*sqrt(x)))";
        let (arcsec_result, arcsec_required_conditions) = require_match(
            diff_inverse_reciprocal_trig_shifted_quotient_mismatch_result(arcsec_input),
        );
        assert_ne!(arcsec_result, "1");
        assert!(!arcsec_result.contains("diff"), "{arcsec_result}");
        assert!(arcsec_result.contains("1 -"), "{arcsec_result}");
        assert!(arcsec_result.contains("2 -"), "{arcsec_result}");
        assert_eq!(
            arcsec_required_conditions,
            vec![
                "x > 0".to_string(),
                "2 - 1 / (2 * sqrt(x) * (x + 1)) ≠ 0".to_string()
            ]
        );

        let arcsec_negative_affine_input =
            "(1 + diff(arcsec(sqrt(5-3*x)), x))/(2 - 3/(2*(5-3*x)*sqrt(4-3*x)))";
        let (arcsec_negative_affine_result, arcsec_negative_affine_required_conditions) =
            require_match(
                diff_inverse_reciprocal_trig_shifted_quotient_mismatch_result(
                    arcsec_negative_affine_input,
                ),
            );
        assert_ne!(arcsec_negative_affine_result, "1");
        assert!(!arcsec_negative_affine_result.contains("diff"));
        assert_eq!(
            arcsec_negative_affine_result,
            "(1 - 3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x))) / (2 - 3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x)))"
        );
        assert_eq!(
            arcsec_negative_affine_required_conditions,
            vec![
                "4 - 3 * x > 0".to_string(),
                "2 - 3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x)) ≠ 0".to_string()
            ]
        );

        let arccsc_negative_affine_input =
            "(1 + diff(arccsc(sqrt(5-3*x)), x))/(2 + 3/(2*(5-3*x)*sqrt(4-3*x)))";
        let (arccsc_negative_affine_result, arccsc_negative_affine_required_conditions) =
            require_match(
                diff_inverse_reciprocal_trig_shifted_quotient_mismatch_result(
                    arccsc_negative_affine_input,
                ),
            );
        assert_ne!(arccsc_negative_affine_result, "1");
        assert!(!arccsc_negative_affine_result.contains("diff"));
        assert_eq!(
            arccsc_negative_affine_result,
            "(3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x)) + 1) / (3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x)) + 2)"
        );
        assert_eq!(
            arccsc_negative_affine_required_conditions,
            vec![
                "4 - 3 * x > 0".to_string(),
                "3 / (2 * sqrt(4 - 3 * x) * (5 - 3 * x)) + 2 ≠ 0".to_string()
            ]
        );

        assert!(
            diff_inverse_reciprocal_trig_shifted_quotient_mismatch_result(
                "(1 - diff(arccsc(sqrt(x+1)), x))/(1 + 1/(2*(x+1)*sqrt(x)))"
            )
            .is_none()
        );
        assert!(
            diff_inverse_reciprocal_trig_shifted_quotient_mismatch_result(
                "(1 - diff(arccsc(sqrt(x+1)), x))/(2 - 1/(2*(x+1)*sqrt(x)))"
            )
            .is_none()
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
