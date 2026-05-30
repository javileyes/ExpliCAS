use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::multipoly::{multipoly_from_expr, multipoly_to_expr, PolyBudget};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

use super::presentation_utils::{
    calculus_sqrt_like_radicand, is_half_power_exponent, rational_const_for_hold,
    unwrap_internal_hold_for_calculus,
};

pub(super) fn rational_const_for_calculus_presentation(
    ctx: &mut Context,
    value: BigRational,
) -> ExprId {
    if value == BigRational::one() {
        ctx.num(1)
    } else {
        ctx.add(Expr::Number(value))
    }
}

pub(super) fn scale_expr_for_calculus_presentation(
    ctx: &mut Context,
    coeff: BigRational,
    expr: ExprId,
) -> ExprId {
    if coeff.is_one() {
        return expr;
    }
    let coeff = rational_const_for_calculus_presentation(ctx, coeff);
    if let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        if value == BigRational::one() {
            return coeff;
        }
        if let Some(coeff_value) = cas_ast::views::as_rational_const(ctx, coeff, 8) {
            return rational_const_for_calculus_presentation(ctx, coeff_value * value);
        }
    }
    cas_math::expr_nary::build_balanced_mul(ctx, &[coeff, expr])
}

pub(super) fn nonzero_rational_parts(value: &BigRational) -> Option<(BigRational, BigRational)> {
    if value.is_zero() {
        return None;
    }

    let numerator = BigRational::from_integer(value.numer().clone());
    let denominator = BigRational::from_integer(value.denom().clone());
    Some((numerator, denominator))
}

pub(super) fn rational_scaled_single_factor_allow_unit(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (inner_scale, inner_factor) = rational_scaled_single_factor_allow_unit(ctx, inner)
            .unwrap_or_else(|| (BigRational::one(), inner));
        return Some((-inner_scale, inner_factor));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut non_numeric = Vec::new();

    for factor in factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else {
            non_numeric.push(factor);
        }
    }

    let [inner] = non_numeric.as_slice() else {
        return None;
    };

    Some((scale, *inner))
}

pub(super) fn rational_scaled_single_factor(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let (inner_scale, inner_factor) = rational_scaled_single_factor_allow_unit(ctx, inner)
            .unwrap_or_else(|| (BigRational::one(), inner));
        let scale = -inner_scale;
        if scale.is_one() {
            return None;
        }
        return Some((scale, inner_factor));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    let mut scale = BigRational::one();
    let mut non_numeric = Vec::new();

    for factor in factors {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else {
            non_numeric.push(factor);
        }
    }

    if scale.is_one() {
        return None;
    }

    let [inner] = non_numeric.as_slice() else {
        return None;
    };

    Some((scale, *inner))
}

pub(super) fn split_outer_numeric_mul_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    let expr = unwrap_internal_hold_for_calculus(ctx, expr);
    let mut scale = BigRational::one();
    let mut core = None;
    for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
        if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
            scale *= value;
        } else if core.replace(factor).is_some() {
            return None;
        }
    }
    Some((scale, core.unwrap_or(expr)))
}

pub(super) fn scale_compact_fraction_numerator_by_rational_for_calculus_presentation(
    ctx: &mut Context,
    derivative: ExprId,
    scale: BigRational,
) -> ExprId {
    if scale.is_one() {
        return unwrap_internal_hold_for_calculus(ctx, derivative);
    }
    if scale.is_zero() {
        return ctx.num(0);
    }

    let derivative = unwrap_internal_hold_for_calculus(ctx, derivative);
    let Expr::Div(num, den) = ctx.get(derivative).clone() else {
        return scale_expr_for_calculus_presentation(ctx, scale, derivative);
    };

    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&scale).unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, num);
    let denominator = if denominator_coeff == BigRational::one() {
        den
    } else {
        scale_expr_for_calculus_presentation(ctx, denominator_coeff, den)
    };

    ctx.add(Expr::Div(numerator, denominator))
}

pub(super) fn scale_fraction_for_calculus_presentation(
    ctx: &mut Context,
    numerator: ExprId,
    denominator: ExprId,
    scale: BigRational,
) -> (ExprId, ExprId) {
    let (numerator_coeff, denominator_coeff) =
        nonzero_rational_parts(&scale).unwrap_or_else(|| (BigRational::zero(), BigRational::one()));
    let numerator = signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator);
    let denominator = if denominator_coeff.is_one() {
        denominator
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, denominator])
    };

    (numerator, denominator)
}

pub(super) fn negate_calculus_presentation(ctx: &mut Context, expr: ExprId) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Neg(inner) => inner,
        Expr::Div(numerator, denominator) => {
            let numerator = match ctx.get(numerator).clone() {
                Expr::Neg(inner) => inner,
                _ => ctx.add(Expr::Neg(numerator)),
            };
            ctx.add(Expr::Div(numerator, denominator))
        }
        _ => ctx.add(Expr::Neg(expr)),
    }
}

pub(super) fn fold_numeric_mul_constants_for_hold(ctx: &mut Context, expr: ExprId) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Mul(_, _) => {
            let mut factors = cas_math::expr_nary::mul_leaves(ctx, expr);
            let mut scale = BigRational::one();
            let mut non_numeric = Vec::new();

            while let Some(factor) = factors.pop() {
                let folded = fold_numeric_mul_constants_for_hold(ctx, factor);
                if matches!(ctx.get(folded), Expr::Mul(_, _)) {
                    factors.extend(cas_math::expr_nary::mul_leaves(ctx, folded));
                    continue;
                }
                if let Some(value) = rational_const_for_hold(ctx, folded) {
                    scale *= value;
                } else {
                    non_numeric.push(folded);
                }
            }

            if scale.is_zero() {
                return ctx.num(0);
            }

            if !scale.is_one() && non_numeric.len() == 1 {
                if let Some(radicand) = sqrt_positive_rational_for_hold(ctx, non_numeric[0]) {
                    let sign = if scale.is_negative() {
                        -BigRational::one()
                    } else {
                        BigRational::one()
                    };
                    let scaled_radicand = &scale * &scale * radicand;
                    let folded = if let Some(root) =
                        exact_positive_rational_sqrt_for_calculus_presentation(&scaled_radicand)
                    {
                        ctx.add(Expr::Number(sign * root))
                    } else {
                        let sqrt = sqrt_positive_rational_expr_for_calculus_presentation(
                            ctx,
                            scaled_radicand,
                        );
                        if sign.is_negative() {
                            negate_calculus_presentation(ctx, sqrt)
                        } else {
                            sqrt
                        }
                    };
                    return folded;
                }
                if let Expr::Div(num, den) = ctx.get(non_numeric[0]).clone() {
                    let scale_expr = ctx.add(Expr::Number(scale));
                    let scaled_num = ctx.add(Expr::Mul(scale_expr, num));
                    let folded_num = fold_numeric_mul_constants_for_hold(ctx, scaled_num);
                    return ctx.add(Expr::Div(folded_num, den));
                }
                if scale == -BigRational::one() {
                    if let Expr::Neg(inner) = ctx.get(non_numeric[0]).clone() {
                        return inner;
                    }
                }
            }

            if !scale.is_one() || non_numeric.is_empty() {
                non_numeric.insert(0, ctx.add(Expr::Number(scale)));
            }

            if non_numeric.len() == 1 {
                non_numeric[0]
            } else {
                cas_math::expr_nary::build_balanced_mul(ctx, &non_numeric)
            }
        }
        Expr::Div(num, den) => {
            let num = fold_numeric_mul_constants_for_hold(ctx, num);
            let den = fold_numeric_mul_constants_for_hold(ctx, den);
            if let Some(den_value) = rational_const_for_hold(ctx, den) {
                if den_value.is_zero() {
                    return ctx.add(Expr::Div(num, den));
                }
                if let Some(num_value) = rational_const_for_hold(ctx, num) {
                    return ctx.add(Expr::Number(num_value / den_value));
                }
                let reciprocal = ctx.add(Expr::Number(BigRational::one() / den_value));
                let scaled = ctx.add(Expr::Mul(reciprocal, num));
                return fold_numeric_mul_constants_for_hold(ctx, scaled);
            }
            if let Some(num_value) = rational_const_for_hold(ctx, num)
                .filter(|_| matches!(ctx.get(den), Expr::Mul(_, _)))
            {
                let mut denominator_scale = BigRational::one();
                let mut denominator_factors = Vec::new();
                for factor in cas_math::expr_nary::mul_leaves(ctx, den) {
                    if let Some(value) = rational_const_for_hold(ctx, factor) {
                        denominator_scale *= value;
                    } else {
                        denominator_factors.push(factor);
                    }
                }
                if !denominator_scale.is_zero()
                    && !denominator_scale.is_one()
                    && !denominator_factors.is_empty()
                {
                    let scaled_num_value = num_value / denominator_scale;
                    if !scaled_num_value.is_integer() {
                        return ctx.add(Expr::Div(num, den));
                    }
                    let num = ctx.add(Expr::Number(scaled_num_value));
                    let den = if denominator_factors.len() == 1 {
                        denominator_factors[0]
                    } else {
                        cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors)
                    };
                    return ctx.add(Expr::Div(num, den));
                }
            }
            if matches!(ctx.get(den), Expr::Mul(_, _)) {
                let mut numerator_scale = BigRational::one();
                let mut numerator_factors = Vec::new();
                let mut raw_numerator_factors = Vec::new();
                mul_leaves_preserve_order_for_calculus_presentation(
                    ctx,
                    num,
                    &mut raw_numerator_factors,
                );
                for factor in raw_numerator_factors {
                    if let Some(value) = rational_const_for_hold(ctx, factor) {
                        numerator_scale *= value;
                    } else {
                        numerator_factors.push(factor);
                    }
                }

                let mut denominator_scale = BigRational::one();
                let mut denominator_factors = Vec::new();
                let mut raw_denominator_factors = Vec::new();
                mul_leaves_preserve_order_for_calculus_presentation(
                    ctx,
                    den,
                    &mut raw_denominator_factors,
                );
                for factor in raw_denominator_factors {
                    if let Some(value) = rational_const_for_hold(ctx, factor) {
                        denominator_scale *= value;
                    } else {
                        denominator_factors.push(factor);
                    }
                }

                if !denominator_scale.is_zero()
                    && !denominator_scale.is_one()
                    && !numerator_factors.is_empty()
                    && !denominator_factors.is_empty()
                {
                    let scaled_numerator = numerator_scale / denominator_scale;
                    if scaled_numerator.is_integer() {
                        let numerator_core = if numerator_factors.len() == 1 {
                            numerator_factors[0]
                        } else {
                            cas_math::expr_nary::build_balanced_mul(ctx, &numerator_factors)
                        };
                        let numerator = signed_numerator_for_calculus_presentation(
                            ctx,
                            scaled_numerator,
                            numerator_core,
                        );
                        let denominator = if denominator_factors.len() == 1 {
                            denominator_factors[0]
                        } else {
                            cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors)
                        };
                        return ctx.add(Expr::Div(numerator, denominator));
                    }
                }
            }
            ctx.add(Expr::Div(num, den))
        }
        Expr::Neg(inner) => {
            let inner = fold_numeric_mul_constants_for_hold(ctx, inner);
            ctx.add(Expr::Neg(inner))
        }
        _ => expr,
    }
}

fn mul_leaves_preserve_order_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    out: &mut Vec<ExprId>,
) {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Mul(left, right) => {
            mul_leaves_preserve_order_for_calculus_presentation(ctx, left, out);
            mul_leaves_preserve_order_for_calculus_presentation(ctx, right, out);
        }
        _ => out.push(expr),
    }
}

pub(super) fn fold_numeric_mul_constants_for_hold_additive_terms(
    ctx: &mut Context,
    expr: ExprId,
) -> ExprId {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => {
            let left = fold_numeric_mul_constants_for_hold_additive_terms(ctx, left);
            let right = fold_numeric_mul_constants_for_hold_additive_terms(ctx, right);
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = fold_numeric_mul_constants_for_hold_additive_terms(ctx, left);
            let right = fold_numeric_mul_constants_for_hold_additive_terms(ctx, right);
            ctx.add(Expr::Sub(left, right))
        }
        _ => fold_numeric_mul_constants_for_hold(ctx, expr),
    }
}

fn sqrt_positive_rational_for_hold(ctx: &Context, expr: ExprId) -> Option<BigRational> {
    let value = match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Sqrt) =>
        {
            cas_ast::views::as_rational_const(ctx, args[0], 8)?
        }
        Expr::Pow(base, exp) if is_half_power_exponent(ctx, *exp) => {
            cas_ast::views::as_rational_const(ctx, *base, 8)?
        }
        _ => return None,
    };
    value.is_positive().then_some(value)
}

pub(super) fn add_one_for_calculus_presentation(ctx: &mut Context, expr: ExprId) -> ExprId {
    add_rational_for_calculus_presentation(ctx, expr, BigRational::one())
}

pub(super) fn subtract_from_one_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
) -> ExprId {
    let one = ctx.num(1);
    let raw = ctx.add(Expr::Sub(one, expr));
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };

    multipoly_from_expr(ctx, raw, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw)
}

pub(super) fn subtract_from_rational_for_calculus_presentation(
    ctx: &mut Context,
    value: BigRational,
    expr: ExprId,
) -> ExprId {
    let constant = rational_const_for_calculus_presentation(ctx, value);
    let raw = ctx.add(Expr::Sub(constant, expr));
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };

    multipoly_from_expr(ctx, raw, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw)
}

pub(super) fn subtract_expr_for_calculus_presentation(
    ctx: &mut Context,
    left: ExprId,
    right: ExprId,
) -> ExprId {
    let raw = ctx.add(Expr::Sub(left, right));
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };

    multipoly_from_expr(ctx, raw, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw)
}

pub(super) fn add_rational_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    value: BigRational,
) -> ExprId {
    if value.is_zero() {
        return expr;
    }

    let constant = rational_const_for_calculus_presentation(ctx, value);
    let raw = ctx.add(Expr::Add(expr, constant));
    let budget = PolyBudget {
        max_terms: 8,
        max_total_degree: 4,
        max_pow_exp: 4,
    };

    multipoly_from_expr(ctx, raw, &budget)
        .map(|poly| multipoly_to_expr(&poly, ctx))
        .unwrap_or(raw)
}

pub(super) fn add_rational_combining_additive_constant_for_calculus_presentation(
    ctx: &mut Context,
    expr: ExprId,
    value: BigRational,
) -> ExprId {
    let terms = cas_math::expr_nary::add_terms_signed(ctx, expr);
    if terms.len() < 2 {
        return add_rational_for_calculus_presentation(ctx, expr, value);
    }

    let mut constant = value;
    let mut saw_constant = false;
    let mut rebuilt_terms = Vec::new();
    for (term, sign) in terms {
        if let Some(term_value) = cas_ast::views::as_rational_const(ctx, term, 8) {
            saw_constant = true;
            if sign == cas_math::expr_nary::Sign::Neg {
                constant -= term_value;
            } else {
                constant += term_value;
            }
            continue;
        }

        if sign == cas_math::expr_nary::Sign::Neg {
            rebuilt_terms.push(ctx.add(Expr::Neg(term)));
        } else {
            rebuilt_terms.push(term);
        }
    }

    if !saw_constant {
        return add_rational_for_calculus_presentation(ctx, expr, constant);
    }
    if !constant.is_zero() {
        rebuilt_terms.push(rational_const_for_calculus_presentation(ctx, constant));
    }

    match rebuilt_terms.len() {
        0 => ctx.num(0),
        1 => rebuilt_terms[0],
        _ => cas_math::expr_nary::build_balanced_add(ctx, &rebuilt_terms),
    }
}

pub(super) fn reciprocal_integer_radicand_content_for_calculus_presentation(
    value: &BigRational,
) -> Option<BigRational> {
    if value.is_positive() && value.numer().is_one() && value.denom() > &BigInt::one() {
        Some(BigRational::from_integer(value.denom().clone()))
    } else {
        None
    }
}

pub(super) fn positive_constant_over_expr_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr) {
        Expr::Div(num, den) => {
            let value = cas_ast::views::as_rational_const(ctx, *num, 8)?;
            value.is_positive().then_some((value, *den))
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new((-1).into(), 1.into())) =>
        {
            Some((BigRational::one(), *base))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut denominator = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    scale *= value;
                    continue;
                }
                let Expr::Pow(base, exp) = ctx.get(factor) else {
                    return None;
                };
                if cas_ast::views::as_rational_const(ctx, *exp, 8)
                    != Some(BigRational::new((-1).into(), 1.into()))
                    || denominator.replace(*base).is_some()
                {
                    return None;
                }
            }
            scale.is_positive().then_some((scale, denominator?))
        }
        _ => None,
    }
}

pub(super) fn positive_constant_over_inverse_sqrt_arg_for_calculus_presentation(
    ctx: &Context,
    arg: ExprId,
) -> Option<(BigRational, ExprId)> {
    match ctx.get(arg) {
        Expr::Function(sqrt_fn, sqrt_args)
            if sqrt_args.len() == 1 && ctx.is_builtin(*sqrt_fn, BuiltinFn::Sqrt) =>
        {
            positive_constant_over_expr_for_calculus_presentation(ctx, sqrt_args[0])
        }
        Expr::Pow(base, exp) if is_half_power_exponent(ctx, *exp) => {
            positive_constant_over_expr_for_calculus_presentation(ctx, *base)
        }
        Expr::Function(abs_fn, abs_args)
            if abs_args.len() == 1 && ctx.is_builtin(*abs_fn, BuiltinFn::Abs) =>
        {
            positive_constant_over_reciprocal_sqrt_expr_for_calculus_presentation(ctx, abs_args[0])
        }
        _ => None,
    }
}

fn positive_constant_over_reciprocal_sqrt_expr_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr) {
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new((-1).into(), 2.into())) =>
        {
            Some((BigRational::one(), *base))
        }
        Expr::Div(num, den) => {
            let numerator = cas_ast::views::as_rational_const(ctx, *num, 8)?;
            if !numerator.is_positive() {
                return None;
            }
            let denominator = calculus_sqrt_like_radicand(ctx, *den)?;
            Some((numerator, denominator))
        }
        Expr::Mul(_, _) => {
            let mut scale = BigRational::one();
            let mut denominator = None;
            for factor in cas_math::expr_nary::mul_leaves(ctx, expr) {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    scale *= value;
                    continue;
                }
                let Expr::Pow(base, exp) = ctx.get(factor) else {
                    return None;
                };
                if cas_ast::views::as_rational_const(ctx, *exp, 8)
                    != Some(BigRational::new((-1).into(), 2.into()))
                    || denominator.replace(*base).is_some()
                {
                    return None;
                }
            }
            if !scale.is_positive() {
                return None;
            }
            let scale_squared = &scale * &scale;
            Some((scale_squared, denominator?))
        }
        _ => None,
    }
}

pub(super) fn positive_rational_sqrt_denominator_factor_for_calculus_presentation(
    ctx: &mut Context,
    value: &BigRational,
) -> Option<(BigRational, Option<ExprId>)> {
    if let Some(root) = exact_positive_rational_sqrt_for_calculus_presentation(value) {
        return Some((root, None));
    }

    let radicand = rational_const_for_calculus_presentation(ctx, value.clone());
    let sqrt_factor = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    Some((value.clone(), Some(sqrt_factor)))
}

pub(super) fn signed_numerator_for_calculus_presentation(
    ctx: &mut Context,
    coeff: BigRational,
    expr: ExprId,
) -> ExprId {
    if coeff == -BigRational::one()
        && !cas_ast::views::as_rational_const(ctx, expr, 8).is_some_and(|value| value.is_one())
    {
        return ctx.add(Expr::Neg(expr));
    }
    scale_expr_for_calculus_presentation(ctx, coeff, expr)
}

pub(super) fn signed_rational_const_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<BigRational> {
    if let Some(value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        return Some(value);
    }
    if let Expr::Neg(inner) = ctx.get(expr) {
        return cas_ast::views::as_rational_const(ctx, *inner, 8).map(|value| -value);
    }
    None
}

pub(super) fn exact_positive_rational_sqrt_for_calculus_presentation(
    value: &BigRational,
) -> Option<BigRational> {
    if !value.is_positive() {
        return None;
    }

    let numer_sqrt = value.numer().sqrt();
    let denom_sqrt = value.denom().sqrt();
    if &numer_sqrt * &numer_sqrt == *value.numer() && &denom_sqrt * &denom_sqrt == *value.denom() {
        Some(BigRational::new(numer_sqrt, denom_sqrt))
    } else {
        None
    }
}

fn split_square_factor_positive_bigint_for_calculus_presentation(
    value: &BigInt,
) -> (BigInt, BigInt) {
    let mut outside = BigInt::one();
    let mut inside = value.clone();

    for factor in 2_i64..=97 {
        let factor = BigInt::from(factor);
        let square = &factor * &factor;
        while (&inside % &square).is_zero() {
            inside /= &square;
            outside *= &factor;
        }
    }

    let root = inside.sqrt();
    if root > BigInt::one() && &root * &root == inside {
        outside *= root;
        inside = BigInt::one();
    }

    (outside, inside)
}

pub(super) fn split_square_factor_positive_rational_for_calculus_presentation(
    value: &BigRational,
) -> (BigRational, BigRational) {
    debug_assert!(value.is_positive());

    let (numer_outside, numer_inside) =
        split_square_factor_positive_bigint_for_calculus_presentation(value.numer());
    let (denom_outside, denom_inside) =
        split_square_factor_positive_bigint_for_calculus_presentation(value.denom());

    (
        BigRational::new(numer_outside, denom_outside),
        BigRational::new(numer_inside, denom_inside),
    )
}

pub(super) fn sqrt_positive_rational_expr_for_calculus_presentation(
    ctx: &mut Context,
    value: BigRational,
) -> ExprId {
    if let Some(sqrt_value) = exact_positive_rational_sqrt_for_calculus_presentation(&value) {
        return rational_const_for_calculus_presentation(ctx, sqrt_value);
    }

    let (outside, inside) = split_square_factor_positive_rational_for_calculus_presentation(&value);
    if inside.is_one() {
        return rational_const_for_calculus_presentation(ctx, outside);
    }

    let radicand = rational_const_for_calculus_presentation(ctx, inside);
    let sqrt = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    scale_expr_for_calculus_presentation(ctx, outside, sqrt)
}

pub(super) fn scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
    ctx: &mut Context,
    value: BigRational,
    expr: ExprId,
) -> ExprId {
    if value.is_one() {
        return expr;
    }

    let sqrt_scale = sqrt_positive_rational_expr_for_calculus_presentation(ctx, value);
    if let Some(expr_value) = cas_ast::views::as_rational_const(ctx, expr, 8) {
        if expr_value.is_one() {
            return sqrt_scale;
        }
        if expr_value == -BigRational::one() {
            return ctx.add(Expr::Neg(sqrt_scale));
        }
        return scale_expr_for_calculus_presentation(ctx, expr_value, sqrt_scale);
    }
    cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_scale, expr])
}
