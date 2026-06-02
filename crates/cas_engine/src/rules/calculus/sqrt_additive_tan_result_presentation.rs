use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, ToPrimitive, Zero};

use super::presentation_compaction::{
    compact_numeric_mul_factors_for_calculus_presentation,
    compact_small_power_exponents_for_calculus_presentation,
    split_signed_numeric_scale_single_core_for_calculus_presentation,
};
use super::presentation_utils::sqrt_raw_for_calculus_presentation;
use super::scalar_presentation::{
    rational_const_for_calculus_presentation, scale_expr_for_calculus_presentation,
};

pub(super) struct SqrtAdditiveTanDerivativePresentationParts {
    pub(super) radicand: ExprId,
    pub(super) tan_arg: ExprId,
    pub(super) reciprocal_trig_builtin: BuiltinFn,
    pub(super) tan_scale: BigRational,
    pub(super) other_derivatives: Vec<ExprId>,
}

pub(super) struct SqrtAdditiveTanCommonDenominatorPresentationParts {
    pub(super) radicand: ExprId,
    pub(super) tan_arg: ExprId,
    pub(super) common_trig_denominator_builtin: BuiltinFn,
    pub(super) tan_scale: BigRational,
    pub(super) common_denominator: Option<ExprId>,
    pub(super) reciprocal_derivative_scales: Vec<BigRational>,
    pub(super) other_derivatives: Vec<ExprId>,
}

pub(super) fn compact_tan_sqrt_common_denominator_numerator_term(
    ctx: &mut Context,
    cos_arg: ExprId,
    cos_square: ExprId,
    derivative: ExprId,
) -> ExprId {
    let (scale, core) =
        split_signed_numeric_scale_single_core_for_calculus_presentation(ctx, derivative)
            .unwrap_or((BigRational::one(), derivative));
    if cas_ast::ordering::compare_expr(ctx, core, cos_arg).is_eq() {
        let three = ctx.num(3);
        let cos_cube = ctx.add_raw(Expr::Pow(cos_arg, three));
        return scale_expr_for_calculus_presentation(ctx, scale, cos_cube);
    }

    let term = ctx.add_raw(Expr::Mul(cos_square, derivative));
    let term = compact_small_power_exponents_for_calculus_presentation(ctx, term);
    let term =
        combine_matching_cos_powers_for_calculus_presentation(ctx, cos_arg, term).unwrap_or(term);
    compact_numeric_mul_factors_for_calculus_presentation(ctx, term)
}

fn push_scaled_tan_derivative_numerator_term(
    ctx: &mut Context,
    numerator_terms: &mut Vec<ExprId>,
    tan_scale: BigRational,
    reciprocal_trig_square: ExprId,
    derivative_multiplier: ExprId,
) {
    let tan_term = scale_expr_for_calculus_presentation(ctx, tan_scale, reciprocal_trig_square);
    let tan_term = ctx.add_raw(Expr::Mul(derivative_multiplier, tan_term));
    numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
        ctx, tan_term,
    ));
}

fn push_scaled_other_derivative_numerator_terms(
    ctx: &mut Context,
    numerator_terms: &mut Vec<ExprId>,
    derivative_multiplier: ExprId,
    other_derivatives: Vec<ExprId>,
) {
    for derivative in other_derivatives {
        let term = ctx.add_raw(Expr::Mul(derivative_multiplier, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
}

pub(super) fn sqrt_additive_tan_common_denominator_derivative_presentation(
    ctx: &mut Context,
    parts: SqrtAdditiveTanCommonDenominatorPresentationParts,
) -> (ExprId, ExprId) {
    let denominator_trig_arg =
        ctx.call_builtin(parts.common_trig_denominator_builtin, vec![parts.tan_arg]);
    let two = ctx.num(2);
    let denominator_trig_square = ctx.add_raw(Expr::Pow(denominator_trig_arg, two));

    let mut numerator_terms = Vec::new();
    let common_denominator = parts
        .common_denominator
        .filter(|_| !parts.reciprocal_derivative_scales.is_empty());
    let tan_numerator = if let Some(denominator) = common_denominator {
        scale_expr_for_calculus_presentation(ctx, parts.tan_scale, denominator)
    } else {
        rational_const_for_calculus_presentation(ctx, parts.tan_scale)
    };
    numerator_terms.push(tan_numerator);
    for scale in parts.reciprocal_derivative_scales {
        numerator_terms.push(scale_expr_for_calculus_presentation(
            ctx,
            scale,
            denominator_trig_square,
        ));
    }
    for derivative in parts.other_derivatives {
        let mut term = compact_tan_sqrt_common_denominator_numerator_term(
            ctx,
            denominator_trig_arg,
            denominator_trig_square,
            derivative,
        );
        if let Some(denominator) = common_denominator {
            term = ctx.add_raw(Expr::Mul(denominator, term));
            term = compact_numeric_mul_factors_for_calculus_presentation(ctx, term);
        }
        numerator_terms.push(term);
    }
    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);

    let denominator_scale =
        rational_const_for_calculus_presentation(ctx, BigRational::from_integer(2.into()));
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![parts.radicand]);
    let denominator = if let Some(common_denominator) = common_denominator {
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[
                denominator_scale,
                common_denominator,
                denominator_trig_square,
                sqrt_radicand,
            ],
        )
    } else {
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[denominator_scale, denominator_trig_square, sqrt_radicand],
        )
    };
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    (cas_ast::hold::wrap_hold(ctx, compact), denominator_trig_arg)
}

pub(super) fn sqrt_additive_tan_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    parts: SqrtAdditiveTanDerivativePresentationParts,
    sqrt_arg: ExprId,
    sqrt_scale: BigRational,
) -> Option<ExprId> {
    if sqrt_scale.is_zero() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, parts.radicand);
    let reciprocal_trig_arg = ctx.call_builtin(parts.reciprocal_trig_builtin, vec![parts.tan_arg]);
    let reciprocal_trig_square = ctx.add_raw(Expr::Pow(reciprocal_trig_arg, two));
    let two_sqrt_arg = ctx.add_raw(Expr::Mul(two, sqrt_arg_root));

    let mut numerator_terms = Vec::new();
    push_scaled_tan_derivative_numerator_term(
        ctx,
        &mut numerator_terms,
        parts.tan_scale,
        reciprocal_trig_square,
        two_sqrt_arg,
    );
    numerator_terms.push(rational_const_for_calculus_presentation(ctx, sqrt_scale));
    push_scaled_other_derivative_numerator_terms(
        ctx,
        &mut numerator_terms,
        two_sqrt_arg,
        parts.other_derivatives,
    );

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, sqrt_arg_root, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(super) fn sqrt_additive_tan_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    parts: SqrtAdditiveTanDerivativePresentationParts,
    sqrt_arg: ExprId,
    sqrt_scale: BigRational,
    reciprocal_sqrt_scale: BigRational,
) -> Option<ExprId> {
    if sqrt_scale.is_zero() || reciprocal_sqrt_scale.is_zero() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, parts.radicand);
    let reciprocal_trig_arg = ctx.call_builtin(parts.reciprocal_trig_builtin, vec![parts.tan_arg]);
    let reciprocal_trig_square = ctx.add_raw(Expr::Pow(reciprocal_trig_arg, two));
    let arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_arg, sqrt_arg_root]);
    let two_arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, sqrt_arg, sqrt_arg_root]);

    let mut numerator_terms = Vec::new();
    push_scaled_tan_derivative_numerator_term(
        ctx,
        &mut numerator_terms,
        parts.tan_scale,
        reciprocal_trig_square,
        two_arg_times_sqrt_arg,
    );
    push_scaled_other_derivative_numerator_terms(
        ctx,
        &mut numerator_terms,
        two_arg_times_sqrt_arg,
        parts.other_derivatives,
    );
    numerator_terms.push(scale_expr_for_calculus_presentation(
        ctx, sqrt_scale, sqrt_arg,
    ));
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, arg_times_sqrt_arg, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(super) fn sqrt_additive_tan_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    parts: SqrtAdditiveTanDerivativePresentationParts,
    reciprocal_sqrt_arg: ExprId,
    reciprocal_sqrt_scale: BigRational,
) -> Option<ExprId> {
    if reciprocal_sqrt_scale.is_zero() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, reciprocal_sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, parts.radicand);
    let reciprocal_trig_arg = ctx.call_builtin(parts.reciprocal_trig_builtin, vec![parts.tan_arg]);
    let reciprocal_trig_square = ctx.add_raw(Expr::Pow(reciprocal_trig_arg, two));
    let arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[reciprocal_sqrt_arg, sqrt_arg_root]);
    let two_arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, reciprocal_sqrt_arg, sqrt_arg_root]);

    let mut numerator_terms = Vec::new();
    push_scaled_tan_derivative_numerator_term(
        ctx,
        &mut numerator_terms,
        parts.tan_scale,
        reciprocal_trig_square,
        two_arg_times_sqrt_arg,
    );
    push_scaled_other_derivative_numerator_terms(
        ctx,
        &mut numerator_terms,
        two_arg_times_sqrt_arg,
        parts.other_derivatives,
    );
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, arg_times_sqrt_arg, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

fn combine_matching_cos_powers_for_calculus_presentation(
    ctx: &mut Context,
    cos_arg: ExprId,
    expr: ExprId,
) -> Option<ExprId> {
    if let Expr::Neg(inner) = ctx.get(expr).clone() {
        let combined = combine_matching_cos_powers_for_calculus_presentation(ctx, cos_arg, inner)?;
        return Some(ctx.add_raw(Expr::Neg(combined)));
    }

    let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
    if factors.len() < 2 {
        return None;
    }

    let mut cos_power_sum: i64 = 0;
    let mut non_cos_factors = Vec::new();
    for factor in factors {
        if let Some(power) =
            matching_cos_integer_power_for_calculus_presentation(ctx, cos_arg, factor)
        {
            cos_power_sum += power;
        } else {
            non_cos_factors.push(factor);
        }
    }

    if cos_power_sum <= 1 {
        return None;
    }

    let cos_power = if cos_power_sum == 1 {
        cos_arg
    } else {
        let exponent = ctx.num(cos_power_sum);
        ctx.add_raw(Expr::Pow(cos_arg, exponent))
    };
    non_cos_factors.push(cos_power);
    Some(cas_math::expr_nary::build_balanced_mul(
        ctx,
        &non_cos_factors,
    ))
}

fn matching_cos_integer_power_for_calculus_presentation(
    ctx: &Context,
    cos_arg: ExprId,
    expr: ExprId,
) -> Option<i64> {
    if cas_ast::ordering::compare_expr(ctx, expr, cos_arg).is_eq() {
        return Some(1);
    }

    let Expr::Pow(base, exp) = ctx.get(expr) else {
        return None;
    };
    if !cas_ast::ordering::compare_expr(ctx, *base, cos_arg).is_eq() {
        return None;
    }
    let value = cas_ast::views::as_rational_const(ctx, *exp, 8)?;
    value
        .is_integer()
        .then(|| value.to_integer().to_i64())
        .flatten()
}
