use cas_ast::{Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Zero};

use super::presentation_utils::sqrt_raw_for_calculus_presentation;
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    scale_expr_for_calculus_presentation,
};
use super::{
    compact_numeric_mul_factors_for_calculus_presentation,
    compact_small_power_exponents_for_calculus_presentation,
};

pub(super) fn sqrt_variable_derivative_term_for_calculus_presentation(
    ctx: &mut Context,
    sqrt_scale: BigRational,
    sqrt_arg: ExprId,
) -> Option<ExprId> {
    let coefficient = sqrt_scale * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_arg_root
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, sqrt_arg_root])
    };
    Some(ctx.add_raw(Expr::Div(numerator, denominator)))
}

pub(super) fn reciprocal_sqrt_derivative_term_for_calculus_presentation(
    ctx: &mut Context,
    reciprocal_sqrt_scale: BigRational,
    reciprocal_sqrt_arg: ExprId,
) -> ExprId {
    let neg_three_half = ctx.rational(-3, 2);
    let reciprocal_sqrt_cubed = ctx.add_raw(Expr::Pow(reciprocal_sqrt_arg, neg_three_half));
    scale_expr_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale * BigRational::new(1.into(), 2.into()),
        reciprocal_sqrt_cubed,
    )
}

pub(super) fn sqrt_additive_generic_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if derivative_terms.is_empty() {
        return None;
    }

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &derivative_terms);
    let numerator = compact_small_power_exponents_for_calculus_presentation(ctx, numerator);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let two = ctx.num(2);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[two, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(super) fn sqrt_additive_generic_common_denominator_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    common_denominator: ExprId,
    reciprocal_derivative_scales: Vec<BigRational>,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if reciprocal_derivative_scales.is_empty() || derivative_terms.is_empty() {
        return None;
    }

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(common_denominator, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    for scale in reciprocal_derivative_scales {
        numerator_terms.push(rational_const_for_calculus_presentation(ctx, scale));
    }

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let two = ctx.num(2);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, common_denominator, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(super) fn sqrt_additive_generic_common_denominator_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    common_denominator: ExprId,
    sqrt_arg: ExprId,
    sqrt_scale: BigRational,
    reciprocal_derivative_scales: Vec<BigRational>,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if sqrt_scale.is_zero()
        || reciprocal_derivative_scales.is_empty()
        || derivative_terms.is_empty()
    {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let two_common_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, common_denominator, sqrt_arg_root]);
    let two_sqrt_arg = ctx.add_raw(Expr::Mul(two, sqrt_arg_root));

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_common_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    for scale in reciprocal_derivative_scales {
        let term = scale_expr_for_calculus_presentation(ctx, scale, two_sqrt_arg);
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(scale_expr_for_calculus_presentation(
        ctx,
        sqrt_scale,
        common_denominator,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[four, common_denominator, sqrt_arg_root, sqrt_radicand],
    );
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(super) fn sqrt_additive_generic_common_denominator_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    common_denominator: ExprId,
    reciprocal_sqrt_scale: BigRational,
    reciprocal_derivative_scales: Vec<BigRational>,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if reciprocal_sqrt_scale.is_zero()
        || reciprocal_derivative_scales.is_empty()
        || derivative_terms.is_empty()
    {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_common_denominator = sqrt_raw_for_calculus_presentation(ctx, common_denominator);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let two_common_sqrt_denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[two, common_denominator, sqrt_common_denominator],
    );
    let two_sqrt_denominator = ctx.add_raw(Expr::Mul(two, sqrt_common_denominator));

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_common_sqrt_denominator, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    for scale in reciprocal_derivative_scales {
        let term = scale_expr_for_calculus_presentation(ctx, scale, two_sqrt_denominator);
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[
            four,
            common_denominator,
            sqrt_common_denominator,
            sqrt_radicand,
        ],
    );
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(super) fn sqrt_additive_generic_common_denominator_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    common_denominator: ExprId,
    sqrt_scale: BigRational,
    reciprocal_sqrt_scale: BigRational,
    reciprocal_derivative_scales: Vec<BigRational>,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if sqrt_scale.is_zero()
        || reciprocal_sqrt_scale.is_zero()
        || reciprocal_derivative_scales.is_empty()
        || derivative_terms.is_empty()
    {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_common_denominator = sqrt_raw_for_calculus_presentation(ctx, common_denominator);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let two_common_sqrt_denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[two, common_denominator, sqrt_common_denominator],
    );
    let two_sqrt_denominator = ctx.add_raw(Expr::Mul(two, sqrt_common_denominator));

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_common_sqrt_denominator, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    for scale in reciprocal_derivative_scales {
        let term = scale_expr_for_calculus_presentation(ctx, scale, two_sqrt_denominator);
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(scale_expr_for_calculus_presentation(
        ctx,
        sqrt_scale,
        common_denominator,
    ));
    numerator_terms.push(rational_const_for_calculus_presentation(
        ctx,
        -reciprocal_sqrt_scale,
    ));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator = cas_math::expr_nary::build_balanced_mul(
        ctx,
        &[
            four,
            common_denominator,
            sqrt_common_denominator,
            sqrt_radicand,
        ],
    );
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(super) fn sqrt_additive_generic_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    sqrt_arg: ExprId,
    sqrt_scale: BigRational,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if sqrt_scale.is_zero() || derivative_terms.is_empty() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let two_sqrt_arg = ctx.add_raw(Expr::Mul(two, sqrt_arg_root));

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
    numerator_terms.push(rational_const_for_calculus_presentation(ctx, sqrt_scale));

    let numerator = cas_math::expr_nary::build_balanced_add(ctx, &numerator_terms);
    let numerator = compact_numeric_mul_factors_for_calculus_presentation(ctx, numerator);
    let denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[four, sqrt_arg_root, sqrt_radicand]);
    let compact = ctx.add_raw(Expr::Div(numerator, denominator));
    Some(cas_ast::hold::wrap_hold(ctx, compact))
}

pub(super) fn sqrt_additive_generic_sqrt_and_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    sqrt_arg: ExprId,
    sqrt_scale: BigRational,
    reciprocal_sqrt_scale: BigRational,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if sqrt_scale.is_zero() || reciprocal_sqrt_scale.is_zero() || derivative_terms.is_empty() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_arg, sqrt_arg_root]);
    let two_arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, sqrt_arg, sqrt_arg_root]);

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_arg_times_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
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

pub(super) fn sqrt_additive_generic_reciprocal_sqrt_variable_derivative_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    reciprocal_sqrt_arg: ExprId,
    reciprocal_sqrt_scale: BigRational,
    derivative_terms: Vec<ExprId>,
) -> Option<ExprId> {
    if reciprocal_sqrt_scale.is_zero() || derivative_terms.is_empty() {
        return None;
    }

    let two = ctx.num(2);
    let four = ctx.num(4);
    let sqrt_arg_root = sqrt_raw_for_calculus_presentation(ctx, reciprocal_sqrt_arg);
    let sqrt_radicand = sqrt_raw_for_calculus_presentation(ctx, radicand);
    let arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[reciprocal_sqrt_arg, sqrt_arg_root]);
    let two_arg_times_sqrt_arg =
        cas_math::expr_nary::build_balanced_mul(ctx, &[two, reciprocal_sqrt_arg, sqrt_arg_root]);

    let mut numerator_terms = Vec::new();
    for derivative in derivative_terms {
        let term = ctx.add_raw(Expr::Mul(two_arg_times_sqrt_arg, derivative));
        numerator_terms.push(compact_numeric_mul_factors_for_calculus_presentation(
            ctx, term,
        ));
    }
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
