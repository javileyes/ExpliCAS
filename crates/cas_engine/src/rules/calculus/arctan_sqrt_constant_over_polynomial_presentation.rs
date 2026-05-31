use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::{
    add_rational_for_calculus_presentation, exact_positive_rational_sqrt_for_calculus_presentation,
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    scale_expr_for_calculus_presentation,
};
use super::{arccot_sqrt_radicand_arg, arctan_sqrt_radicand_arg};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed};

pub(super) fn arctan_sqrt_constant_over_polynomial_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let radicand = arctan_sqrt_radicand_arg(ctx, target)?;
    sqrt_constant_over_polynomial_presentation(ctx, radicand, var_name, derivative_sign)
}

pub(super) fn arccot_sqrt_constant_over_polynomial_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let radicand = arccot_sqrt_radicand_arg(ctx, target)?;
    sqrt_constant_over_polynomial_presentation(ctx, radicand, var_name, -BigRational::one())
}

fn sqrt_constant_over_polynomial_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    var_name: &str,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let Expr::Div(num, den) = ctx.get(radicand).clone() else {
        return None;
    };
    let numerator_value = cas_ast::views::as_rational_const(ctx, num, 8)?;
    if !numerator_value.is_positive() {
        return None;
    }

    let denominator_poly = polynomial_radicand_for_calculus_presentation(ctx, den, var_name)?;
    let derivative_poly = denominator_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let sqrt_numerator_value =
        exact_positive_rational_sqrt_for_calculus_presentation(&numerator_value);
    let displayed_numerator_value = sqrt_numerator_value
        .clone()
        .unwrap_or_else(|| numerator_value.clone());
    let coefficient = -derivative_sign
        * displayed_numerator_value
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let denominator_plus_numerator =
        add_rational_for_calculus_presentation(ctx, den, numerator_value);
    let core_denominator = if sqrt_numerator_value.is_some() {
        let sqrt_denominator = ctx.call_builtin(BuiltinFn::Sqrt, vec![den]);
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[sqrt_denominator, denominator_plus_numerator],
        )
    } else {
        let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
        cas_math::expr_nary::build_balanced_mul(
            ctx,
            &[sqrt_radicand, den, denominator_plus_numerator],
        )
    };
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}
