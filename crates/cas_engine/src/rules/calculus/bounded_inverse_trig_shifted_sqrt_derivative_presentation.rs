use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::result_presentation::scale_compact_derivative_by_rational;
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
    rational_scaled_single_factor, signed_numerator_for_calculus_presentation,
};
use super::scaled_sqrt_args::scaled_sqrt_radicand_for_calculus_presentation;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

fn shifted_unit_interval_sqrt_arg_for_calculus_presentation(
    ctx: &Context,
    expr: ExprId,
) -> Option<(ExprId, BigRational)> {
    let two = BigRational::from_integer(2.into());
    match ctx.get(expr) {
        Expr::Sub(left, right) => {
            if cas_ast::views::as_rational_const(ctx, *right, 8) == Some(BigRational::one()) {
                let (scale, radicand) = scaled_sqrt_radicand_for_calculus_presentation(ctx, *left)?;
                if scale == two {
                    return Some((radicand, BigRational::one()));
                }
            }

            if cas_ast::views::as_rational_const(ctx, *left, 8) == Some(BigRational::one()) {
                let (scale, radicand) =
                    scaled_sqrt_radicand_for_calculus_presentation(ctx, *right)?;
                if scale == two {
                    return Some((radicand, -BigRational::one()));
                }
            }

            None
        }
        _ => None,
    }
}

pub(super) fn unit_interval_bounded_inverse_trig_shifted_sqrt_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    let derivative_sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Arcsin | BuiltinFn::Asin) => BigRational::one(),
        Some(BuiltinFn::Arccos | BuiltinFn::Acos) => -BigRational::one(),
        _ => return None,
    };
    if args.len() != 1 {
        return None;
    }

    let (radicand, arg_sign) =
        shifted_unit_interval_sqrt_arg_for_calculus_presentation(ctx, args[0])?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);

    let coefficient =
        derivative_sign * arg_sign * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let gap = ctx.add(Expr::Sub(sqrt_radicand, radicand));
    let sqrt_gap = ctx.call_builtin(BuiltinFn::Sqrt, vec![gap]);
    let core_denominator = cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, sqrt_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn constant_scaled_unit_interval_bounded_inverse_trig_shifted_sqrt_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (scale, inner) = rational_scaled_single_factor(ctx, target)?;
    let derivative = unit_interval_bounded_inverse_trig_shifted_sqrt_derivative_presentation(
        ctx, inner, var_name,
    )?;

    Some(scale_compact_derivative_by_rational(ctx, derivative, scale))
}

pub(super) fn unit_interval_bounded_inverse_trig_shifted_sqrt_family_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    if let Some(compact) = unit_interval_bounded_inverse_trig_shifted_sqrt_derivative_presentation(
        ctx, target, var_name,
    ) {
        return Some(compact);
    }

    constant_scaled_unit_interval_bounded_inverse_trig_shifted_sqrt_derivative_presentation(
        ctx, target, var_name,
    )
}
