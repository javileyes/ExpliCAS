use super::domain_checks::positive_polynomial_radicand_required_conditions;
use super::inverse_tangent_polynomial_root_derivative_presentation::arctan_sqrt_polynomial_derivative_presentation;
use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    rational_polynomial_content_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::calculus_sqrt_like_radicand;
use super::scalar_presentation::{
    add_rational_for_calculus_presentation, nonzero_rational_parts,
    rational_const_for_calculus_presentation,
    reciprocal_integer_radicand_content_for_calculus_presentation,
    scale_expr_for_calculus_presentation, signed_numerator_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

fn arctan_reciprocal_abs_inverse_sqrt_radicand_arg(
    ctx: &Context,
    target: ExprId,
) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan)
        )
    {
        return None;
    }

    let Expr::Div(num, den) = ctx.get(args[0]) else {
        return None;
    };
    let numerator = cas_ast::views::as_rational_const(ctx, *num, 8)?;
    if !numerator.is_one() {
        return None;
    }
    let Expr::Function(abs_fn, abs_args) = ctx.get(*den) else {
        return None;
    };
    if abs_args.len() != 1 || !ctx.is_builtin(*abs_fn, BuiltinFn::Abs) {
        return None;
    }
    let Expr::Pow(base, exp) = ctx.get(abs_args[0]) else {
        return None;
    };
    if cas_ast::views::as_rational_const(ctx, *exp, 8)
        != Some(BigRational::new((-1).into(), 2.into()))
    {
        return None;
    }

    Some(*base)
}

fn reciprocal_sqrt_radicand_arg_for_inverse_tangent(
    ctx: &Context,
    target: ExprId,
) -> Option<(ExprId, BigRational, BigRational)> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1 {
        return None;
    }

    let sign = match ctx.builtin_of(*fn_id) {
        Some(BuiltinFn::Atan | BuiltinFn::Arctan) => -BigRational::one(),
        Some(BuiltinFn::Acot | BuiltinFn::Arccot) => BigRational::one(),
        _ => return None,
    };

    let mut abs_wrapped = false;
    let arg = match ctx.get(args[0]) {
        Expr::Function(abs_fn, abs_args)
            if abs_args.len() == 1 && ctx.is_builtin(*abs_fn, BuiltinFn::Abs) =>
        {
            abs_wrapped = true;
            abs_args[0]
        }
        _ => args[0],
    };

    let (radicand, mut argument_scale) = match ctx.get(arg) {
        Expr::Div(num, den) => {
            let scale = cas_ast::views::as_rational_const(ctx, *num, 8)?;
            if scale.is_zero() {
                return None;
            }
            (calculus_sqrt_like_radicand(ctx, *den)?, scale)
        }
        Expr::Mul(_, _) => {
            let factors = cas_math::expr_nary::MulView::from_expr(ctx, args[0]).factors;
            let mut scale = BigRational::one();
            let mut radicand = None;

            for factor in factors {
                if let Some(value) = cas_ast::views::as_rational_const(ctx, factor, 8) {
                    scale *= value;
                    continue;
                }

                let Expr::Pow(base, exp) = ctx.get(factor) else {
                    return None;
                };
                if cas_ast::views::as_rational_const(ctx, *exp, 8)
                    != Some(BigRational::new((-1).into(), 2.into()))
                    || radicand.replace(*base).is_some()
                {
                    return None;
                }
            }

            if scale.is_zero() {
                return None;
            }
            (radicand?, scale)
        }
        Expr::Pow(base, exp)
            if cas_ast::views::as_rational_const(ctx, *exp, 8)
                == Some(BigRational::new((-1).into(), 2.into())) =>
        {
            (*base, BigRational::one())
        }
        _ => return None,
    };
    if abs_wrapped {
        argument_scale = argument_scale.abs();
    }

    Some((radicand, sign, argument_scale))
}

pub(super) fn arctan_sqrt_reciprocal_content_presentation(
    ctx: &mut Context,
    radicand: ExprId,
    radicand_poly: &Polynomial,
    derivative_sign: BigRational,
) -> Option<ExprId> {
    let radicand_content = rational_polynomial_content_for_calculus_presentation(radicand_poly);
    let reciprocal_content =
        reciprocal_integer_radicand_content_for_calculus_presentation(&radicand_content)?;
    let primitive_radicand_poly = radicand_poly.div_scalar(&radicand_content);
    let derivative_poly = primitive_radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let primitive_radicand = primitive_radicand_poly.to_expr(ctx);
    let compact_gap =
        add_rational_for_calculus_presentation(ctx, primitive_radicand, reciprocal_content);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, compact_gap]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };
    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn arctan_reciprocal_abs_inverse_sqrt_polynomial_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let radicand = arctan_reciprocal_abs_inverse_sqrt_radicand_arg(ctx, target)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let synthetic_arctan = ctx.call_builtin(BuiltinFn::Arctan, vec![sqrt_radicand]);
    let result = arctan_sqrt_polynomial_derivative_presentation(
        ctx,
        synthetic_arctan,
        var_name,
        BigRational::one(),
    )?;
    let required_conditions =
        positive_polynomial_radicand_required_conditions(radicand, &radicand_poly);
    Some((result, required_conditions))
}

pub(super) fn inverse_tangent_reciprocal_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (radicand, derivative_sign, argument_scale) =
        reciprocal_sqrt_radicand_arg_for_inverse_tangent(ctx, target)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let argument_scale_square = argument_scale.clone() * argument_scale.clone();
    let radicand_content = rational_polynomial_content_for_calculus_presentation(&radicand_poly);
    if radicand_content.is_positive() && !radicand_content.is_one() {
        let primitive_radicand_poly = radicand_poly.div_scalar(&radicand_content);
        let derivative_poly = primitive_radicand_poly.derivative();
        if derivative_poly.is_zero() {
            return Some(ctx.num(0));
        }
        let derivative = derivative_poly.to_expr(ctx);

        let (derivative_core, derivative_content) =
            split_polynomial_content_for_calculus_presentation(ctx, derivative);
        let coefficient = derivative_sign
            * argument_scale.clone()
            * derivative_content
            * BigRational::new(1.into(), 2.into());
        let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
        let numerator =
            signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

        let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
        let primitive_radicand = primitive_radicand_poly.to_expr(ctx);
        let radicand_plus_one = add_rational_for_calculus_presentation(
            ctx,
            primitive_radicand,
            argument_scale_square / radicand_content,
        );
        let core_denominator =
            cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_plus_one]);
        let denominator = if denominator_coeff == BigRational::one() {
            core_denominator
        } else {
            let denominator_scale =
                rational_const_for_calculus_presentation(ctx, denominator_coeff);
            cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
        };

        return Some(ctx.add(Expr::Div(numerator, denominator)));
    }

    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);

    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = derivative_sign
        * argument_scale
        * derivative_content
        * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, derivative_core);

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let radicand_plus_one =
        add_rational_for_calculus_presentation(ctx, radicand, argument_scale_square);
    let core_denominator =
        cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, radicand_plus_one]);
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn inverse_tangent_reciprocal_sqrt_polynomial_derivative_shortcut(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let (radicand, _, _) = reciprocal_sqrt_radicand_arg_for_inverse_tangent(ctx, target)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let result =
        inverse_tangent_reciprocal_sqrt_polynomial_derivative_presentation(ctx, target, var_name)?;
    let required_conditions =
        positive_polynomial_radicand_required_conditions(radicand, &radicand_poly);
    Some((result, required_conditions))
}
