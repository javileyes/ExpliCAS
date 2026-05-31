use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::scalar_presentation::{
    scale_expr_by_sqrt_positive_rational_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::{One, Signed};

pub(super) fn bounded_inverse_trig_self_normalized_projection_derivative_presentation(
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

    let (numerator_arg, denominator_radicand) =
        bounded_inverse_trig_self_normalized_projection_arg(ctx, args[0])?;
    let numerator_poly =
        polynomial_radicand_for_calculus_presentation(ctx, numerator_arg, var_name)?;
    if numerator_poly.degree() > 2 {
        return None;
    }
    let denominator_poly =
        polynomial_radicand_for_calculus_presentation(ctx, denominator_radicand, var_name)?;
    let gap_poly = denominator_poly.sub(&numerator_poly.mul(&numerator_poly));
    if gap_poly.degree() != 0 {
        return None;
    }
    let gap_constant = gap_poly.coeffs.first().cloned()?;
    if !gap_constant.is_positive() {
        return None;
    }

    let mut derivative_poly = numerator_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let mut derivative_sign = derivative_sign;
    if derivative_poly.leading_coeff().is_negative() {
        derivative_poly = derivative_poly.neg();
        derivative_sign = -derivative_sign;
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let numerator = signed_numerator_for_calculus_presentation(
        ctx,
        derivative_sign * derivative_content,
        derivative_core,
    );
    let numerator = scale_expr_by_sqrt_positive_rational_for_calculus_presentation(
        ctx,
        gap_constant,
        numerator,
    );
    let denominator = denominator_radicand;
    let compact = ctx.add(Expr::Div(numerator, denominator));

    Some(compact)
}

fn bounded_inverse_trig_self_normalized_projection_arg(
    ctx: &mut Context,
    arg: ExprId,
) -> Option<(ExprId, ExprId)> {
    match ctx.get(arg).clone() {
        Expr::Neg(inner) => {
            let (numerator, denominator_radicand) =
                bounded_inverse_trig_self_normalized_projection_arg(ctx, inner)?;
            let numerator = ctx.add(Expr::Neg(numerator));
            Some((numerator, denominator_radicand))
        }
        Expr::Div(numerator, denominator) => {
            Some((numerator, extract_square_root_base(ctx, denominator)?))
        }
        Expr::Mul(_, _) => {
            let factors = cas_math::expr_nary::mul_leaves(ctx, arg);
            let mut numerator_factors = Vec::new();
            let mut denominator_radicand = None;

            for factor in factors {
                match ctx.get(factor) {
                    Expr::Pow(base, exp)
                        if cas_ast::views::as_rational_const(ctx, *exp, 8)
                            == Some(BigRational::new((-1).into(), 2.into())) =>
                    {
                        if denominator_radicand.replace(*base).is_some() {
                            return None;
                        }
                    }
                    _ => numerator_factors.push(factor),
                }
            }

            let denominator_radicand = denominator_radicand?;
            let numerator = match numerator_factors.as_slice() {
                [] => ctx.num(1),
                [only] => *only,
                [first, rest @ ..] => {
                    let mut product = *first;
                    for factor in rest {
                        product = ctx.add(Expr::Mul(product, *factor));
                    }
                    product
                }
            };

            Some((numerator, denominator_radicand))
        }
        _ => None,
    }
}
