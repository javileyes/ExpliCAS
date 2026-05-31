use super::polynomial_support::{
    polynomial_radicand_for_calculus_presentation,
    split_polynomial_content_for_calculus_presentation,
};
use super::presentation_utils::calculus_sqrt_like_radicand;
use super::scalar_presentation::{
    negate_calculus_presentation, nonzero_rational_parts, rational_const_for_calculus_presentation,
    signed_numerator_for_calculus_presentation,
};
use cas_ast::{BuiltinFn, Constant, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

fn elementary_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    #[derive(Clone, Copy)]
    enum SqrtChainShape {
        NumeratorFunction(BuiltinFn),
        ExponentialPower,
        DenominatorSquare(BuiltinFn),
        NumeratorOverDenominatorSquare {
            numerator_fn: BuiltinFn,
            denominator_fn: BuiltinFn,
        },
        ReciprocalTrigProduct {
            primary_fn: BuiltinFn,
            companion_fn: BuiltinFn,
        },
    }

    let (arg, shape, sign) = match ctx.get(target).clone() {
        Expr::Function(fn_id, args) => {
            if args.len() != 1 {
                return None;
            }
            let (shape, sign) = match ctx.builtin_of(fn_id) {
                Some(BuiltinFn::Exp) => (
                    SqrtChainShape::NumeratorFunction(BuiltinFn::Exp),
                    BigRational::one(),
                ),
                Some(BuiltinFn::Sin) => (
                    SqrtChainShape::NumeratorFunction(BuiltinFn::Cos),
                    BigRational::one(),
                ),
                Some(BuiltinFn::Cos) => (
                    SqrtChainShape::NumeratorFunction(BuiltinFn::Sin),
                    -BigRational::one(),
                ),
                Some(BuiltinFn::Tan) => (
                    SqrtChainShape::DenominatorSquare(BuiltinFn::Cos),
                    BigRational::one(),
                ),
                Some(BuiltinFn::Cot) => (
                    SqrtChainShape::DenominatorSquare(BuiltinFn::Sin),
                    -BigRational::one(),
                ),
                Some(BuiltinFn::Sec) => (
                    SqrtChainShape::ReciprocalTrigProduct {
                        primary_fn: BuiltinFn::Sec,
                        companion_fn: BuiltinFn::Tan,
                    },
                    BigRational::one(),
                ),
                Some(BuiltinFn::Csc) => (
                    SqrtChainShape::ReciprocalTrigProduct {
                        primary_fn: BuiltinFn::Csc,
                        companion_fn: BuiltinFn::Cot,
                    },
                    -BigRational::one(),
                ),
                Some(BuiltinFn::Sinh) => (
                    SqrtChainShape::NumeratorFunction(BuiltinFn::Cosh),
                    BigRational::one(),
                ),
                Some(BuiltinFn::Cosh) => (
                    SqrtChainShape::NumeratorFunction(BuiltinFn::Sinh),
                    BigRational::one(),
                ),
                Some(BuiltinFn::Tanh) => (
                    SqrtChainShape::DenominatorSquare(BuiltinFn::Cosh),
                    BigRational::one(),
                ),
                _ => return None,
            };
            (args[0], shape, sign)
        }
        Expr::Div(numerator, denominator) => {
            let numerator_sign = cas_ast::views::as_rational_const(ctx, numerator, 8)
                .filter(|value| value == &BigRational::one() || value == &-BigRational::one())?;
            let Expr::Function(den_fn_id, den_args) = ctx.get(denominator).clone() else {
                return None;
            };
            if den_args.len() != 1 {
                return None;
            }
            let (shape, sign) = match ctx.builtin_of(den_fn_id) {
                Some(BuiltinFn::Cos) => (
                    SqrtChainShape::ReciprocalTrigProduct {
                        primary_fn: BuiltinFn::Sec,
                        companion_fn: BuiltinFn::Tan,
                    },
                    BigRational::one(),
                ),
                Some(BuiltinFn::Sin) => (
                    SqrtChainShape::ReciprocalTrigProduct {
                        primary_fn: BuiltinFn::Csc,
                        companion_fn: BuiltinFn::Cot,
                    },
                    -BigRational::one(),
                ),
                Some(BuiltinFn::Cosh) => (
                    SqrtChainShape::NumeratorOverDenominatorSquare {
                        numerator_fn: BuiltinFn::Sinh,
                        denominator_fn: BuiltinFn::Cosh,
                    },
                    -BigRational::one(),
                ),
                Some(BuiltinFn::Sinh) => (
                    SqrtChainShape::NumeratorOverDenominatorSquare {
                        numerator_fn: BuiltinFn::Cosh,
                        denominator_fn: BuiltinFn::Sinh,
                    },
                    -BigRational::one(),
                ),
                _ => return None,
            };
            (den_args[0], shape, sign * numerator_sign)
        }
        Expr::Pow(base, exp) if matches!(ctx.get(base), Expr::Constant(Constant::E)) => {
            (exp, SqrtChainShape::ExponentialPower, BigRational::one())
        }
        _ => return None,
    };

    let radicand = calculus_sqrt_like_radicand(ctx, arg)?;
    let radicand_poly = polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name)?;
    let derivative_poly = radicand_poly.derivative();
    if derivative_poly.is_zero() {
        return Some(ctx.num(0));
    }
    let derivative = derivative_poly.to_expr(ctx);
    let (derivative_core, derivative_content) =
        split_polynomial_content_for_calculus_presentation(ctx, derivative);
    let coefficient = sign * derivative_content * BigRational::new(1.into(), 2.into());
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&coefficient)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let numerator_function = match shape {
        SqrtChainShape::NumeratorFunction(outer_derivative_fn) => {
            Some(ctx.call_builtin(outer_derivative_fn, vec![sqrt_radicand]))
        }
        SqrtChainShape::ExponentialPower => {
            let e = ctx.add(Expr::Constant(Constant::E));
            Some(ctx.add(Expr::Pow(e, sqrt_radicand)))
        }
        SqrtChainShape::DenominatorSquare(_) => None,
        SqrtChainShape::NumeratorOverDenominatorSquare { numerator_fn, .. } => {
            Some(ctx.call_builtin(numerator_fn, vec![sqrt_radicand]))
        }
        SqrtChainShape::ReciprocalTrigProduct {
            primary_fn,
            companion_fn,
        } => {
            let primary = ctx.call_builtin(primary_fn, vec![sqrt_radicand]);
            let companion = ctx.call_builtin(companion_fn, vec![sqrt_radicand]);
            Some(cas_math::expr_nary::build_balanced_mul(
                ctx,
                &[primary, companion],
            ))
        }
    };
    let derivative_core_is_one = cas_ast::views::as_rational_const(ctx, derivative_core, 8)
        .is_some_and(|value| value.is_one());
    let numerator_core = match numerator_function {
        Some(numerator_function) if derivative_core_is_one => numerator_function,
        Some(numerator_function) => {
            cas_math::expr_nary::build_balanced_mul(ctx, &[derivative_core, numerator_function])
        }
        None => derivative_core,
    };
    let numerator =
        signed_numerator_for_calculus_presentation(ctx, numerator_coeff, numerator_core);

    let core_denominator = match shape {
        SqrtChainShape::DenominatorSquare(denominator_fn) => {
            let denominator_arg = ctx.call_builtin(denominator_fn, vec![sqrt_radicand]);
            let two = ctx.num(2);
            let denominator_square = ctx.add(Expr::Pow(denominator_arg, two));
            cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, denominator_square])
        }
        SqrtChainShape::NumeratorOverDenominatorSquare { denominator_fn, .. } => {
            let denominator_arg = ctx.call_builtin(denominator_fn, vec![sqrt_radicand]);
            let two = ctx.num(2);
            let denominator_square = ctx.add(Expr::Pow(denominator_arg, two));
            cas_math::expr_nary::build_balanced_mul(ctx, &[sqrt_radicand, denominator_square])
        }
        SqrtChainShape::ReciprocalTrigProduct { .. } => sqrt_radicand,
        _ => sqrt_radicand,
    };
    let denominator = if denominator_coeff == BigRational::one() {
        core_denominator
    } else {
        let denominator_scale = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_scale, core_denominator])
    };

    Some(ctx.add(Expr::Div(numerator, denominator)))
}

pub(super) fn signed_elementary_sqrt_polynomial_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Neg(inner) = ctx.get(target).clone() else {
        return elementary_sqrt_polynomial_derivative_presentation(ctx, target, var_name);
    };

    let compact = elementary_sqrt_polynomial_derivative_presentation(ctx, inner, var_name)?;
    Some(negate_calculus_presentation(ctx, compact))
}
