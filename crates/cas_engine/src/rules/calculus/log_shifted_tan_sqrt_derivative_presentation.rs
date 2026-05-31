//! Log derivative presentation for shifted tangent sqrt-chain arguments.

use super::presentation_utils::calculus_sqrt_like_radicand;
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
};
use super::sqrt_chain_factor_presentation::sqrt_chain_linear_derivative_coeff;
use super::unary_function_presentation::{
    signed_unary_builtin_arg_for_calculus, unary_builtin_arg_for_calculus,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};

struct ShiftedTangentLogArg {
    tan_arg: ExprId,
    tan_sign: BigRational,
    shift: BigRational,
}

fn shifted_tangent_log_arg(ctx: &Context, expr: ExprId) -> Option<ShiftedTangentLogArg> {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            if let Some(shift) =
                cas_ast::views::as_rational_const(ctx, *left, 8).filter(|value| !value.is_zero())
            {
                let (tan_arg, tan_sign) =
                    signed_unary_builtin_arg_for_calculus(ctx, *right, BuiltinFn::Tan)?;
                Some(ShiftedTangentLogArg {
                    tan_arg,
                    tan_sign,
                    shift,
                })
            } else if let Some(shift) =
                cas_ast::views::as_rational_const(ctx, *right, 8).filter(|value| !value.is_zero())
            {
                let (tan_arg, tan_sign) =
                    signed_unary_builtin_arg_for_calculus(ctx, *left, BuiltinFn::Tan)?;
                Some(ShiftedTangentLogArg {
                    tan_arg,
                    tan_sign,
                    shift,
                })
            } else {
                None
            }
        }
        Expr::Sub(left, right) => {
            if let Some(shift) =
                cas_ast::views::as_rational_const(ctx, *left, 8).filter(|value| !value.is_zero())
            {
                let tan_arg = unary_builtin_arg_for_calculus(ctx, *right, BuiltinFn::Tan)?;
                Some(ShiftedTangentLogArg {
                    tan_arg,
                    tan_sign: -BigRational::one(),
                    shift,
                })
            } else if let Some(shift) =
                cas_ast::views::as_rational_const(ctx, *right, 8).filter(|value| !value.is_zero())
            {
                let tan_arg = unary_builtin_arg_for_calculus(ctx, *left, BuiltinFn::Tan)?;
                Some(ShiftedTangentLogArg {
                    tan_arg,
                    tan_sign: BigRational::one(),
                    shift: -shift,
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

fn build_compact_shifted_tangent_log_arg(
    ctx: &mut Context,
    tan: ExprId,
    shift: BigRational,
    tan_sign: &BigRational,
) -> Option<ExprId> {
    if shift.is_zero() {
        return None;
    }

    let shift_expr = rational_const_for_calculus_presentation(ctx, shift.abs());
    if tan_sign == &BigRational::one() {
        if shift.is_positive() {
            Some(ctx.add(Expr::Add(tan, shift_expr)))
        } else {
            Some(ctx.add(Expr::Sub(tan, shift_expr)))
        }
    } else if tan_sign == &-BigRational::one() {
        if shift.is_positive() {
            Some(ctx.add(Expr::Sub(shift_expr, tan)))
        } else {
            let neg_tan = ctx.add(Expr::Neg(tan));
            Some(ctx.add(Expr::Sub(neg_tan, shift_expr)))
        }
    } else {
        None
    }
}

pub(super) fn ln_constant_shifted_tan_sqrt_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let Expr::Function(ln_fn, ln_args) = ctx.get(target).clone() else {
        return None;
    };
    if ctx.builtin_of(ln_fn) != Some(BuiltinFn::Ln) || ln_args.len() != 1 {
        return None;
    }

    let log_arg = ln_args[0];
    let shifted_tan = shifted_tangent_log_arg(ctx, log_arg)?;
    let radicand = calculus_sqrt_like_radicand(ctx, shifted_tan.tan_arg)?;
    let chain_coeff = sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
    let signed_chain_coeff = shifted_tan.tan_sign.clone() * chain_coeff;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&signed_chain_coeff)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let tan = ctx.call_builtin(BuiltinFn::Tan, vec![sqrt_radicand]);
    let compact_log_arg =
        build_compact_shifted_tangent_log_arg(ctx, tan, shifted_tan.shift, &shifted_tan.tan_sign)?;
    let cos = ctx.call_builtin(BuiltinFn::Cos, vec![sqrt_radicand]);
    let two = ctx.num(2);
    let cos_sq = ctx.add(Expr::Pow(cos, two));

    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let mut denominator_factors = Vec::new();
    if denominator_coeff != BigRational::one() {
        denominator_factors.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_factors.push(sqrt_radicand);
    denominator_factors.push(cos_sq);
    denominator_factors.push(compact_log_arg);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    let derivative = ctx.add(Expr::Div(numerator, denominator));

    let cos_guard = ctx.call_builtin(BuiltinFn::Cos, vec![sqrt_radicand]);
    Some((
        cas_ast::hold::wrap_hold(ctx, derivative),
        vec![
            crate::ImplicitCondition::Positive(radicand),
            crate::ImplicitCondition::Positive(compact_log_arg),
            crate::ImplicitCondition::NonZero(cos_guard),
        ],
    ))
}
