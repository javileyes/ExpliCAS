use super::derivative_integrand_factor_parts::split_mul_div_factor_parts;
use super::presentation_utils::{calculus_sqrt_like_radicand, same_sqrt_like_argument};
use super::scalar_presentation::{
    negate_calculus_presentation, nonzero_rational_parts, rational_const_for_calculus_presentation,
    scale_expr_for_calculus_presentation,
};
use super::signed_factor_presentation::signed_mul_leaves_for_calculus_presentation;
use super::sqrt_chain_factor_presentation::{
    sqrt_chain_factor_coeff_over_sqrt, sqrt_chain_linear_derivative_coeff,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

pub(super) fn compact_direct_sqrt_hyperbolic_log_derivative_integrand(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (numerator_factors, denominator_factors) = match ctx.get(target).clone() {
        Expr::Neg(inner) => {
            let compact =
                compact_direct_sqrt_hyperbolic_log_derivative_integrand(ctx, inner, var_name)?;
            return Some(negate_calculus_presentation(ctx, compact));
        }
        Expr::Div(numerator, denominator) => (
            signed_mul_leaves_for_calculus_presentation(ctx, numerator),
            signed_mul_leaves_for_calculus_presentation(ctx, denominator),
        ),
        Expr::Mul(_, _) => split_mul_div_factor_parts(ctx, target)?,
        _ => return None,
    };

    if let Some((num_index, den_index, tanh_in_numerator, arg)) =
        hyperbolic_sinh_cosh_quotient_parts(ctx, &numerator_factors, &denominator_factors)
    {
        let radicand = calculus_sqrt_like_radicand(ctx, arg)?;
        let chain_coeff = sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
        let remaining_numerator: Vec<_> = numerator_factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != num_index).then_some(*factor))
            .collect();
        let remaining_denominator: Vec<_> = denominator_factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != den_index).then_some(*factor))
            .collect();
        let observed_coeff = sqrt_chain_factor_coeff_over_sqrt(
            ctx,
            &remaining_numerator,
            &remaining_denominator,
            radicand,
            var_name,
        )?;
        if observed_coeff == chain_coeff || observed_coeff == -chain_coeff {
            return build_compact_sqrt_hyperbolic_log_integrand(
                ctx,
                tanh_in_numerator,
                radicand,
                observed_coeff,
            );
        }
    }

    if let Some((tanh_index, arg)) =
        numerator_factors
            .iter()
            .enumerate()
            .find_map(|(idx, factor)| match ctx.get(*factor) {
                Expr::Function(fn_id, args)
                    if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Tanh) =>
                {
                    Some((idx, args[0]))
                }
                _ => None,
            })
    {
        let radicand = calculus_sqrt_like_radicand(ctx, arg)?;
        let chain_coeff = sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
        let remaining_numerator: Vec<_> = numerator_factors
            .iter()
            .enumerate()
            .filter_map(|(idx, factor)| (idx != tanh_index).then_some(*factor))
            .collect();
        let observed_coeff = sqrt_chain_factor_coeff_over_sqrt(
            ctx,
            &remaining_numerator,
            &denominator_factors,
            radicand,
            var_name,
        )?;
        if observed_coeff == chain_coeff || observed_coeff == -chain_coeff {
            return build_compact_sqrt_hyperbolic_log_integrand(
                ctx,
                true,
                radicand,
                observed_coeff,
            );
        }
    }

    let (tanh_index, arg) = denominator_factors
        .iter()
        .enumerate()
        .find_map(|(idx, factor)| match ctx.get(*factor) {
            Expr::Function(fn_id, args)
                if args.len() == 1 && ctx.builtin_of(*fn_id) == Some(BuiltinFn::Tanh) =>
            {
                Some((idx, args[0]))
            }
            _ => None,
        })?;
    let radicand = calculus_sqrt_like_radicand(ctx, arg)?;
    let chain_coeff = sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
    let remaining_denominator: Vec<_> = denominator_factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != tanh_index).then_some(*factor))
        .collect();
    let observed_coeff = sqrt_chain_factor_coeff_over_sqrt(
        ctx,
        &numerator_factors,
        &remaining_denominator,
        radicand,
        var_name,
    )?;
    if observed_coeff != chain_coeff && observed_coeff != -chain_coeff {
        return None;
    }

    build_compact_sqrt_hyperbolic_log_integrand(ctx, false, radicand, observed_coeff)
}

fn hyperbolic_sinh_cosh_quotient_parts(
    ctx: &mut Context,
    numerator_factors: &[ExprId],
    denominator_factors: &[ExprId],
) -> Option<(usize, usize, bool, ExprId)> {
    for (num_index, numerator_factor) in numerator_factors.iter().enumerate() {
        let Expr::Function(num_fn, num_args) = ctx.get(*numerator_factor).clone() else {
            continue;
        };
        if num_args.len() != 1 {
            continue;
        }
        let Some(num_builtin) = ctx.builtin_of(num_fn) else {
            continue;
        };
        let expected_den_builtin = match num_builtin {
            BuiltinFn::Sinh => BuiltinFn::Cosh,
            BuiltinFn::Cosh => BuiltinFn::Sinh,
            _ => continue,
        };
        for (den_index, denominator_factor) in denominator_factors.iter().enumerate() {
            let Expr::Function(den_fn, den_args) = ctx.get(*denominator_factor).clone() else {
                continue;
            };
            if den_args.len() != 1 || ctx.builtin_of(den_fn) != Some(expected_den_builtin) {
                continue;
            }
            if !same_sqrt_like_argument(ctx, num_args[0], den_args[0]) {
                continue;
            }
            return Some((
                num_index,
                den_index,
                num_builtin == BuiltinFn::Sinh,
                num_args[0],
            ));
        }
    }
    None
}

pub(super) fn sqrt_cosh_log_derivative_presentation(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let Expr::Function(ln_fn, ln_args) = ctx.get(target).clone() else {
        return None;
    };
    if ctx.builtin_of(ln_fn) != Some(BuiltinFn::Ln) || ln_args.len() != 1 {
        return None;
    }
    let Expr::Function(cosh_fn, cosh_args) = ctx.get(ln_args[0]).clone() else {
        return None;
    };
    if ctx.builtin_of(cosh_fn) != Some(BuiltinFn::Cosh) || cosh_args.len() != 1 {
        return None;
    }

    let radicand = calculus_sqrt_like_radicand(ctx, cosh_args[0])?;
    let chain_coeff = sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
    build_compact_sqrt_hyperbolic_log_integrand(ctx, true, radicand, chain_coeff)
}

fn build_compact_sqrt_hyperbolic_log_integrand(
    ctx: &mut Context,
    tanh_in_numerator: bool,
    radicand: ExprId,
    chain_coeff: BigRational,
) -> Option<ExprId> {
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let tanh = ctx.call_builtin(BuiltinFn::Tanh, vec![sqrt_radicand]);
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&chain_coeff)?;

    if tanh_in_numerator {
        let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, tanh);
        let denominator = if denominator_coeff == BigRational::one() {
            sqrt_radicand
        } else {
            let denominator_coeff =
                rational_const_for_calculus_presentation(ctx, denominator_coeff);
            cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, sqrt_radicand])
        };
        return Some(ctx.add(Expr::Div(numerator, denominator)));
    }

    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let denominator_core = cas_math::expr_nary::build_balanced_mul(ctx, &[tanh, sqrt_radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        denominator_core
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, denominator_core])
    };
    Some(ctx.add(Expr::Div(numerator, denominator)))
}
