//! Compact integrand presentation for sqrt-chain trig-log families.

use super::derivative_integrand_factor_parts::split_mul_div_factor_parts;
use super::presentation_utils::{calculus_sqrt_like_radicand, same_sqrt_like_argument};
use super::scalar_presentation::{
    negate_calculus_presentation, nonzero_rational_parts, rational_const_for_calculus_presentation,
    scale_expr_for_calculus_presentation,
};
use super::sqrt_chain_factor_presentation::{
    sqrt_chain_factor_coeff_over_sqrt, sqrt_chain_linear_derivative_coeff,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;
use num_rational::BigRational;
use num_traits::One;

pub(super) fn compact_direct_sqrt_trig_log_derivative_integrand(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (numerator_factors, denominator_factors) = match ctx.get(target).clone() {
        Expr::Neg(inner) => {
            let compact = compact_direct_sqrt_trig_log_derivative_integrand(ctx, inner, var_name)?;
            return Some(negate_calculus_presentation(ctx, compact));
        }
        Expr::Div(numerator, denominator) => (
            cas_math::expr_nary::mul_leaves(ctx, numerator)
                .into_iter()
                .collect(),
            cas_math::expr_nary::mul_leaves(ctx, denominator)
                .into_iter()
                .collect(),
        ),
        Expr::Mul(_, _) => split_mul_div_factor_parts(ctx, target).unwrap_or_else(|| {
            (
                cas_math::expr_nary::mul_leaves(ctx, target)
                    .into_iter()
                    .collect(),
                Vec::new(),
            )
        }),
        _ => return None,
    };
    let (trig_index, trig_builtin, arg) =
        numerator_factors
            .iter()
            .enumerate()
            .find_map(|(idx, factor)| match ctx.get(*factor) {
                Expr::Function(fn_id, args)
                    if args.len() == 1
                        && matches!(
                            ctx.builtin_of(*fn_id),
                            Some(BuiltinFn::Tan | BuiltinFn::Cot)
                        ) =>
                {
                    Some((idx, ctx.builtin_of(*fn_id)?, args[0]))
                }
                _ => None,
            })?;
    let radicand = calculus_sqrt_like_radicand(ctx, arg)?;
    let chain_coeff = sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
    let remaining_numerator: Vec<_> = numerator_factors
        .iter()
        .enumerate()
        .filter_map(|(idx, factor)| (idx != trig_index).then_some(*factor))
        .collect();
    let observed_coeff = sqrt_chain_factor_coeff_over_sqrt(
        ctx,
        &remaining_numerator,
        &denominator_factors,
        radicand,
        var_name,
    )?;
    if observed_coeff != chain_coeff && observed_coeff != -chain_coeff {
        return None;
    }

    build_compact_sqrt_trig_log_integrand(ctx, trig_builtin, radicand, observed_coeff)
}

pub(super) fn compact_sqrt_trig_log_derivative_integrand(
    ctx: &mut Context,
    target: ExprId,
    var_name: &str,
) -> Option<ExprId> {
    let (numerator_factors, denominator_factors) = match ctx.get(target).clone() {
        Expr::Neg(inner) => {
            let compact = compact_sqrt_trig_log_derivative_integrand(ctx, inner, var_name)?;
            return Some(negate_calculus_presentation(ctx, compact));
        }
        Expr::Div(numerator, denominator) => (
            cas_math::expr_nary::mul_leaves(ctx, numerator)
                .into_iter()
                .collect(),
            cas_math::expr_nary::mul_leaves(ctx, denominator)
                .into_iter()
                .collect(),
        ),
        Expr::Mul(_, _) => split_mul_div_factor_parts(ctx, target)?,
        _ => return None,
    };
    let (den_index, den_builtin, arg) =
        denominator_factors
            .iter()
            .enumerate()
            .find_map(|(idx, factor)| match ctx.get(*factor) {
                Expr::Function(fn_id, args)
                    if args.len() == 1
                        && matches!(
                            ctx.builtin_of(*fn_id),
                            Some(BuiltinFn::Cos | BuiltinFn::Sin)
                        ) =>
                {
                    Some((idx, ctx.builtin_of(*fn_id)?, args[0]))
                }
                _ => None,
            })?;
    let numerator_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Sin,
        BuiltinFn::Sin => BuiltinFn::Cos,
        _ => return None,
    };
    let trig_builtin = match den_builtin {
        BuiltinFn::Cos => BuiltinFn::Tan,
        BuiltinFn::Sin => BuiltinFn::Cot,
        _ => return None,
    };
    let (num_index, _) = numerator_factors.iter().enumerate().find(|(_, factor)| {
        let Expr::Function(fn_id, args) = ctx.get(**factor).clone() else {
            return false;
        };
        args.len() == 1
            && ctx.builtin_of(fn_id) == Some(numerator_builtin)
            && same_sqrt_like_argument(ctx, args[0], arg)
    })?;
    let radicand = extract_square_root_base(ctx, arg)?;
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
    if observed_coeff != chain_coeff && observed_coeff != -chain_coeff {
        return None;
    }

    build_compact_sqrt_trig_log_integrand(ctx, trig_builtin, radicand, observed_coeff)
}

pub(super) fn build_compact_sqrt_trig_log_integrand(
    ctx: &mut Context,
    trig_builtin: BuiltinFn,
    radicand: ExprId,
    chain_coeff: BigRational,
) -> Option<ExprId> {
    if !matches!(trig_builtin, BuiltinFn::Tan | BuiltinFn::Cot) {
        return None;
    }
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let trig = ctx.call_builtin(trig_builtin, vec![sqrt_radicand]);
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&chain_coeff)?;
    let compact_numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, trig);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_coeff = rational_const_for_calculus_presentation(ctx, denominator_coeff);
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, sqrt_radicand])
    };
    Some(ctx.add(Expr::Div(compact_numerator, denominator)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn compact_direct_sqrt_trig_log_derivative_integrand_preserves_tangent_form() {
        let mut ctx = Context::new();
        let expr = parse("tan(sqrt(x))/(2*sqrt(x))", &mut ctx).unwrap();
        let compact =
            compact_direct_sqrt_trig_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "tan(sqrt(x)) / (2 * sqrt(x))");
    }

    #[test]
    fn compact_direct_sqrt_trig_log_derivative_integrand_accepts_mul_div_chain() {
        let mut ctx = Context::new();
        let expr = parse("tan(sqrt(3*x+1))*3/(2*sqrt(3*x+1))", &mut ctx).unwrap();
        let compact =
            compact_direct_sqrt_trig_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }

    #[test]
    fn compact_sqrt_trig_log_derivative_integrand_preserves_negative_tangent_form() {
        let mut ctx = Context::new();
        let inner = parse("sin(sqrt(x))*x^(-1/2)/(2*cos(sqrt(x)))", &mut ctx).unwrap();
        let expr = ctx.add(Expr::Neg(inner));
        let compact = compact_sqrt_trig_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(rendered(&ctx, compact), "-tan(sqrt(x)) / (2 * sqrt(x))");
    }

    #[test]
    fn compact_sqrt_trig_log_derivative_integrand_accepts_half_power_argument() {
        let mut ctx = Context::new();
        let expr = parse(
            "((3*x+1)^(-1/2) * sin((3*x+1)^(1/2)) * 3)/(2 * cos((3*x+1)^(1/2)))",
            &mut ctx,
        )
        .unwrap();
        let compact = compact_sqrt_trig_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }

    #[test]
    fn compact_sqrt_trig_log_derivative_integrand_accepts_scaled_radicand_denominator() {
        let mut ctx = Context::new();
        let expr = parse(
            "((3*x+1)^(1/2) * sin((3*x+1)^(1/2)) * 3)/(cos((3*x+1)^(1/2)) * (6*x+2))",
            &mut ctx,
        )
        .unwrap();
        let compact = compact_sqrt_trig_log_derivative_integrand(&mut ctx, expr, "x").unwrap();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
    }
}
