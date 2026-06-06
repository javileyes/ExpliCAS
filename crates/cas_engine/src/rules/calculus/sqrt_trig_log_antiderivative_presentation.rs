//! Post-calculus derivative presentation for sqrt-chain trig log antiderivatives.

use super::presentation_utils::calculus_sqrt_like_radicand;
use super::scalar_presentation::{
    negate_calculus_presentation, nonzero_rational_parts, scale_expr_for_calculus_presentation,
};
use super::shifted_sqrt_args::shifted_sqrt_arg_radicand_and_sign;
use super::sqrt_chain_factor_presentation::sqrt_chain_linear_derivative_coeff;
use super::sqrt_trig_log_integrand_presentation::build_compact_sqrt_trig_log_integrand;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::{One, Signed};

pub(super) fn sqrt_trig_log_antiderivative_derivative_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> Option<(ExprId, Vec<crate::ImplicitCondition>)> {
    let mut expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    let mut outer_sign = BigRational::one();
    while let Expr::Neg(inner) = ctx.get(expr).clone() {
        expr = inner;
        outer_sign = -outer_sign;
    }

    let Expr::Function(fn_id, args) = ctx.get(expr).clone() else {
        return None;
    };
    if ctx.builtin_of(fn_id) != Some(BuiltinFn::Ln) || args.len() != 1 {
        return None;
    }

    let Expr::Function(abs_fn, abs_args) = ctx.get(args[0]).clone() else {
        return None;
    };
    if ctx.builtin_of(abs_fn) != Some(BuiltinFn::Abs) || abs_args.len() != 1 {
        return None;
    }

    let Expr::Function(trig_fn, trig_args) = ctx.get(abs_args[0]).clone() else {
        return None;
    };
    if trig_args.len() != 1 {
        return None;
    }
    let (trig_builtin, derivative_builtin, sign) = match ctx.builtin_of(trig_fn)? {
        BuiltinFn::Cos => (BuiltinFn::Cos, BuiltinFn::Tan, -BigRational::one()),
        BuiltinFn::Sin => (BuiltinFn::Sin, BuiltinFn::Cot, BigRational::one()),
        _ => return None,
    };

    let direct_radicand = calculus_sqrt_like_radicand(ctx, trig_args[0]);
    let shifted_radicand_and_sign = shifted_sqrt_arg_radicand_and_sign(ctx, trig_args[0], var_name);
    let (radicand, arg_sign, direct_sqrt_arg) = if let Some(radicand) = direct_radicand {
        (radicand, BigRational::one(), true)
    } else {
        let (radicand, arg_sign) = shifted_radicand_and_sign?;
        (radicand, arg_sign, false)
    };
    let chain_coeff =
        outer_sign * sign * arg_sign * sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
    let compact = if direct_sqrt_arg {
        build_compact_sqrt_trig_log_integrand(ctx, derivative_builtin, radicand, chain_coeff)?
    } else {
        shifted_sqrt_trig_log_derivative_quotient(
            ctx,
            derivative_builtin,
            trig_args[0],
            radicand,
            chain_coeff,
        )?
    };
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let trig_display_arg = if direct_sqrt_arg {
        sqrt_radicand
    } else {
        trig_args[0]
    };
    let trig_display = ctx.call_builtin(trig_builtin, vec![trig_display_arg]);
    let conditions = vec![
        crate::ImplicitCondition::Positive(radicand),
        crate::ImplicitCondition::NonZero(trig_display),
    ];

    Some((compact, conditions))
}

fn shifted_sqrt_trig_log_derivative_quotient(
    ctx: &mut Context,
    derivative_builtin: BuiltinFn,
    trig_arg: ExprId,
    radicand: ExprId,
    chain_coeff: BigRational,
) -> Option<ExprId> {
    if !matches!(derivative_builtin, BuiltinFn::Tan | BuiltinFn::Cot) {
        return None;
    }

    let negative = chain_coeff.is_negative();
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&chain_coeff.abs())?;
    let trig = ctx.call_builtin(derivative_builtin, vec![trig_arg]);
    let numerator = scale_expr_for_calculus_presentation(ctx, numerator_coeff, trig);
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator = if denominator_coeff == BigRational::one() {
        sqrt_radicand
    } else {
        let denominator_coeff = ctx.add(Expr::Number(denominator_coeff));
        cas_math::expr_nary::build_balanced_mul(ctx, &[denominator_coeff, sqrt_radicand])
    };
    let quotient = ctx.add_raw(Expr::Div(numerator, denominator));
    let quotient = if negative {
        negate_calculus_presentation(ctx, quotient)
    } else {
        quotient
    };
    Some(cas_ast::hold::wrap_hold(ctx, quotient))
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
    fn sqrt_trig_log_antiderivative_derivative_presentation_compacts_shifted_chain() {
        let mut ctx = Context::new();
        let expr = parse("-ln(abs(cos(sqrt(3*x+1))))", &mut ctx).unwrap();
        let (compact, conditions) =
            sqrt_trig_log_antiderivative_derivative_presentation(&mut ctx, expr, "x").unwrap();
        let rendered_conditions: Vec<_> = conditions
            .iter()
            .map(|condition| match condition {
                crate::ImplicitCondition::Positive(expr) => {
                    format!("{} > 0", rendered(&ctx, *expr))
                }
                crate::ImplicitCondition::NonZero(expr) => {
                    format!("{} != 0", rendered(&ctx, *expr))
                }
                other => format!("{other:?}"),
            })
            .collect();

        assert_eq!(
            rendered(&ctx, compact),
            "3 * tan(sqrt(3 * x + 1)) / (2 * sqrt(3 * x + 1))"
        );
        assert_eq!(
            rendered_conditions,
            vec!["3 * x + 1 > 0", "cos(sqrt(3 * x + 1)) != 0"]
        );
    }

    #[test]
    fn sqrt_trig_log_antiderivative_derivative_presentation_accepts_shifted_sqrt_arg() {
        let mut ctx = Context::new();
        let expr = parse("ln(abs(cos(b - sqrt(x))))", &mut ctx).unwrap();
        let (compact, conditions) =
            sqrt_trig_log_antiderivative_derivative_presentation(&mut ctx, expr, "x").unwrap();
        let rendered_conditions: Vec<_> = conditions
            .iter()
            .map(|condition| match condition {
                crate::ImplicitCondition::Positive(expr) => {
                    format!("{} > 0", rendered(&ctx, *expr))
                }
                crate::ImplicitCondition::NonZero(expr) => {
                    format!("{} != 0", rendered(&ctx, *expr))
                }
                other => format!("{other:?}"),
            })
            .collect();

        assert_eq!(rendered(&ctx, compact), "tan(b - sqrt(x)) / (2 * sqrt(x))");
        assert_eq!(rendered_conditions, vec!["x > 0", "cos(b - sqrt(x)) != 0"]);
    }

    #[test]
    fn sqrt_trig_log_antiderivative_derivative_presentation_accepts_shifted_cot_arg() {
        let mut ctx = Context::new();
        let expr = parse("ln(abs(sin(b - sqrt(x))))", &mut ctx).unwrap();
        let (compact, conditions) =
            sqrt_trig_log_antiderivative_derivative_presentation(&mut ctx, expr, "x").unwrap();
        let rendered_conditions: Vec<_> = conditions
            .iter()
            .map(|condition| match condition {
                crate::ImplicitCondition::Positive(expr) => {
                    format!("{} > 0", rendered(&ctx, *expr))
                }
                crate::ImplicitCondition::NonZero(expr) => {
                    format!("{} != 0", rendered(&ctx, *expr))
                }
                other => format!("{other:?}"),
            })
            .collect();

        assert_eq!(rendered(&ctx, compact), "-cot(b - sqrt(x)) / (2 * sqrt(x))");
        assert_eq!(rendered_conditions, vec!["x > 0", "sin(b - sqrt(x)) != 0"]);
    }
}
