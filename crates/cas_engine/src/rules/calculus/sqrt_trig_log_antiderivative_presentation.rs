//! Post-calculus derivative presentation for sqrt-chain trig log antiderivatives.

use super::presentation_utils::calculus_sqrt_like_radicand;
use super::sqrt_chain_factor_presentation::sqrt_chain_linear_derivative_coeff;
use super::sqrt_trig_log_integrand_presentation::build_compact_sqrt_trig_log_integrand;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

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

    let radicand = calculus_sqrt_like_radicand(ctx, trig_args[0])?;
    let chain_coeff =
        outer_sign * sign * sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
    let compact =
        build_compact_sqrt_trig_log_integrand(ctx, derivative_builtin, radicand, chain_coeff)?;
    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let trig_display = ctx.call_builtin(trig_builtin, vec![sqrt_radicand]);
    let conditions = vec![
        crate::ImplicitCondition::Positive(radicand),
        crate::ImplicitCondition::NonZero(trig_display),
    ];

    Some((compact, conditions))
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
}
