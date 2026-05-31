//! Log derivative presentation for reciprocal trig sqrt-chain arguments.

use super::presentation_utils::calculus_sqrt_like_radicand;
use super::scalar_presentation::{
    nonzero_rational_parts, rational_const_for_calculus_presentation,
};
use super::sqrt_chain_factor_presentation::sqrt_chain_linear_derivative_coeff;
use super::unary_function_presentation::{
    same_sqrt_like_unary_pair_for_calculus, unordered_same_sqrt_like_unary_pair_for_calculus,
};
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use num_rational::BigRational;
use num_traits::One;

pub(super) fn ln_reciprocal_trig_sqrt_derivative_presentation(
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
    let (sqrt_arg, reciprocal_builtin, denominator_builtin) =
        reciprocal_trig_log_sqrt_parts(ctx, log_arg)?;
    let radicand = calculus_sqrt_like_radicand(ctx, sqrt_arg)?;
    let chain_coeff = sqrt_chain_linear_derivative_coeff(ctx, radicand, var_name)?;
    let (numerator_coeff, denominator_coeff) = nonzero_rational_parts(&chain_coeff)?;

    let sqrt_radicand = ctx.call_builtin(BuiltinFn::Sqrt, vec![radicand]);
    let denominator_trig = ctx.call_builtin(denominator_builtin, vec![sqrt_radicand]);
    let numerator = rational_const_for_calculus_presentation(ctx, numerator_coeff);
    let mut denominator_factors = Vec::new();
    if denominator_coeff != BigRational::one() {
        denominator_factors.push(rational_const_for_calculus_presentation(
            ctx,
            denominator_coeff,
        ));
    }
    denominator_factors.push(sqrt_radicand);
    denominator_factors.push(denominator_trig);
    let denominator = cas_math::expr_nary::build_balanced_mul(ctx, &denominator_factors);
    let derivative = ctx.add(Expr::Div(numerator, denominator));

    let reciprocal = ctx.call_builtin(reciprocal_builtin, vec![sqrt_radicand]);
    let companion_builtin = match reciprocal_builtin {
        BuiltinFn::Sec => BuiltinFn::Tan,
        BuiltinFn::Csc => BuiltinFn::Cot,
        _ => return None,
    };
    let companion = ctx.call_builtin(companion_builtin, vec![sqrt_radicand]);
    let compact_log_arg = match reciprocal_builtin {
        BuiltinFn::Sec => ctx.add(Expr::Add(companion, reciprocal)),
        BuiltinFn::Csc => ctx.add(Expr::Sub(reciprocal, companion)),
        _ => return None,
    };

    Some((
        cas_ast::hold::wrap_hold(ctx, derivative),
        vec![
            crate::ImplicitCondition::Positive(radicand),
            crate::ImplicitCondition::NonZero(denominator_trig),
            crate::ImplicitCondition::Positive(compact_log_arg),
        ],
    ))
}

fn reciprocal_trig_log_sqrt_parts(
    ctx: &mut Context,
    expr: ExprId,
) -> Option<(ExprId, BuiltinFn, BuiltinFn)> {
    match ctx.get(expr).clone() {
        Expr::Add(left, right) => unordered_same_sqrt_like_unary_pair_for_calculus(
            ctx,
            left,
            BuiltinFn::Sec,
            right,
            BuiltinFn::Tan,
        )
        .map(|arg| (arg, BuiltinFn::Sec, BuiltinFn::Cos)),
        Expr::Sub(left, right) => {
            same_sqrt_like_unary_pair_for_calculus(ctx, left, BuiltinFn::Csc, right, BuiltinFn::Cot)
                .map(|arg| (arg, BuiltinFn::Csc, BuiltinFn::Sin))
        }
        _ => None,
    }
}
