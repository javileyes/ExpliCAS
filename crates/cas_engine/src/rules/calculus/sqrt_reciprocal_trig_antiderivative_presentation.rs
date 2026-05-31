//! Result-side sqrt-chain reciprocal-trig antiderivative detection.

use super::polynomial_support::polynomial_radicand_for_calculus_presentation;
use super::presentation_utils::rational_const_for_hold;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;

pub(super) fn sqrt_reciprocal_trig_antiderivative_result(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(ctx.builtin_of(fn_id), Some(BuiltinFn::Sec | BuiltinFn::Csc)) =>
        {
            let Some(radicand) = extract_square_root_base(ctx, args[0]) else {
                return false;
            };
            polynomial_radicand_for_calculus_presentation(ctx, radicand, var_name).is_some_and(
                |poly| {
                    let derivative = poly.derivative();
                    !derivative.is_zero() && derivative.degree() == 0
                },
            )
        }
        Expr::Neg(inner) => sqrt_reciprocal_trig_antiderivative_result(ctx, inner, var_name),
        Expr::Mul(_, _) => {
            let factors = cas_math::expr_nary::mul_leaves(ctx, expr);
            let mut non_numeric = Vec::new();
            for factor in factors {
                if rational_const_for_hold(ctx, factor).is_none() {
                    non_numeric.push(factor);
                }
            }
            non_numeric.len() == 1
                && sqrt_reciprocal_trig_antiderivative_result(ctx, non_numeric[0], var_name)
        }
        _ => false,
    }
}
