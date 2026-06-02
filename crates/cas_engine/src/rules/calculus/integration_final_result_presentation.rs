//! Final integration-result presentation cleanup.
//!
//! Keep this after route-specific preservation: it only rewrites public result
//! display for already-computed antiderivatives.

use super::rationalized_sqrt_result_presentation::compact_acosh_surd_width_arg_for_integration_presentation;
use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_traits::One;

pub(super) fn apply_integration_final_presentation(
    ctx: &mut Context,
    mut result: ExprId,
    var_name: &str,
) -> ExprId {
    if let Some(compact) = compact_acosh_surd_width_arg_for_integration_presentation(ctx, result) {
        result = compact;
    }
    compact_integer_affine_inverse_args_for_integration_presentation(ctx, result, var_name)
}

fn compact_integer_affine_inverse_args_for_integration_presentation(
    ctx: &mut Context,
    expr: ExprId,
    var_name: &str,
) -> ExprId {
    match ctx.get(expr).clone() {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(fn_id),
                    Some(
                        BuiltinFn::Arcsin
                            | BuiltinFn::Asin
                            | BuiltinFn::Arccos
                            | BuiltinFn::Acos
                            | BuiltinFn::Acosh
                    )
                ) =>
        {
            let arg = args[0];
            let Some(builtin) = ctx.builtin_of(fn_id) else {
                return expr;
            };
            let compact_arg = Polynomial::from_expr(ctx, arg, var_name)
                .ok()
                .filter(|poly| {
                    poly.degree() == 1
                        && (poly.coeffs.iter().all(|c| c.is_integer())
                            || (builtin == BuiltinFn::Acosh
                                && poly
                                    .coeffs
                                    .first()
                                    .is_some_and(|constant| constant.is_one())))
                })
                .map(|poly| poly.to_expr(ctx))
                .unwrap_or(arg);
            ctx.add(Expr::Function(fn_id, vec![compact_arg]))
        }
        Expr::Neg(inner) => {
            let inner = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, inner, var_name,
            );
            ctx.add(Expr::Neg(inner))
        }
        Expr::Add(left, right) => {
            let left = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Add(left, right))
        }
        Expr::Sub(left, right) => {
            let left = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Sub(left, right))
        }
        Expr::Mul(left, right) => {
            let left = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Mul(left, right))
        }
        Expr::Div(left, right) => {
            let left = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, left, var_name,
            );
            let right = compact_integer_affine_inverse_args_for_integration_presentation(
                ctx, right, var_name,
            );
            ctx.add(Expr::Div(left, right))
        }
        _ => expr,
    }
}
