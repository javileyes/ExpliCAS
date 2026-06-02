//! Final presentation cleanup for inverse-function arguments in integrals.
//!
//! This is display-facing only: it compacts affine arguments after an
//! antiderivative has already been computed and verified by the integration
//! pipeline.

use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::polynomial::Polynomial;
use num_traits::One;

pub(super) fn compact_integer_affine_inverse_args_for_integration_presentation(
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

#[cfg(test)]
mod tests {
    use cas_ast::{Context, ExprId};
    use cas_formatter::DisplayExpr;
    use cas_parser::parse;

    use super::compact_integer_affine_inverse_args_for_integration_presentation;

    fn rendered(ctx: &Context, id: ExprId) -> String {
        format!("{}", DisplayExpr { context: ctx, id })
    }

    #[test]
    fn inverse_argument_presentation_compacts_nested_affine_terms() {
        let mut ctx = Context::new();
        let expr = parse("arcsin(1+2*x) + acosh(1+x/2)", &mut ctx).unwrap();
        let compact =
            compact_integer_affine_inverse_args_for_integration_presentation(&mut ctx, expr, "x");

        assert_eq!(
            rendered(&ctx, compact),
            "arcsin(2 * x + 1) + acosh(1/2 * x + 1)"
        );
    }
}
