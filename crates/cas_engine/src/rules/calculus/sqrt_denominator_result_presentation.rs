//! Result-side sqrt-denominator detection for integration presentation.

use cas_ast::{BuiltinFn, Context, Expr, ExprId};
use cas_math::root_forms::extract_square_root_base;

pub(super) fn inverse_sqrt_quotient_arg_result(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if args.len() == 1
                && matches!(
                    ctx.builtin_of(*fn_id),
                    Some(
                        BuiltinFn::Arcsin | BuiltinFn::Asin | BuiltinFn::Arctan | BuiltinFn::Asinh
                    )
                ) =>
        {
            matches!(ctx.get(args[0]), Expr::Div(_, den) if extract_square_root_base(ctx, *den).is_some())
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right) => {
            inverse_sqrt_quotient_arg_result(ctx, *left)
                || inverse_sqrt_quotient_arg_result(ctx, *right)
        }
        Expr::Neg(inner) => inverse_sqrt_quotient_arg_result(ctx, *inner),
        _ => false,
    }
}

pub(super) fn has_sqrt_denominator_result(ctx: &Context, expr: ExprId) -> bool {
    let expr = cas_ast::hold::unwrap_internal_hold(ctx, expr);
    match ctx.get(expr) {
        Expr::Div(num, den) => {
            extract_square_root_base(ctx, *den).is_some()
                || has_sqrt_denominator_result(ctx, *num)
                || has_sqrt_denominator_result(ctx, *den)
        }
        Expr::Add(left, right) | Expr::Sub(left, right) | Expr::Mul(left, right) => {
            has_sqrt_denominator_result(ctx, *left) || has_sqrt_denominator_result(ctx, *right)
        }
        Expr::Neg(inner) => has_sqrt_denominator_result(ctx, *inner),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use cas_ast::Context;
    use cas_parser::parse;

    use super::inverse_sqrt_quotient_arg_result;

    #[test]
    fn inverse_sqrt_quotient_arg_result_detects_compact_inverse_sqrt_substitution() {
        let mut ctx = Context::new();
        let expr = parse("arcsin(x^2/sqrt(3))", &mut ctx).unwrap();

        assert!(inverse_sqrt_quotient_arg_result(&ctx, expr));

        let rationalized = parse("arcsin(1/3 * sqrt(3) * x^2)", &mut ctx).unwrap();

        assert!(!inverse_sqrt_quotient_arg_result(&ctx, rationalized));

        let arctan = parse("arctan((2*x+2)/sqrt(6))/sqrt(6)", &mut ctx).unwrap();

        assert!(inverse_sqrt_quotient_arg_result(&ctx, arctan));
    }
}
