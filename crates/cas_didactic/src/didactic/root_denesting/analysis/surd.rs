use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;
use num_rational::BigRational;

pub(super) fn analyze_surd(ctx: &Context, expr: ExprId) -> Option<(BigRational, ExprId)> {
    match ctx.get(expr) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            Some((BigRational::from_integer(BigInt::from(1)), args[0]))
        }
        Expr::Mul(left, right) => analyze_scaled_surd(ctx, *left, *right),
        _ => None,
    }
}

fn analyze_scaled_surd(
    ctx: &Context,
    left: ExprId,
    right: ExprId,
) -> Option<(BigRational, ExprId)> {
    match (ctx.get(left), ctx.get(right)) {
        (Expr::Number(coeff), Expr::Function(fn_id, args))
            if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            Some((coeff.clone(), args[0]))
        }
        (Expr::Function(fn_id, args), Expr::Number(coeff))
            if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            Some((coeff.clone(), args[0]))
        }
        _ => None,
    }
}
