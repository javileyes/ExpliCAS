use cas_ast::{Context, Expr, ExprId};
use num_bigint::BigInt;

pub(super) fn get_sqrt_inner(ctx: &Context, id: ExprId) -> Option<ExprId> {
    match ctx.get(id) {
        Expr::Function(fn_id, args)
            if ctx.is_builtin(*fn_id, cas_ast::BuiltinFn::Sqrt) && args.len() == 1 =>
        {
            Some(args[0])
        }
        Expr::Pow(base, exp) => {
            let Expr::Number(n) = ctx.get(*exp) else {
                return None;
            };
            if *n.numer() == BigInt::from(1) && *n.denom() == BigInt::from(2) {
                Some(*base)
            } else {
                None
            }
        }
        _ => None,
    }
}
