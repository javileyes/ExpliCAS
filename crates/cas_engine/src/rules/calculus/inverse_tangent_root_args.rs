use cas_ast::{BuiltinFn, Context, Expr, ExprId};

use super::presentation_utils::is_half_power_exponent;

pub(super) fn arctan_sqrt_radicand_arg(ctx: &Context, target: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Atan | BuiltinFn::Arctan)
        )
    {
        return None;
    }

    let radicand = match ctx.get(args[0]) {
        Expr::Function(sqrt_fn, sqrt_args)
            if sqrt_args.len() == 1 && ctx.is_builtin(*sqrt_fn, BuiltinFn::Sqrt) =>
        {
            sqrt_args[0]
        }
        Expr::Pow(base, exp) if is_half_power_exponent(ctx, *exp) => *base,
        _ => return None,
    };

    Some(radicand)
}

pub(super) fn arccot_sqrt_radicand_arg(ctx: &Context, target: ExprId) -> Option<ExprId> {
    let Expr::Function(fn_id, args) = ctx.get(target) else {
        return None;
    };
    if args.len() != 1
        || !matches!(
            ctx.builtin_of(*fn_id),
            Some(BuiltinFn::Acot | BuiltinFn::Arccot)
        )
    {
        return None;
    }

    let radicand = match ctx.get(args[0]) {
        Expr::Function(sqrt_fn, sqrt_args)
            if sqrt_args.len() == 1 && ctx.is_builtin(*sqrt_fn, BuiltinFn::Sqrt) =>
        {
            sqrt_args[0]
        }
        Expr::Pow(base, exp) if is_half_power_exponent(ctx, *exp) => *base,
        _ => return None,
    };

    Some(radicand)
}
