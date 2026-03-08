use cas_ast::{Context, Expr, ExprId};

pub(super) fn rewrite_children(
    ctx: &mut Context,
    expr: ExprId,
    expr_data: Expr,
    recur: fn(&mut Context, ExprId) -> ExprId,
) -> ExprId {
    match expr_data {
        Expr::Add(l, r) => {
            let new_l = recur(ctx, l);
            let new_r = recur(ctx, r);
            ctx.add(Expr::Add(new_l, new_r))
        }
        Expr::Sub(l, r) => {
            let new_l = recur(ctx, l);
            let new_r = recur(ctx, r);
            ctx.add(Expr::Sub(new_l, new_r))
        }
        Expr::Mul(l, r) => {
            let new_l = recur(ctx, l);
            let new_r = recur(ctx, r);
            ctx.add(Expr::Mul(new_l, new_r))
        }
        Expr::Div(l, r) => {
            let new_l = recur(ctx, l);
            let new_r = recur(ctx, r);
            ctx.add(Expr::Div(new_l, new_r))
        }
        Expr::Pow(base, exp) => {
            let new_base = recur(ctx, base);
            let new_exp = recur(ctx, exp);
            ctx.add(Expr::Pow(new_base, new_exp))
        }
        Expr::Neg(inner) => {
            let new_inner = recur(ctx, inner);
            ctx.add(Expr::Neg(new_inner))
        }
        Expr::Function(name, args) => {
            let new_args: Vec<_> = args.iter().map(|&arg| recur(ctx, arg)).collect();
            ctx.add(Expr::Function(name, new_args))
        }
        _ => expr,
    }
}
