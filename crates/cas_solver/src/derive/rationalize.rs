use cas_ast::{BuiltinFn, Expr, ExprId};

pub(crate) fn looks_rationalizable_source(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    let Expr::Div(_, denominator) = ctx.get(expr) else {
        return false;
    };
    contains_root_like(ctx, *denominator)
}

fn contains_root_like(ctx: &cas_ast::Context, expr: ExprId) -> bool {
    match ctx.get(expr) {
        Expr::Pow(_, exp) => matches!(ctx.get(*exp), Expr::Number(n) if !n.is_integer()),
        Expr::Function(name, args)
            if (ctx.is_builtin(*name, BuiltinFn::Sqrt)
                || ctx.is_builtin(*name, BuiltinFn::Root))
                && !args.is_empty() =>
        {
            true
        }
        Expr::Add(left, right)
        | Expr::Sub(left, right)
        | Expr::Mul(left, right)
        | Expr::Div(left, right) => {
            contains_root_like(ctx, *left) || contains_root_like(ctx, *right)
        }
        Expr::Neg(inner) | Expr::Hold(inner) => contains_root_like(ctx, *inner),
        Expr::Function(_, args) => args.iter().any(|arg| contains_root_like(ctx, *arg)),
        Expr::Matrix { data, .. } => data.iter().any(|arg| contains_root_like(ctx, *arg)),
        Expr::Number(_) | Expr::Constant(_) | Expr::Variable(_) | Expr::SessionRef(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::looks_rationalizable_source;

    #[test]
    fn detects_root_denominator_as_rationalizable() {
        let mut ctx = cas_ast::Context::new();
        let expr = cas_parser::parse("1/(sqrt(x)-1)", &mut ctx).expect("parse");
        assert!(looks_rationalizable_source(&ctx, expr));
    }
}
