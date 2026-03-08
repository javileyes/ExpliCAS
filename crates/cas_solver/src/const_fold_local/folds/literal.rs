use cas_ast::{Context, Expr, ExprId};

pub(crate) fn is_constant_literal(ctx: &Context, id: ExprId) -> bool {
    match ctx.get(id) {
        Expr::Number(_) => true,
        Expr::Constant(c) => matches!(
            c,
            cas_ast::Constant::Pi
                | cas_ast::Constant::E
                | cas_ast::Constant::I
                | cas_ast::Constant::Infinity
                | cas_ast::Constant::Undefined
        ),
        _ => false,
    }
}

pub(crate) fn fold_neg(ctx: &mut Context, inner: ExprId) -> Option<ExprId> {
    match ctx.get(inner) {
        Expr::Number(n) => Some(ctx.add(Expr::Number(-n.clone()))),
        _ => None,
    }
}
