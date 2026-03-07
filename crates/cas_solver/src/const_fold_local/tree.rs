use cas_ast::{Context, Expr, ExprId};

pub(super) fn get_children(ctx: &Context, id: ExprId) -> Vec<ExprId> {
    match ctx.get(id) {
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) | Expr::Pow(a, b) => {
            vec![*a, *b]
        }
        Expr::Neg(inner) | Expr::Hold(inner) => vec![*inner],
        Expr::Function(_, args) => args.clone(),
        Expr::Matrix { data, .. } => data.clone(),
        _ => vec![],
    }
}
