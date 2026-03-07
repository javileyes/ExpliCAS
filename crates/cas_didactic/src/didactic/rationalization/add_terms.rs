use cas_ast::{Context, Expr, ExprId};

pub(super) fn collect_add_terms(ctx: &Context, expr: ExprId) -> Vec<ExprId> {
    let mut terms = Vec::new();
    collect_add_terms_recursive(ctx, expr, &mut terms);
    terms
}

fn collect_add_terms_recursive(ctx: &Context, expr: ExprId, terms: &mut Vec<ExprId>) {
    match ctx.get(expr) {
        Expr::Add(left, right) => {
            collect_add_terms_recursive(ctx, *left, terms);
            collect_add_terms_recursive(ctx, *right, terms);
        }
        _ => terms.push(expr),
    }
}
