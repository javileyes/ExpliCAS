use cas_ast::{Context, ExprId};

pub(super) fn display_discrete_solution_set(
    context: &Context,
    exprs: &[ExprId],
    display_expr: fn(&Context, ExprId) -> String,
) -> String {
    let rendered: Vec<String> = exprs
        .iter()
        .map(|expr| display_expr(context, *expr))
        .collect();
    format!("{{ {} }}", rendered.join(", "))
}
