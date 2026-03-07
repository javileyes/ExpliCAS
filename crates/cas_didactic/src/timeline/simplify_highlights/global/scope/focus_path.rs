use cas_ast::{Context, Expr, ExprId, ExprPath};
use cas_formatter::path::find_path_to_expr;

pub(super) fn resolve_focus_path(
    context: &Context,
    before_expr: ExprId,
    before_local: ExprId,
) -> ExprPath {
    if matches!(context.get(before_local), Expr::Add(_, _)) {
        Vec::new()
    } else {
        find_path_to_expr(context, before_expr, before_local)
    }
}
