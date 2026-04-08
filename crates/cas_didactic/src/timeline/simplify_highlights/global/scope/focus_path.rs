use cas_ast::{Context, Expr, ExprId, ExprPath};
use cas_formatter::path::{diff_find_path_to_expr, diff_find_paths_by_structure};

pub(super) fn resolve_focus_path(
    context: &Context,
    before_expr: ExprId,
    before_local: ExprId,
) -> ExprPath {
    if matches!(context.get(before_local), Expr::Add(_, _)) {
        Vec::new()
    } else {
        diff_find_path_to_expr(context, before_expr, before_local)
            .or_else(|| {
                diff_find_paths_by_structure(context, before_expr, before_local)
                    .into_iter()
                    .next()
            })
            .unwrap_or_default()
    }
}
