use cas_ast::{Context, ExprPath};
use cas_formatter::path::{diff_find_all_paths_to_expr, diff_find_paths_by_structure};

pub(super) fn collect_additive_focus_paths(
    context: &Context,
    search_scope: cas_ast::ExprId,
    scope_path_prefix: &ExprPath,
    focus_terms: &[cas_ast::ExprId],
) -> Vec<ExprPath> {
    let mut found_paths: Vec<ExprPath> = Vec::new();

    for term in focus_terms {
        let paths_before = found_paths.len();

        for sub_path in diff_find_all_paths_to_expr(context, search_scope, *term) {
            let mut full_path = scope_path_prefix.clone();
            full_path.extend(sub_path.clone());
            if !found_paths.contains(&full_path) {
                found_paths.push(full_path);
            }
        }

        if found_paths.len() == paths_before {
            for sub_path in diff_find_paths_by_structure(context, search_scope, *term) {
                let mut full_path = scope_path_prefix.clone();
                full_path.extend(sub_path.clone());
                if !found_paths.contains(&full_path) {
                    found_paths.push(full_path);
                }
            }
        }
    }

    found_paths
}
