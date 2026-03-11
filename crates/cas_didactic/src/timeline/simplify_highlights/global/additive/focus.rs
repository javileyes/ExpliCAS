use super::super::additive_search::collect_additive_focus_paths;
use crate::cas_solver::{pathsteps_to_expr_path, Step};
use cas_ast::{Context, ExprId, ExprPath};
use cas_formatter::path::{diff_find_path_to_expr, navigate_to_subexpr};

pub(super) fn collect_additive_focus_paths_with_scope(
    context: &Context,
    step: &Step,
    global_before_expr: ExprId,
    focus_before: ExprId,
    focus_terms: &[ExprId],
) -> Vec<ExprPath> {
    let step_path_prefix = pathsteps_to_expr_path(step.path());
    let subexpr_at_path = navigate_to_subexpr(context, global_before_expr, &step_path_prefix);
    let before_local_path = diff_find_path_to_expr(context, subexpr_at_path, focus_before);

    let (search_scope, scope_path_prefix) = if let Some(path_to_local) = &before_local_path {
        let local_scope = navigate_to_subexpr(context, subexpr_at_path, path_to_local);
        let mut full_prefix = step_path_prefix.clone();
        full_prefix.extend(path_to_local.clone());
        (local_scope, full_prefix)
    } else {
        (subexpr_at_path, step_path_prefix.clone())
    };

    collect_additive_focus_paths(context, search_scope, &scope_path_prefix, focus_terms)
}
