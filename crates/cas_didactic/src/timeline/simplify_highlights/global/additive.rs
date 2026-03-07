use super::super::TimelineStepSnapshots;
use super::{
    collect_additive_focus_paths, render_after_additive_focus, render_before_additive_focus,
};
use cas_ast::Context;
use cas_formatter::path::{diff_find_path_to_expr, extract_add_terms, navigate_to_subexpr};
use cas_formatter::{DisplayContext, StylePreferences};
use cas_solver::{pathsteps_to_expr_path, Step};

pub(super) fn render_additive_focus_transition(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    focus_before: cas_ast::ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> (String, String) {
    let focus_after = step.after_local().unwrap_or(step.after);
    let focus_terms = extract_add_terms(context, focus_before);
    let step_path_prefix = pathsteps_to_expr_path(step.path());
    let subexpr_at_path =
        navigate_to_subexpr(context, snapshots.global_before_expr, &step_path_prefix);
    let before_local_path = diff_find_path_to_expr(context, subexpr_at_path, focus_before);

    let (search_scope, scope_path_prefix) = if let Some(path_to_local) = &before_local_path {
        let local_scope = navigate_to_subexpr(context, subexpr_at_path, path_to_local);
        let mut full_prefix = step_path_prefix.clone();
        full_prefix.extend(path_to_local.clone());
        (local_scope, full_prefix)
    } else {
        (subexpr_at_path, step_path_prefix.clone())
    };

    let found_paths =
        collect_additive_focus_paths(context, search_scope, &scope_path_prefix, &focus_terms);
    let before = render_before_additive_focus(
        context,
        snapshots.global_before_expr,
        &found_paths,
        step,
        display_hints,
        style_prefs,
    );
    let after = render_after_additive_focus(
        context,
        snapshots.global_after_expr,
        focus_after,
        display_hints,
        style_prefs,
    );

    (before, after)
}
