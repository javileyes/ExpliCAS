use super::{render_additive_focus_transition, render_direct_focus_transition};
use crate::timeline::simplify_highlights::TimelineStepSnapshots;
use cas_ast::{Context, Expr, ExprId, ExprPath};
use cas_formatter::path::find_path_to_expr;
use cas_formatter::{DisplayContext, StylePreferences};
use cas_solver::Step;

pub(super) fn render_local_scope_transition(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    before_local: ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> (String, String) {
    let focus_path = resolve_focus_path(context, step.before, before_local);

    if !focus_path.is_empty() {
        return render_direct_focus_transition(
            context,
            step,
            snapshots,
            focus_path,
            display_hints,
            style_prefs,
        );
    }

    render_additive_focus_transition(
        context,
        step,
        snapshots,
        before_local,
        display_hints,
        style_prefs,
    )
}

fn resolve_focus_path(context: &Context, before_expr: ExprId, before_local: ExprId) -> ExprPath {
    if matches!(context.get(before_local), Expr::Add(_, _)) {
        Vec::new()
    } else {
        find_path_to_expr(context, before_expr, before_local)
    }
}
