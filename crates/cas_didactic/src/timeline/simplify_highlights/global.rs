mod additive;
mod additive_render;
mod additive_search;
mod direct;

use self::additive::render_additive_focus_transition;
use self::additive_render::{render_after_additive_focus, render_before_additive_focus};
use self::additive_search::collect_additive_focus_paths;
use self::direct::render_direct_focus_transition;
use super::renderers::render_with_single_path;
use super::TimelineStepSnapshots;
use cas_ast::{Context, Expr};
use cas_formatter::path::find_path_to_expr;
use cas_formatter::{DisplayContext, HighlightColor, StylePreferences};
use cas_solver::{pathsteps_to_expr_path, Step};

pub(super) fn render_global_transition_latex(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> (String, String) {
    if let Some(before_local) = step.before_local().filter(|&bl| bl != step.before) {
        return render_local_scope_transition(
            context,
            step,
            snapshots,
            before_local,
            display_hints,
            style_prefs,
        );
    }

    let expr_path = pathsteps_to_expr_path(step.path());
    let before = render_with_single_path(
        context,
        snapshots.global_before_expr,
        expr_path.clone(),
        HighlightColor::Red,
        display_hints,
        style_prefs,
    );
    let after = render_with_single_path(
        context,
        snapshots.global_after_expr,
        expr_path,
        HighlightColor::Green,
        display_hints,
        style_prefs,
    );
    (before, after)
}

fn render_local_scope_transition(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    before_local: cas_ast::ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> (String, String) {
    let before_local_is_add = matches!(context.get(before_local), Expr::Add(_, _));
    let focus_path = if !before_local_is_add {
        find_path_to_expr(context, step.before, before_local)
    } else {
        Vec::new()
    };

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
