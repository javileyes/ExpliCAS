mod additive;
mod additive_render;
mod additive_search;
mod default_transition;
mod direct;
mod scope;

use self::scope::render_local_scope_transition;
use super::TimelineStepSnapshots;
use crate::runtime::Step;
use cas_ast::{Context, Expr, ExprId};
use cas_formatter::{DisplayContext, StylePreferences};

pub(super) fn render_global_transition_latex(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> (String, String) {
    let local_scope = preferred_local_scope(context, step);

    if let Some(before_local) = local_scope {
        return render_local_scope_transition(
            context,
            step,
            snapshots,
            before_local,
            display_hints,
            style_prefs,
        );
    }

    default_transition::render_default_global_transition(
        context,
        step,
        snapshots,
        display_hints,
        style_prefs,
    )
}

fn preferred_local_scope(context: &Context, step: &Step) -> Option<ExprId> {
    let focus_before = step.before_local().unwrap_or(step.before);
    if step.before_local().is_some() {
        return Some(focus_before);
    }

    match context.get(focus_before) {
        Expr::Function(_, _)
        | Expr::Add(_, _)
        | Expr::Sub(_, _)
        | Expr::Div(_, _)
        | Expr::Pow(_, _) => Some(focus_before),
        _ => None,
    }
}
