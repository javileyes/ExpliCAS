mod focus_path;

use super::super::renderers::render_with_single_path;
use super::additive::render_additive_focus_transition;
use super::direct::render_direct_focus_transition;
use crate::runtime::Step;
use crate::timeline::simplify_highlights::TimelineStepSnapshots;
use cas_ast::{Context, Expr, ExprId, ExprPath};
use cas_formatter::path::{
    diff_find_path_to_expr, diff_find_paths_by_structure, navigate_to_subexpr,
};
use cas_formatter::{DisplayContext, StylePreferences};
use num_traits::{One, Zero};

pub(super) fn render_local_scope_transition(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    before_local: ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> (String, String) {
    if !matches!(context.get(before_local), Expr::Add(_, _) | Expr::Sub(_, _)) {
        if let Some(absolute_transition) = render_absolute_scope_transition(
            context,
            step,
            snapshots,
            before_local,
            display_hints,
            style_prefs,
        ) {
            return absolute_transition;
        }
    }

    let focus_path = focus_path::resolve_focus_path(context, step.before, before_local);

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

fn render_absolute_scope_transition(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    before_local: ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> Option<(String, String)> {
    let focus_after = step.after_local().unwrap_or(step.after);
    let mut before_path = find_absolute_path(context, snapshots.global_before_expr, before_local);
    let mut after_path = find_absolute_path(context, snapshots.global_after_expr, focus_after);

    if before_path.is_none() {
        if let Some(candidate) = after_path.clone().filter(|path| !path.is_empty()) {
            let candidate_expr =
                navigate_to_subexpr(context, snapshots.global_before_expr, &candidate);
            if candidate_expr != snapshots.global_before_expr {
                before_path = Some(candidate);
            }
        }
    }

    if after_path.is_none()
        && !matches!(context.get(focus_after), Expr::Number(n) if n.is_zero() || n.is_one())
    {
        if let Some(candidate) = before_path.clone().filter(|path| !path.is_empty()) {
            let candidate_expr =
                navigate_to_subexpr(context, snapshots.global_after_expr, &candidate);
            if candidate_expr != snapshots.global_after_expr {
                after_path = Some(candidate);
            }
        }
    }

    let before_path = before_path?;
    let after_path = after_path?;
    let before_path = narrow_before_path_to_changed_additive_child(
        context,
        before_local,
        &before_path,
        &after_path,
    )
    .unwrap_or(before_path);

    let before = render_with_single_path(
        context,
        snapshots.global_before_expr,
        before_path,
        cas_formatter::HighlightColor::Red,
        display_hints,
        style_prefs,
    );
    let after = render_with_single_path(
        context,
        snapshots.global_after_expr,
        after_path,
        cas_formatter::HighlightColor::Green,
        display_hints,
        style_prefs,
    );
    Some((before, after))
}

fn find_absolute_path(context: &Context, root: ExprId, target: ExprId) -> Option<ExprPath> {
    diff_find_path_to_expr(context, root, target).or_else(|| {
        diff_find_paths_by_structure(context, root, target)
            .into_iter()
            .next()
    })
}

fn narrow_before_path_to_changed_additive_child(
    context: &Context,
    before_local: ExprId,
    before_path: &ExprPath,
    after_path: &ExprPath,
) -> Option<ExprPath> {
    let Expr::Div(numerator, denominator) = context.get(before_local) else {
        return None;
    };

    [(0_u8, *numerator), (1_u8, *denominator)]
        .into_iter()
        .find_map(|(child_idx, child)| {
            if !matches!(context.get(child), Expr::Add(_, _) | Expr::Sub(_, _)) {
                return None;
            }

            let mut child_path = before_path.clone();
            child_path.push(child_idx);
            after_path.starts_with(&child_path).then_some(child_path)
        })
}
