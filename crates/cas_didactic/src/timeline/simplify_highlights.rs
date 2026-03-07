use cas_ast::{Context, Expr, ExprId, ExprPath};
use cas_formatter::path::{
    diff_find_all_paths_to_expr, diff_find_path_to_expr, diff_find_paths_by_structure,
    extract_add_terms, find_path_to_expr, navigate_to_subexpr,
};
use cas_formatter::{
    DisplayContext, HighlightColor, HighlightConfig, LaTeXExprHighlighted,
    LaTeXExprHighlightedWithHints, PathHighlightConfig, PathHighlightedLatexRenderer,
    StylePreferences,
};
use cas_solver::{pathsteps_to_expr_path, Step};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TimelineStepSnapshots {
    pub global_before_expr: ExprId,
    pub global_after_expr: ExprId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TimelineRenderedStepMath {
    pub global_before: String,
    pub global_after: String,
    pub local_change_latex: String,
}

pub(super) fn resolve_timeline_step_global_snapshots(
    context: &mut Context,
    steps: &[Step],
    original_expr: ExprId,
    step_idx: usize,
    step: &Step,
) -> TimelineStepSnapshots {
    let global_before_expr = step.global_before.unwrap_or_else(|| {
        if step_idx == 0 {
            original_expr
        } else {
            steps
                .get(step_idx - 1)
                .and_then(|prev| prev.global_after)
                .unwrap_or(original_expr)
        }
    });
    let global_after_expr = step.global_after.unwrap_or_else(|| {
        cas_solver::reconstruct_global_expr(context, global_before_expr, step.path(), step.after)
    });

    TimelineStepSnapshots {
        global_before_expr,
        global_after_expr,
    }
}

pub(super) fn render_timeline_step_math(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> TimelineRenderedStepMath {
    let (global_before, global_after) =
        render_global_transition_latex(context, step, snapshots, display_hints, style_prefs);
    let local_change_latex = render_local_change_latex(context, step, display_hints, style_prefs);

    TimelineRenderedStepMath {
        global_before,
        global_after,
        local_change_latex,
    }
}

fn render_global_transition_latex(
    context: &Context,
    step: &Step,
    snapshots: TimelineStepSnapshots,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> (String, String) {
    if let Some(before_local) = step.before_local().filter(|&bl| bl != step.before) {
        let before_local_is_add = matches!(context.get(before_local), Expr::Add(_, _));
        let focus_path = if !before_local_is_add {
            find_path_to_expr(context, step.before, before_local)
        } else {
            Vec::new()
        };

        if !focus_path.is_empty() {
            let mut extended = pathsteps_to_expr_path(step.path());
            extended.extend(focus_path);

            let before = render_with_single_path(
                context,
                snapshots.global_before_expr,
                extended.clone(),
                HighlightColor::Red,
                display_hints,
                style_prefs,
            );
            let after = render_with_single_path(
                context,
                snapshots.global_after_expr,
                extended,
                HighlightColor::Green,
                display_hints,
                style_prefs,
            );

            return (before, after);
        }

        let focus_before = before_local;
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

        let mut found_paths: Vec<ExprPath> = Vec::new();
        for term in &focus_terms {
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

        let before = if !found_paths.is_empty() {
            let mut before_config = PathHighlightConfig::new();
            for path in found_paths {
                before_config.add(path, HighlightColor::Red);
            }
            render_with_paths(
                context,
                snapshots.global_before_expr,
                &before_config,
                display_hints,
                style_prefs,
            )
        } else {
            render_with_single_path(
                context,
                snapshots.global_before_expr,
                pathsteps_to_expr_path(step.path()),
                HighlightColor::Red,
                display_hints,
                style_prefs,
            )
        };

        let after = if let Some(after_path) =
            diff_find_path_to_expr(context, snapshots.global_after_expr, focus_after)
        {
            render_with_single_path(
                context,
                snapshots.global_after_expr,
                after_path,
                HighlightColor::Green,
                display_hints,
                style_prefs,
            )
        } else {
            let mut after_config = HighlightConfig::new();
            after_config.add(focus_after, HighlightColor::Green);
            LaTeXExprHighlighted {
                context,
                id: snapshots.global_after_expr,
                highlights: &after_config,
            }
            .to_latex()
        };

        return (before, after);
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

fn render_local_change_latex(
    context: &Context,
    step: &Step,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    let focus_before = step.before_local().unwrap_or(step.before);
    let focus_after = step.after_local().unwrap_or(step.after);

    let mut rule_before_config = HighlightConfig::new();
    rule_before_config.add(focus_before, HighlightColor::Red);
    let local_before_colored = LaTeXExprHighlightedWithHints {
        context,
        id: focus_before,
        highlights: &rule_before_config,
        hints: display_hints,
        style_prefs: Some(style_prefs),
    }
    .to_latex();

    let mut rule_after_config = HighlightConfig::new();
    rule_after_config.add(focus_after, HighlightColor::Green);
    let local_after_colored = LaTeXExprHighlightedWithHints {
        context,
        id: focus_after,
        highlights: &rule_after_config,
        hints: display_hints,
        style_prefs: Some(style_prefs),
    }
    .to_latex();

    format!(
        "{} \\rightarrow {}",
        local_before_colored, local_after_colored
    )
}

fn render_with_single_path(
    context: &Context,
    id: ExprId,
    path: ExprPath,
    color: HighlightColor,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    let mut config = PathHighlightConfig::new();
    config.add(path, color);
    render_with_paths(context, id, &config, display_hints, style_prefs)
}

fn render_with_paths(
    context: &Context,
    id: ExprId,
    path_highlights: &PathHighlightConfig,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    PathHighlightedLatexRenderer {
        context,
        id,
        path_highlights,
        hints: Some(display_hints),
        style_prefs: Some(style_prefs),
    }
    .to_latex()
}
