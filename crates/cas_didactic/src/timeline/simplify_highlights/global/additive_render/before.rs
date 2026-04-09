mod config;
mod fallback;
mod with_paths;

use crate::runtime::Step;
use cas_ast::{Context, Expr, ExprId, ExprPath};
use cas_formatter::path::{
    diff_find_path_to_expr, diff_find_paths_by_structure, navigate_to_subexpr,
};
use cas_formatter::{
    DisplayContext, HighlightColor, HighlightConfig, LaTeXExprHighlightedWithHints,
    PathHighlightConfig, StylePreferences,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn render_before_additive_focus(
    context: &Context,
    global_before_expr: cas_ast::ExprId,
    focus_before: ExprId,
    found_paths: &[ExprPath],
    step: &Step,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
    render_with_paths: fn(
        &Context,
        cas_ast::ExprId,
        &PathHighlightConfig,
        &DisplayContext,
        &StylePreferences,
    ) -> String,
    render_with_single_path: fn(
        &Context,
        cas_ast::ExprId,
        ExprPath,
        HighlightColor,
        &DisplayContext,
        &StylePreferences,
    ) -> String,
) -> String {
    let scope_path =
        diff_find_path_to_expr(context, global_before_expr, focus_before).or_else(|| {
            diff_find_paths_by_structure(context, global_before_expr, focus_before)
                .into_iter()
                .next()
        });
    if let Some(scope_path) = scope_path {
        return render_with_single_path(
            context,
            global_before_expr,
            scope_path,
            HighlightColor::Red,
            display_hints,
            style_prefs,
        );
    }

    if !found_paths.is_empty() {
        let normalized_paths =
            normalize_additive_term_paths(context, global_before_expr, found_paths);
        let path_rendered = with_paths::render_before_additive_focus_with_paths(
            context,
            global_before_expr,
            &normalized_paths,
            display_hints,
            style_prefs,
            config::build_before_additive_focus_config,
            render_with_paths,
        );
        if count_color_markers(&path_rendered, HighlightColor::Red) >= normalized_paths.len() {
            return path_rendered;
        }

        return render_before_additive_focus_with_expr_ids(
            context,
            global_before_expr,
            &normalized_paths,
            display_hints,
            style_prefs,
        );
    }

    fallback::render_before_additive_focus_fallback(
        context,
        global_before_expr,
        focus_before,
        step,
        display_hints,
        style_prefs,
        HighlightColor::Red,
        render_with_single_path,
    )
}

fn render_before_additive_focus_with_expr_ids(
    context: &Context,
    global_before_expr: ExprId,
    paths: &[ExprPath],
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    let mut config = HighlightConfig::new();
    for path in paths {
        let expr = navigate_to_subexpr(context, global_before_expr, path);
        config.add(expr, HighlightColor::Red);
    }
    LaTeXExprHighlightedWithHints {
        context,
        id: global_before_expr,
        highlights: &config,
        hints: display_hints,
        style_prefs: Some(style_prefs),
    }
    .to_latex()
}

fn count_color_markers(latex: &str, color: HighlightColor) -> usize {
    latex
        .match_indices(&format!("\\color{{{}}}", color.to_latex()))
        .count()
}

fn normalize_additive_term_paths(
    context: &Context,
    global_before_expr: ExprId,
    found_paths: &[ExprPath],
) -> Vec<ExprPath> {
    let mut normalized = Vec::with_capacity(found_paths.len());
    for path in found_paths {
        let promoted = promote_to_additive_term_boundary(context, global_before_expr, path);
        if !normalized.contains(&promoted) {
            normalized.push(promoted);
        }
    }
    normalized
}

fn promote_to_additive_term_boundary(context: &Context, root: ExprId, path: &ExprPath) -> ExprPath {
    let mut current = path.clone();

    while !current.is_empty() {
        let parent = current[..current.len() - 1].to_vec();
        let parent_expr = navigate_to_subexpr(context, root, &parent);

        match context.get(parent_expr) {
            Expr::Add(_, _) | Expr::Sub(_, _) => return current,
            Expr::Neg(_) => {
                let grandparent = if parent.is_empty() {
                    None
                } else {
                    Some(parent[..parent.len() - 1].to_vec())
                };
                if let Some(grandparent_path) = grandparent {
                    let grandparent_expr = navigate_to_subexpr(context, root, &grandparent_path);
                    if matches!(
                        context.get(grandparent_expr),
                        Expr::Add(_, _) | Expr::Sub(_, _)
                    ) {
                        return parent;
                    }
                }
                current = parent;
            }
            _ => current = parent,
        }
    }

    path.clone()
}
