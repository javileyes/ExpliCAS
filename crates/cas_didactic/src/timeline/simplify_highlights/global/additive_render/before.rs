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
    if let Some(scope_path) =
        find_additive_scope_path_by_display(context, global_before_expr, focus_before)
    {
        return render_with_single_path(
            context,
            global_before_expr,
            scope_path,
            HighlightColor::Red,
            display_hints,
            style_prefs,
        );
    }

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

fn find_additive_scope_path_by_display(
    context: &Context,
    root: ExprId,
    target: ExprId,
) -> Option<ExprPath> {
    if !matches!(context.get(target), Expr::Add(_, _) | Expr::Sub(_, _)) {
        return None;
    }

    let target_display = render_display_key(context, target);
    let mut path = Vec::new();
    find_additive_scope_path_by_display_rec(context, root, &target_display, &mut path)
}

fn find_additive_scope_path_by_display_rec(
    context: &Context,
    current: ExprId,
    target_display: &str,
    path: &mut ExprPath,
) -> Option<ExprPath> {
    if matches!(context.get(current), Expr::Add(_, _) | Expr::Sub(_, _))
        && render_display_key(context, current) == target_display
    {
        return Some(path.clone());
    }

    match context.get(current) {
        Expr::Add(l, r) | Expr::Sub(l, r) | Expr::Mul(l, r) | Expr::Div(l, r) | Expr::Pow(l, r) => {
            path.push(0);
            if let Some(found) =
                find_additive_scope_path_by_display_rec(context, *l, target_display, path)
            {
                return Some(found);
            }
            path.pop();

            path.push(1);
            if let Some(found) =
                find_additive_scope_path_by_display_rec(context, *r, target_display, path)
            {
                return Some(found);
            }
            path.pop();
            None
        }
        Expr::Neg(inner) | Expr::Hold(inner) => {
            path.push(0);
            let found =
                find_additive_scope_path_by_display_rec(context, *inner, target_display, path);
            path.pop();
            found
        }
        Expr::Function(_, args) => {
            for (i, arg) in args.iter().enumerate() {
                path.push(i as u8);
                if let Some(found) =
                    find_additive_scope_path_by_display_rec(context, *arg, target_display, path)
                {
                    return Some(found);
                }
                path.pop();
            }
            None
        }
        Expr::Matrix { data, .. } => {
            for (i, elem) in data.iter().enumerate() {
                path.push(i as u8);
                if let Some(found) =
                    find_additive_scope_path_by_display_rec(context, *elem, target_display, path)
                {
                    return Some(found);
                }
                path.pop();
            }
            None
        }
        Expr::Number(_) | Expr::Variable(_) | Expr::Constant(_) | Expr::SessionRef(_) => None,
    }
}

fn render_display_key(context: &Context, expr: ExprId) -> String {
    cas_formatter::clean_display_string(&crate::didactic::latex_to_plain_text(
        &cas_formatter::LaTeXExpr { context, id: expr }.to_latex(),
    ))
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
