mod fallback;
mod with_path;

use cas_ast::{Context, Expr, ExprPath};
use cas_formatter::path::{
    diff_find_path_to_expr, diff_find_paths_by_structure, navigate_to_subexpr,
};
use cas_formatter::{DisplayContext, HighlightColor, LaTeXExpr, StylePreferences};
use num_traits::Zero;

pub(super) fn render_after_additive_focus(
    context: &Context,
    global_after_expr: cas_ast::ExprId,
    focus_after: cas_ast::ExprId,
    found_paths: &[ExprPath],
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
    render_with_single_path: fn(
        &Context,
        cas_ast::ExprId,
        cas_ast::ExprPath,
        HighlightColor,
        &DisplayContext,
        &StylePreferences,
    ) -> String,
) -> String {
    let direct_path =
        diff_find_path_to_expr(context, global_after_expr, focus_after).or_else(|| {
            diff_find_paths_by_structure(context, global_after_expr, focus_after)
                .into_iter()
                .next()
        });
    let common_path = common_surviving_additive_scope_path(context, global_after_expr, found_paths);
    if let Some(after_path) = direct_path {
        return with_path::render_after_additive_focus_with_path(
            context,
            global_after_expr,
            after_path,
            display_hints,
            style_prefs,
            HighlightColor::Green,
            render_with_single_path,
        );
    }

    if matches!(context.get(focus_after), Expr::Number(n) if n.is_zero()) {
        return LaTeXExpr {
            context,
            id: global_after_expr,
        }
        .to_latex();
    }

    if let Some(scope_path) = common_path {
        return with_path::render_after_additive_focus_with_path(
            context,
            global_after_expr,
            scope_path,
            display_hints,
            style_prefs,
            HighlightColor::Green,
            render_with_single_path,
        );
    }

    if found_paths.len() > 1 {
        return with_path::render_after_additive_focus_with_path(
            context,
            global_after_expr,
            Vec::new(),
            display_hints,
            style_prefs,
            HighlightColor::Green,
            render_with_single_path,
        );
    }

    fallback::render_after_additive_focus_fallback(
        context,
        global_after_expr,
        focus_after,
        HighlightColor::Green,
    )
}

fn common_surviving_additive_scope_path(
    context: &Context,
    global_after_expr: cas_ast::ExprId,
    found_paths: &[ExprPath],
) -> Option<ExprPath> {
    if found_paths.len() < 2 {
        return None;
    }

    let mut prefix = found_paths[0].clone();
    for path in &found_paths[1..] {
        let common_len = prefix
            .iter()
            .zip(path.iter())
            .take_while(|(left, right)| left == right)
            .count();
        prefix.truncate(common_len);
    }

    if prefix.is_empty() {
        return None;
    }

    let scope_expr = navigate_to_subexpr(context, global_after_expr, &prefix);
    if scope_expr == global_after_expr {
        return None;
    }

    matches!(context.get(scope_expr), Expr::Add(_, _) | Expr::Sub(_, _)).then_some(prefix)
}
