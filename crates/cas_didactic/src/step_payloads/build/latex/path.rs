use crate::runtime::Step;
use cas_ast::Context;
use cas_formatter::path::{diff_find_path_to_expr, diff_find_paths_by_structure};

pub(super) fn render_step_before_latex(context: &Context, step: &Step) -> String {
    let before_expr = step.global_before.unwrap_or(step.before);
    let before_focus = step.before_local().unwrap_or(step.before);
    render_step_side_latex(
        context,
        before_expr,
        before_focus,
        step.path(),
        cas_formatter::HighlightColor::Red,
    )
}

pub(super) fn render_step_after_latex(context: &Context, step: &Step) -> String {
    let after_expr = step.global_after.unwrap_or(step.after);
    let after_focus = step.after_local().unwrap_or(step.after);
    render_step_side_latex(
        context,
        after_expr,
        after_focus,
        step.path(),
        cas_formatter::HighlightColor::Green,
    )
}

fn render_step_side_latex(
    context: &Context,
    expr: cas_ast::ExprId,
    focus: cas_ast::ExprId,
    fallback_path: &[cas_solver_core::step_types::PathStep],
    color: cas_formatter::HighlightColor,
) -> String {
    let resolved_focus_path = diff_find_path_to_expr(context, expr, focus)
        .or_else(|| diff_find_paths_by_structure(context, expr, focus).into_iter().next());
    let expr_path = resolved_focus_path
        .clone()
        .unwrap_or_else(|| crate::runtime::pathsteps_to_expr_path(fallback_path));
    let highlighted =
        crate::step_payload_render::render_step_path_latex(context, expr, expr_path, color);
    let plain = cas_formatter::LaTeXExpr { context, id: expr }.to_latex();

    if resolved_focus_path.is_some() || is_structurally_safe_highlight(&plain, &highlighted) {
        highlighted
    } else {
        plain
    }
}

fn is_structurally_safe_highlight(plain_latex: &str, highlighted_latex: &str) -> bool {
    cas_formatter::clean_display_string(&crate::didactic::latex_to_plain_text(plain_latex))
        == cas_formatter::clean_display_string(&crate::didactic::latex_to_plain_text(
            highlighted_latex,
        ))
}
