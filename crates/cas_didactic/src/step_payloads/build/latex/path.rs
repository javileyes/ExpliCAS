use crate::runtime::Step;
use cas_ast::Context;

pub(super) fn render_step_before_latex(context: &Context, step: &Step) -> String {
    let before_expr = step.global_before.unwrap_or(step.before);
    render_step_side_latex(
        context,
        step,
        before_expr,
        cas_formatter::HighlightColor::Red,
    )
}

pub(super) fn render_step_after_latex(context: &Context, step: &Step) -> String {
    let after_expr = step.global_after.unwrap_or(step.after);
    render_step_side_latex(
        context,
        step,
        after_expr,
        cas_formatter::HighlightColor::Green,
    )
}

fn render_step_side_latex(
    context: &Context,
    step: &Step,
    expr: cas_ast::ExprId,
    color: cas_formatter::HighlightColor,
) -> String {
    let expr_path = crate::runtime::pathsteps_to_expr_path(step.path());
    let highlighted =
        crate::step_payload_render::render_step_path_latex(context, expr, expr_path, color);
    let plain = cas_formatter::LaTeXExpr { context, id: expr }.to_latex();

    if is_structurally_safe_highlight(&plain, &highlighted) {
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
