use cas_ast::Context;
use cas_solver::Step;

pub(super) fn render_step_before_latex(context: &Context, step: &Step) -> String {
    let before_expr = step.global_before.unwrap_or(step.before);
    let expr_path = cas_solver::pathsteps_to_expr_path(step.path());
    crate::step_payload_render::render_step_path_latex(
        context,
        before_expr,
        expr_path,
        cas_formatter::HighlightColor::Red,
    )
}

pub(super) fn render_step_after_latex(context: &Context, step: &Step) -> String {
    let after_expr = step.global_after.unwrap_or(step.after);
    let expr_path = cas_solver::pathsteps_to_expr_path(step.path());
    crate::step_payload_render::render_step_path_latex(
        context,
        after_expr,
        expr_path,
        cas_formatter::HighlightColor::Green,
    )
}
