use super::super::super::simplify_summary::render_timeline_global_requires_html;
use cas_ast::Context;
use cas_solver::ImplicitCondition;

pub(super) fn render_requires_footer_html(
    context: &mut Context,
    global_requires: &[ImplicitCondition],
) -> String {
    render_timeline_global_requires_html(context, global_requires)
}
