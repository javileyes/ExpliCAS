mod normalize;
mod render;

use cas_ast::Context;
use cas_solver::ImplicitCondition;

pub(super) fn render_timeline_global_requires_html(
    context: &mut Context,
    global_requires: &[ImplicitCondition],
) -> String {
    let requires_messages = normalize::normalize_global_requires(context, global_requires);
    render::render_global_requires_html(&requires_messages)
}
