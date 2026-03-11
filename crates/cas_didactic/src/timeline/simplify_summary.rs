mod final_result;
mod requires;

use crate::cas_solver::ImplicitCondition;
use cas_ast::{Context, ExprId};
use cas_formatter::{DisplayContext, StylePreferences};

pub(super) fn render_timeline_final_result_html(
    context: &Context,
    final_result_expr: ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    final_result::render_timeline_final_result_html(
        context,
        final_result_expr,
        display_hints,
        style_prefs,
    )
}

pub(super) fn render_timeline_global_requires_html(
    context: &mut Context,
    global_requires: &[ImplicitCondition],
) -> String {
    requires::render_timeline_global_requires_html(context, global_requires)
}
