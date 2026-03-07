use super::{render_timeline_final_result_html, render_timeline_global_requires_html};
use cas_ast::{Context, ExprId};
use cas_formatter::{DisplayContext, StylePreferences};
use cas_solver::ImplicitCondition;

pub(super) fn render_timeline_footer(
    context: &mut Context,
    html: &mut String,
    final_result_expr: ExprId,
    global_requires: &[ImplicitCondition],
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) {
    html.push_str(&render_timeline_final_result_html(
        context,
        final_result_expr,
        display_hints,
        style_prefs,
    ));
    html.push_str(&render_timeline_global_requires_html(
        context,
        global_requires,
    ));
}
