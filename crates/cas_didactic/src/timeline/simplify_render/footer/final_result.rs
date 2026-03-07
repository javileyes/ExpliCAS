use super::super::super::simplify_summary::render_timeline_final_result_html;
use cas_ast::{Context, ExprId};
use cas_formatter::{DisplayContext, StylePreferences};

pub(super) fn render_final_result_footer_html(
    context: &Context,
    final_result_expr: ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    render_timeline_final_result_html(context, final_result_expr, display_hints, style_prefs)
}
