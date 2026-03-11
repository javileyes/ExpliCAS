mod final_result;
mod requires;

use crate::cas_solver::ImplicitCondition;
use cas_ast::{Context, ExprId};
use cas_formatter::{DisplayContext, StylePreferences};

pub(super) fn render_timeline_footer(
    context: &mut Context,
    html: &mut String,
    final_result_expr: ExprId,
    global_requires: &[ImplicitCondition],
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) {
    html.push_str(&final_result::render_final_result_footer_html(
        context,
        final_result_expr,
        display_hints,
        style_prefs,
    ));
    html.push_str(&requires::render_requires_footer_html(
        context,
        global_requires,
    ));
}
