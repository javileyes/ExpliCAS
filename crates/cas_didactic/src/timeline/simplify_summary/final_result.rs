mod polynomial;
mod standard;

use cas_ast::{Context, ExprId};
use cas_formatter::{DisplayContext, StylePreferences};

pub(super) fn render_timeline_final_result_html(
    context: &Context,
    final_result_expr: ExprId,
    display_hints: &DisplayContext,
    style_prefs: &StylePreferences,
) -> String {
    if let Some(poly_text) =
        cas_math::poly_store::try_render_poly_result(context, final_result_expr)
    {
        return polynomial::render_polynomial_final_result_html(&poly_text);
    }

    standard::render_standard_final_result_html(
        context,
        final_result_expr,
        display_hints,
        style_prefs,
    )
}
