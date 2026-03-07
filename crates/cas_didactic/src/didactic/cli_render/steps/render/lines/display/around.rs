use cas_ast::{Context, ExprId};

pub(super) fn render_before_line(
    ctx: &mut Context,
    before_expr: ExprId,
    style_prefs: &cas_formatter::root_style::StylePreferences,
    render_expr: fn(&mut Context, ExprId, &cas_formatter::root_style::StylePreferences) -> String,
) -> String {
    format!(
        "   Before: {}",
        cas_formatter::clean_display_string(&render_expr(ctx, before_expr, style_prefs))
    )
}

pub(super) fn render_after_line(
    ctx: &mut Context,
    after_expr: ExprId,
    style_prefs: &cas_formatter::root_style::StylePreferences,
    render_expr: fn(&mut Context, ExprId, &cas_formatter::root_style::StylePreferences) -> String,
) -> String {
    format!(
        "   After: {}",
        cas_formatter::clean_display_string(&render_expr(ctx, after_expr, style_prefs))
    )
}
