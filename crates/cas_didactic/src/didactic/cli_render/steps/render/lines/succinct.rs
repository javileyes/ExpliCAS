use cas_ast::{Context, ExprId};
use cas_formatter::DisplayExprStyled;

pub(crate) fn render_succinct_step_line(
    ctx: &mut Context,
    current_root: ExprId,
    style_prefs: &cas_formatter::root_style::StylePreferences,
) -> String {
    format!(
        "-> {}",
        DisplayExprStyled::new(ctx, current_root, style_prefs)
    )
}
