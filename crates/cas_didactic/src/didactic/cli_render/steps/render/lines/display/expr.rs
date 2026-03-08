use cas_ast::{Context, ExprId};
use cas_formatter::DisplayExprStyled;

pub(super) fn display_expr_styled(
    ctx: &mut Context,
    expr: ExprId,
    prefs: &cas_formatter::root_style::StylePreferences,
) -> String {
    format!("{}", DisplayExprStyled::new(ctx, expr, prefs))
}
