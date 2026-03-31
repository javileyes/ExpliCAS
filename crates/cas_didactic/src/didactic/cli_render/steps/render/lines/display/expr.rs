use cas_ast::{Context, ExprId};

pub(super) fn display_expr_styled(
    ctx: &mut Context,
    expr: ExprId,
    _prefs: &cas_formatter::root_style::StylePreferences,
) -> String {
    crate::didactic::latex_to_plain_text(
        &cas_formatter::LaTeXExpr {
            context: ctx,
            id: expr,
        }
        .to_latex(),
    )
}
