use cas_ast::{Context, ExprId};

pub(super) fn build_cli_style_preferences(
    ctx: &mut Context,
    expr: ExprId,
    style_signals: cas_formatter::root_style::ParseStyleSignals,
) -> cas_formatter::root_style::StylePreferences {
    cas_formatter::root_style::StylePreferences::from_expression_with_signals(
        ctx,
        expr,
        Some(&style_signals),
    )
}
