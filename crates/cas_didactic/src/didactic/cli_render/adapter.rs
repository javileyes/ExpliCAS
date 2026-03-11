use super::StepDisplayMode;
use crate::runtime::Step;
use cas_ast::ExprId;

pub(super) fn format_cli_simplification_steps_with_simplifier(
    simplifier: &mut crate::runtime::Simplifier,
    expr: ExprId,
    steps: &[Step],
    style_signals: cas_formatter::root_style::ParseStyleSignals,
    display_mode: StepDisplayMode,
    format_cli_simplification_steps: fn(
        &mut cas_ast::Context,
        ExprId,
        &[Step],
        cas_formatter::root_style::ParseStyleSignals,
        StepDisplayMode,
    ) -> Vec<String>,
) -> Vec<String> {
    format_cli_simplification_steps(
        &mut simplifier.context,
        expr,
        steps,
        style_signals,
        display_mode,
    )
}
