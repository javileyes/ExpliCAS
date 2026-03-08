use super::StepDisplayMode;
use cas_ast::ExprId;
use cas_solver::Step;

pub(super) fn format_cli_simplification_steps_with_simplifier(
    simplifier: &mut cas_solver::Simplifier,
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
