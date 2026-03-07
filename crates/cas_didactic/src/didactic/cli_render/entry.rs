use super::StepDisplayMode;
use cas_ast::{Context, ExprId};
use cas_solver::Step;

pub(super) fn format_cli_simplification_steps(
    ctx: &mut Context,
    expr: ExprId,
    steps: &[Step],
    style_signals: cas_formatter::root_style::ParseStyleSignals,
    display_mode: StepDisplayMode,
    render_empty_simplification_lines: fn(&mut Context, ExprId, StepDisplayMode) -> Vec<String>,
    build_cli_style_preferences: fn(
        &mut Context,
        ExprId,
        cas_formatter::root_style::ParseStyleSignals,
    ) -> cas_formatter::root_style::StylePreferences,
    render_simplification_step_lines: fn(
        &mut Context,
        ExprId,
        &[Step],
        &cas_formatter::root_style::StylePreferences,
        StepDisplayMode,
    ) -> Vec<String>,
) -> Vec<String> {
    if display_mode == StepDisplayMode::None {
        return Vec::new();
    }

    if steps.is_empty() {
        return render_empty_simplification_lines(ctx, expr, display_mode);
    }

    let style_prefs = build_cli_style_preferences(ctx, expr, style_signals);
    render_simplification_step_lines(ctx, expr, steps, &style_prefs, display_mode)
}
