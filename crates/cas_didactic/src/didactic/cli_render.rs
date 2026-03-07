mod empty;
mod prepare;
mod steps;

use super::display_policy::StepDisplayMode;
use cas_ast::{Context, ExprId};
use cas_solver::Step;

/// Format simplification steps for CLI/REPL text output.
///
/// This keeps didactic rendering rules outside frontends so clients can remain
/// thin and only handle I/O.
pub fn format_cli_simplification_steps(
    ctx: &mut Context,
    expr: ExprId,
    steps: &[Step],
    style_signals: cas_formatter::root_style::ParseStyleSignals,
    display_mode: StepDisplayMode,
) -> Vec<String> {
    if display_mode == StepDisplayMode::None {
        return Vec::new();
    }

    if steps.is_empty() {
        return empty::render_empty_simplification_lines(ctx, expr, display_mode);
    }

    let style_prefs = prepare::build_cli_style_preferences(ctx, expr, style_signals);
    steps::render_simplification_step_lines(ctx, expr, steps, &style_prefs, display_mode)
}

/// Variant of [`format_cli_simplification_steps`] that accepts a simplifier.
///
/// Keeps REPL frontends from reaching into `simplifier.context` directly.
pub fn format_cli_simplification_steps_with_simplifier(
    simplifier: &mut cas_solver::Simplifier,
    expr: ExprId,
    steps: &[Step],
    style_signals: cas_formatter::root_style::ParseStyleSignals,
    display_mode: StepDisplayMode,
) -> Vec<String> {
    format_cli_simplification_steps(
        &mut simplifier.context,
        expr,
        steps,
        style_signals,
        display_mode,
    )
}
