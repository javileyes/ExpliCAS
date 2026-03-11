mod adapter;
mod empty;
mod entry;
mod prepare;
mod steps;

use super::display_policy::StepDisplayMode;
use crate::runtime::Step;
use cas_ast::{Context, ExprId};

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
    entry::format_cli_simplification_steps(
        ctx,
        expr,
        steps,
        style_signals,
        display_mode,
        empty::render_empty_simplification_lines,
        prepare::build_cli_style_preferences,
        steps::render_simplification_step_lines,
    )
}

/// Variant of [`format_cli_simplification_steps`] that accepts a simplifier.
///
/// Keeps REPL frontends from reaching into `simplifier.context` directly.
pub fn format_cli_simplification_steps_with_simplifier(
    simplifier: &mut crate::runtime::Simplifier,
    expr: ExprId,
    steps: &[Step],
    style_signals: cas_formatter::root_style::ParseStyleSignals,
    display_mode: StepDisplayMode,
) -> Vec<String> {
    adapter::format_cli_simplification_steps_with_simplifier(
        simplifier,
        expr,
        steps,
        style_signals,
        display_mode,
        format_cli_simplification_steps,
    )
}
