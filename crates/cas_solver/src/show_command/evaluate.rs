use super::ShowCommandContext;
use crate::{
    format_history_eval_metadata_sections, format_inspect_history_entry_error_message,
    format_show_history_command_lines_with_context, Engine, HistoryEntryInspection,
    InspectHistoryEntryInputError,
};

/// Evaluate `show #id` command and return formatted lines for CLI rendering.
///
/// The caller provides `inspect` to bridge stateful session lookup from outer layers.
pub fn evaluate_show_command_lines_with<F>(
    engine: &mut Engine,
    line: &str,
    inspect: F,
) -> Result<Vec<String>, String>
where
    F: FnOnce(&mut Engine, &str) -> Result<HistoryEntryInspection, InspectHistoryEntryInputError>,
{
    let inspection = inspect(engine, line)
        .map_err(|error| format_inspect_history_entry_error_message(&error))?;

    Ok(format_show_history_command_lines_with_context(
        &inspection,
        &engine.simplifier.context,
        |context, expr_info| {
            format_history_eval_metadata_sections(
                context,
                &expr_info.required_conditions,
                &expr_info.domain_warnings,
                &expr_info.blocked_hints,
            )
        },
    ))
}

/// Evaluate `show #id` command using a generic context.
pub fn evaluate_show_command_lines<C: ShowCommandContext>(
    context: &mut C,
    engine: &mut Engine,
    line: &str,
) -> Result<Vec<String>, String> {
    evaluate_show_command_lines_with(engine, line, |engine, line| {
        context.inspect_history_entry_input(engine, line)
    })
}
