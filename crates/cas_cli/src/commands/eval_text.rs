//! eval (text output) command handler.
//!
//! Handles parse/eval flow for text mode and optional session persistence.

use super::output::CommandOutput;
use crate::EvalArgs;

pub(crate) fn render(args: &EvalArgs) -> Result<CommandOutput, String> {
    let (result, load_warning, save_warning) =
        cas_session::eval::evaluate_eval_command_with_session(
            args.session.as_deref(),
            super::eval::eval_command_config(&args.expr, args),
            |_steps, _events, _context, _steps_mode| Vec::new(),
        );
    let mut output = CommandOutput::default();
    output.stderr_lines.extend(load_warning);

    match result {
        Ok(result_wire) => {
            for warning in result_wire.warnings {
                output.push_stderr_line(format!("⚠ {} ({})", warning.assumption, warning.rule));
            }
            for line in
                cas_api_models::wire::format_blocked_hint_message_lines(&result_wire.blocked_hints)
            {
                output.push_stderr_line(line);
            }
            output.stdout = result_wire.result;
        }
        Err(message) => return Err(message),
    }

    if let Some(warning) = save_warning {
        output.push_stderr_line(warning);
    }
    Ok(output)
}
