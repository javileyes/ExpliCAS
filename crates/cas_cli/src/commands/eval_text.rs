//! eval (text output) command handler.
//!
//! Handles parse/eval flow for text mode and optional session persistence.

use super::output::CommandOutput;
use crate::{DomainArg, EvalArgs};

pub(crate) fn render(args: &EvalArgs) -> Result<CommandOutput, String> {
    let domain = match args.domain {
        DomainArg::Strict => "strict",
        DomainArg::Generic => "generic",
        DomainArg::Assume => "assume",
    };

    let (result, load_warning, save_warning) =
        cas_session::eval_api::evaluate_eval_text_command_with_session(
            args.session.as_deref(),
            domain,
            &args.expr,
            args.session.is_some(),
        );
    let mut output = CommandOutput::default();
    output.stderr_lines.extend(load_warning);

    match result {
        Ok(result_str) => output.stdout = result_str,
        Err(message) => return Err(message),
    }

    if let Some(warning) = save_warning {
        output.push_stderr_line(warning);
    }
    Ok(output)
}
