//! substitute subcommand handler.
//!
//! Keeps CLI I/O and delegates stateless wire/text logic to solver helpers.

use super::output::CommandOutput;
use crate::{OutputFormat, SubstituteArgs, SubstituteModeArg};

pub(crate) fn render(args: SubstituteArgs) -> Result<CommandOutput, String> {
    let mode = match args.mode {
        SubstituteModeArg::Exact => {
            cas_solver::command_api::substitute::SubstituteCommandMode::Exact
        }
        SubstituteModeArg::Power => {
            cas_solver::command_api::substitute::SubstituteCommandMode::Power
        }
    };

    let output = cas_solver::command_api::substitute::evaluate_substitute_subcommand(
        &args.expr,
        &args.target,
        &args.replacement,
        mode,
        args.steps,
        matches!(args.format, OutputFormat::Json),
    )?;

    match output {
        cas_solver::command_api::substitute::SubstituteSubcommandOutput::Wire(payload) => {
            Ok(CommandOutput::from_stdout(payload))
        }
        cas_solver::command_api::substitute::SubstituteSubcommandOutput::TextLines(lines) => {
            Ok(CommandOutput::from_stdout(lines.join("\n")))
        }
    }
}
