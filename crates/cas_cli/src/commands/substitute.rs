//! substitute subcommand handler.
//!
//! Keeps CLI I/O and delegates stateless wire/text logic to solver helpers.

use crate::{OutputFormat, SubstituteArgs, SubstituteModeArg};

/// Run the substitute command.
pub fn run(args: SubstituteArgs) {
    let mode = match args.mode {
        SubstituteModeArg::Exact => {
            cas_solver::command_api::substitute::SubstituteCommandMode::Exact
        }
        SubstituteModeArg::Power => {
            cas_solver::command_api::substitute::SubstituteCommandMode::Power
        }
    };

    let output = match cas_solver::command_api::substitute::evaluate_substitute_subcommand(
        &args.expr,
        &args.target,
        &args.replacement,
        mode,
        args.steps,
        matches!(args.format, OutputFormat::Json),
    ) {
        Ok(output) => output,
        Err(message) => {
            eprintln!("{}", message);
            std::process::exit(1);
        }
    };

    match output {
        cas_solver::command_api::substitute::SubstituteSubcommandOutput::Wire(payload) => {
            println!("{}", payload);
        }
        cas_solver::command_api::substitute::SubstituteSubcommandOutput::TextLines(lines) => {
            for line in lines {
                println!("{}", line);
            }
        }
    }
}
