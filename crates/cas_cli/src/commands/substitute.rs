//! substitute subcommand handler.
//!
//! Keeps CLI I/O and delegates canonical JSON/text logic to local command helpers.

use crate::{OutputFormat, SubstituteArgs, SubstituteModeArg};

/// Run the substitute command.
pub fn run(args: SubstituteArgs) {
    let mode = match args.mode {
        SubstituteModeArg::Exact => cas_session::SubstituteCommandMode::Exact,
        SubstituteModeArg::Power => cas_session::SubstituteCommandMode::Power,
    };

    let output = match cas_session::evaluate_substitute_subcommand(
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
        cas_session::SubstituteSubcommandOutput::Wire(payload) => {
            println!("{}", payload);
        }
        cas_session::SubstituteSubcommandOutput::TextLines(lines) => {
            for line in lines {
                println!("{}", line);
            }
        }
    }
}
