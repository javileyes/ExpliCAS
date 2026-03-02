//! substitute subcommand handler.
//!
//! Keeps CLI I/O and delegates canonical JSON/text logic to `cas_solver`.

use crate::{OutputFormat, SubstituteArgs, SubstituteModeArg};

/// Run the substitute command.
pub fn run(args: SubstituteArgs) {
    let mode_str = match args.mode {
        SubstituteModeArg::Exact => "exact",
        SubstituteModeArg::Power => "power",
    };

    if matches!(args.format, OutputFormat::Json) {
        let out = cas_solver::evaluate_substitute_subcommand_json(
            &args.expr,
            &args.target,
            &args.replacement,
            mode_str,
            args.steps,
        );
        println!("{}", out);
        return;
    }

    let lines = match cas_solver::evaluate_substitute_subcommand_text_lines_with_mode(
        &args.expr,
        &args.target,
        &args.replacement,
        mode_str,
        args.steps,
    ) {
        Ok(lines) => lines,
        Err(message) => {
            eprintln!("{}", message);
            std::process::exit(1);
        }
    };

    for line in lines {
        println!("{}", line);
    }
}
