//! limit subcommand handler.
//!
//! Keeps CLI I/O and delegates computation to `cas_solver` JSON/eval APIs.

use crate::{ApproachArg, LimitArgs, OutputFormat, PreSimplifyArg};

/// Run the limit command.
pub fn run(args: LimitArgs) {
    use cas_solver::{Approach, PreSimplifyMode};

    let approach = match args.to {
        ApproachArg::Infinity => Approach::PosInfinity,
        ApproachArg::NegInfinity => Approach::NegInfinity,
    };

    let presimplify = match args.presimplify {
        PreSimplifyArg::Off => PreSimplifyMode::Off,
        PreSimplifyArg::Safe => PreSimplifyMode::Safe,
    };

    match super::limit_command::evaluate_limit_subcommand_output(
        &args.expr,
        &args.var,
        approach,
        presimplify,
        matches!(args.format, OutputFormat::Json),
    ) {
        Ok(super::limit_command::LimitSubcommandOutput::Json(out)) => {
            println!("{}", out);
        }
        Ok(super::limit_command::LimitSubcommandOutput::Text { result, warning }) => {
            println!("{}", result);
            if let Some(warning) = warning {
                eprintln!("Warning: {}", warning);
            }
        }
        Err(error) => {
            eprintln!(
                "{}",
                super::limit_command::format_limit_subcommand_error(&error)
            );
            std::process::exit(1);
        }
    }
}
