//! limit subcommand handler.
//!
//! Keeps CLI I/O and delegates computation to `cas_solver` JSON/eval APIs.

use crate::{ApproachArg, LimitArgs, OutputFormat, PreSimplifyArg};

/// Run the limit command.
pub fn run(args: LimitArgs) {
    let approach = match args.to {
        ApproachArg::Infinity => cas_session::LimitCommandApproach::Infinity,
        ApproachArg::NegInfinity => cas_session::LimitCommandApproach::NegInfinity,
    };

    let presimplify = match args.presimplify {
        PreSimplifyArg::Off => cas_session::LimitCommandPreSimplify::Off,
        PreSimplifyArg::Safe => cas_session::LimitCommandPreSimplify::Safe,
    };

    match cas_session::evaluate_limit_subcommand(
        &args.expr,
        &args.var,
        approach,
        presimplify,
        matches!(args.format, OutputFormat::Json),
    ) {
        Ok(cas_session::LimitSubcommandOutput::Json(out)) => {
            println!("{}", out);
        }
        Ok(cas_session::LimitSubcommandOutput::Text { result, warning }) => {
            println!("{}", result);
            if let Some(warning) = warning {
                eprintln!("Warning: {}", warning);
            }
        }
        Err(message) => {
            eprintln!("{}", message);
            std::process::exit(1);
        }
    }
}
