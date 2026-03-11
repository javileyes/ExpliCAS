//! limit subcommand handler.
//!
//! Keeps CLI I/O and delegates stateless computation to `cas_solver`.

use crate::{ApproachArg, LimitArgs, OutputFormat, PreSimplifyArg};

/// Run the limit command.
pub fn run(args: LimitArgs) {
    let approach = match args.to {
        ApproachArg::Infinity => cas_solver::LimitCommandApproach::Infinity,
        ApproachArg::NegInfinity => cas_solver::LimitCommandApproach::NegInfinity,
    };

    let presimplify = match args.presimplify {
        PreSimplifyArg::Off => cas_solver::LimitCommandPreSimplify::Off,
        PreSimplifyArg::Safe => cas_solver::LimitCommandPreSimplify::Safe,
    };

    match cas_solver::evaluate_limit_subcommand(
        &args.expr,
        &args.var,
        approach,
        presimplify,
        matches!(args.format, OutputFormat::Json),
    ) {
        Ok(cas_solver::LimitSubcommandOutput::Wire(out)) => {
            println!("{}", out);
        }
        Ok(cas_solver::LimitSubcommandOutput::Text { result, warning }) => {
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
