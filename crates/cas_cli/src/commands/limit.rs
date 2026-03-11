//! limit subcommand handler.
//!
//! Keeps CLI I/O and delegates stateless computation to `cas_solver`.

use crate::{ApproachArg, LimitArgs, OutputFormat, PreSimplifyArg};

/// Run the limit command.
pub fn run(args: LimitArgs) {
    let approach = match args.to {
        ApproachArg::Infinity => cas_solver::command_api::limit::LimitCommandApproach::Infinity,
        ApproachArg::NegInfinity => {
            cas_solver::command_api::limit::LimitCommandApproach::NegInfinity
        }
    };

    let presimplify = match args.presimplify {
        PreSimplifyArg::Off => cas_solver::command_api::limit::LimitCommandPreSimplify::Off,
        PreSimplifyArg::Safe => cas_solver::command_api::limit::LimitCommandPreSimplify::Safe,
    };

    match cas_solver::command_api::limit::evaluate_limit_subcommand(
        &args.expr,
        &args.var,
        approach,
        presimplify,
        matches!(args.format, OutputFormat::Json),
    ) {
        Ok(cas_solver::command_api::limit::LimitSubcommandOutput::Wire(out)) => {
            println!("{}", out);
        }
        Ok(cas_solver::command_api::limit::LimitSubcommandOutput::Text { result, warning }) => {
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
