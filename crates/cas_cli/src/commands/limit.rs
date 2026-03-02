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

    if matches!(args.format, OutputFormat::Json) {
        let out = cas_solver::json::limit_str_to_json(
            &args.expr,
            &args.var,
            approach,
            presimplify,
            false,
        );
        println!("{}", out);
        return;
    }

    match cas_solver::json::eval_limit_from_str(&args.expr, &args.var, approach, presimplify) {
        Ok(limit_result) => {
            println!("{}", limit_result.result);
            if let Some(warning) = &limit_result.warning {
                eprintln!("Warning: {}", warning);
            }
        }
        Err(cas_solver::json::LimitEvalError::Parse(message)) => {
            eprintln!("{}", message);
            std::process::exit(1);
        }
        Err(cas_solver::json::LimitEvalError::Limit(message)) => {
            eprintln!("Error: {}", message);
            std::process::exit(1);
        }
    }
}
