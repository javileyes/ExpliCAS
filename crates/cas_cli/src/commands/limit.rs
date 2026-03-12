//! limit subcommand handler.
//!
//! Keeps CLI I/O and delegates stateless computation to `cas_solver`.

use super::output::CommandOutput;
use crate::{ApproachArg, LimitArgs, OutputFormat, PreSimplifyArg};

pub(crate) fn render(args: LimitArgs) -> Result<CommandOutput, String> {
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
            Ok(CommandOutput::from_stdout(out))
        }
        Ok(cas_solver::command_api::limit::LimitSubcommandOutput::Text { result, warning }) => {
            let mut output = CommandOutput::from_stdout(result);
            if let Some(warning) = warning {
                output.push_stderr_line(format!("Warning: {}", warning));
            }
            Ok(output)
        }
        Err(message) => Err(message),
    }
}
