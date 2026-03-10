//! Stateless CLI-subcommand helpers for `limit`.

use crate::limit_command_eval::{
    evaluate_limit_subcommand_output, format_limit_subcommand_error, LimitSubcommandEvalOutput,
};
pub use cas_solver_core::limit_subcommand_types::{
    LimitCommandApproach, LimitCommandPreSimplify, LimitSubcommandOutput,
};

/// Evaluate limit subcommand and map solver contracts to CLI-friendly output.
pub fn evaluate_limit_subcommand(
    expr: &str,
    var: &str,
    approach: LimitCommandApproach,
    presimplify: LimitCommandPreSimplify,
    wire_output: bool,
) -> Result<LimitSubcommandOutput, String> {
    let approach = match approach {
        LimitCommandApproach::Infinity => crate::Approach::PosInfinity,
        LimitCommandApproach::NegInfinity => crate::Approach::NegInfinity,
    };
    let presimplify = match presimplify {
        LimitCommandPreSimplify::Off => crate::PreSimplifyMode::Off,
        LimitCommandPreSimplify::Safe => crate::PreSimplifyMode::Safe,
    };

    match evaluate_limit_subcommand_output(expr, var, approach, presimplify, wire_output) {
        Ok(LimitSubcommandEvalOutput::Json(out)) => Ok(LimitSubcommandOutput::Json(out)),
        Ok(LimitSubcommandEvalOutput::Text { result, warning }) => {
            Ok(LimitSubcommandOutput::Text { result, warning })
        }
        Err(error) => Err(format_limit_subcommand_error(&error)),
    }
}
