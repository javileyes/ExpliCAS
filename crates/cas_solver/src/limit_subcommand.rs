//! Stateless CLI-subcommand helpers for `limit`.

use crate::limit_command_eval::{
    evaluate_limit_subcommand_output, format_limit_subcommand_error, LimitSubcommandEvalOutput,
};

/// Limit direction for subcommand-level evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitCommandApproach {
    Infinity,
    NegInfinity,
}

/// Pre-simplification policy for subcommand-level limit evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitCommandPreSimplify {
    Off,
    Safe,
}

/// CLI-friendly output contract for `limit` subcommand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitSubcommandOutput {
    Json(String),
    Text {
        result: String,
        warning: Option<String>,
    },
}

/// Evaluate limit subcommand and map solver contracts to CLI-friendly output.
pub fn evaluate_limit_subcommand(
    expr: &str,
    var: &str,
    approach: LimitCommandApproach,
    presimplify: LimitCommandPreSimplify,
    json_output: bool,
) -> Result<LimitSubcommandOutput, String> {
    let approach = match approach {
        LimitCommandApproach::Infinity => crate::Approach::PosInfinity,
        LimitCommandApproach::NegInfinity => crate::Approach::NegInfinity,
    };
    let presimplify = match presimplify {
        LimitCommandPreSimplify::Off => crate::PreSimplifyMode::Off,
        LimitCommandPreSimplify::Safe => crate::PreSimplifyMode::Safe,
    };

    match evaluate_limit_subcommand_output(expr, var, approach, presimplify, json_output) {
        Ok(LimitSubcommandEvalOutput::Json(out)) => Ok(LimitSubcommandOutput::Json(out)),
        Ok(LimitSubcommandEvalOutput::Text { result, warning }) => {
            Ok(LimitSubcommandOutput::Text { result, warning })
        }
        Err(error) => Err(format_limit_subcommand_error(&error)),
    }
}
