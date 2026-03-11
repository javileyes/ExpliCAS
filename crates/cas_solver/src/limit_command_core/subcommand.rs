use crate::limit_command_core::core::{eval_limit_from_str, limit_str_to_wire};
use crate::limit_command_types::{LimitSubcommandEvalError, LimitSubcommandEvalOutput};

pub fn evaluate_limit_subcommand_output(
    expr: &str,
    var: &str,
    approach: crate::Approach,
    presimplify: crate::PreSimplifyMode,
    wire_output: bool,
) -> Result<LimitSubcommandEvalOutput, LimitSubcommandEvalError> {
    if wire_output {
        return Ok(LimitSubcommandEvalOutput::Wire(limit_str_to_wire(
            expr,
            var,
            approach,
            presimplify,
            false,
        )));
    }

    match eval_limit_from_str(expr, var, approach, presimplify) {
        Ok(limit_result) => Ok(LimitSubcommandEvalOutput::Text {
            result: limit_result.result,
            warning: limit_result.warning,
        }),
        Err(cas_api_models::LimitEvalError::Parse(message)) => {
            Err(LimitSubcommandEvalError::Parse(message))
        }
        Err(cas_api_models::LimitEvalError::Limit(message)) => {
            Err(LimitSubcommandEvalError::Limit(message))
        }
    }
}
