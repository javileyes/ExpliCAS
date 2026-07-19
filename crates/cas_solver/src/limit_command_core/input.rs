use crate::limit_command_core::core::eval_limit_from_str_spec;
use crate::limit_command_parse::parse_limit_command_input;
use crate::limit_command_parse_types::LimitCommandApproachSpec;
use cas_api_models::{LimitCommandApproach, LimitCommandEvalError, LimitCommandEvalOutput};

pub(crate) fn evaluate_limit_command_input_in_domain(
    input: &str,
    complex_enabled: bool,
) -> Result<LimitCommandEvalOutput, LimitCommandEvalError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(LimitCommandEvalError::EmptyInput);
    }

    let parsed = parse_limit_command_input(trimmed).map_err(LimitCommandEvalError::Parse)?;
    match eval_limit_from_str_spec(
        parsed.expr,
        parsed.var,
        parsed.approach,
        parsed.presimplify,
        complex_enabled,
    ) {
        Ok(limit_result) => Ok(LimitCommandEvalOutput {
            var: parsed.var.to_string(),
            approach: limit_command_approach_from_runtime(parsed.approach),
            result: limit_result.result,
            warning: limit_result.warning,
        }),
        Err(cas_api_models::LimitEvalError::Parse(message)) => {
            Err(LimitCommandEvalError::Parse(message))
        }
        Err(cas_api_models::LimitEvalError::Limit(message)) => {
            Err(LimitCommandEvalError::Limit(message))
        }
    }
}

fn limit_command_approach_from_runtime(
    approach: LimitCommandApproachSpec<'_>,
) -> LimitCommandApproach {
    match approach {
        LimitCommandApproachSpec::PosInfinity => LimitCommandApproach::Infinity,
        LimitCommandApproachSpec::NegInfinity => LimitCommandApproach::NegInfinity,
        LimitCommandApproachSpec::Finite(point) => {
            LimitCommandApproach::Finite(point.trim().to_string())
        }
    }
}
