use crate::limit_command_core::core::eval_limit_from_str;
use crate::limit_command_parse::parse_limit_command_input;
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
    match eval_limit_from_str(
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
    approach: cas_math::limit_types::Approach,
) -> LimitCommandApproach {
    match approach {
        cas_math::limit_types::Approach::PosInfinity => LimitCommandApproach::Infinity,
        cas_math::limit_types::Approach::NegInfinity => LimitCommandApproach::NegInfinity,
        cas_math::limit_types::Approach::Finite(_)
        | cas_math::limit_types::Approach::FiniteOneSided(_, _) => {
            unreachable!("limit command parser does not construct finite approaches")
        }
    }
}
