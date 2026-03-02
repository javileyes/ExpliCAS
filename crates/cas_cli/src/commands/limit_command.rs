/// Output payload for CLI-style `limit` subcommand execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitSubcommandOutput {
    Json(String),
    Text {
        result: String,
        warning: Option<String>,
    },
}

/// Error payload for CLI-style `limit` subcommand execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitSubcommandError {
    Parse(String),
    Limit(String),
}

/// Evaluate `limit` subcommand and return typed output for CLI printing.
pub fn evaluate_limit_subcommand_output(
    expr: &str,
    var: &str,
    approach: cas_solver::Approach,
    presimplify: cas_solver::PreSimplifyMode,
    json_output: bool,
) -> Result<LimitSubcommandOutput, LimitSubcommandError> {
    if json_output {
        return Ok(LimitSubcommandOutput::Json(
            cas_solver::json::limit_str_to_json(expr, var, approach, presimplify, false),
        ));
    }

    match cas_solver::json::eval_limit_from_str(expr, var, approach, presimplify) {
        Ok(limit_result) => Ok(LimitSubcommandOutput::Text {
            result: limit_result.result,
            warning: limit_result.warning,
        }),
        Err(cas_solver::json::LimitEvalError::Parse(message)) => {
            Err(LimitSubcommandError::Parse(message))
        }
        Err(cas_solver::json::LimitEvalError::Limit(message)) => {
            Err(LimitSubcommandError::Limit(message))
        }
    }
}

/// Format a user-facing limit subcommand error message.
pub fn format_limit_subcommand_error(error: &LimitSubcommandError) -> String {
    match error {
        LimitSubcommandError::Parse(message) => message.clone(),
        LimitSubcommandError::Limit(message) => format!("Error: {}", message),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_limit_subcommand_output, format_limit_subcommand_error, LimitSubcommandError,
        LimitSubcommandOutput,
    };

    #[test]
    fn evaluate_limit_subcommand_output_json_mode_returns_payload() {
        let out = match evaluate_limit_subcommand_output(
            "(x^2+1)/(2*x^2-3)",
            "x",
            cas_solver::Approach::PosInfinity,
            cas_solver::PreSimplifyMode::Off,
            true,
        ) {
            Ok(out) => out,
            Err(err) => panic!("json output failed: {err:?}"),
        };

        match out {
            LimitSubcommandOutput::Json(payload) => {
                let json: serde_json::Value = match serde_json::from_str(&payload) {
                    Ok(json) => json,
                    Err(err) => panic!("invalid json: {err}"),
                };
                assert_eq!(json["ok"], true);
            }
            _ => panic!("expected json payload"),
        }
    }

    #[test]
    fn evaluate_limit_subcommand_output_parse_error_is_typed() {
        let err = match evaluate_limit_subcommand_output(
            "sin(",
            "x",
            cas_solver::Approach::PosInfinity,
            cas_solver::PreSimplifyMode::Off,
            false,
        ) {
            Ok(_) => panic!("expected parse error"),
            Err(err) => err,
        };

        match err {
            LimitSubcommandError::Parse(message) => {
                assert!(message.starts_with("Parse error:"));
            }
            _ => panic!("expected parse error"),
        }
    }

    #[test]
    fn format_limit_subcommand_error_matches_cli_contract() {
        let parse = format_limit_subcommand_error(&LimitSubcommandError::Parse(
            "Parse error: bad input".to_string(),
        ));
        assert_eq!(parse, "Parse error: bad input");

        let limit = format_limit_subcommand_error(&LimitSubcommandError::Limit(
            "cannot compute".to_string(),
        ));
        assert_eq!(limit, "Error: cannot compute");
    }
}
