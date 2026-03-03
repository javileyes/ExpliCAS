use cas_api_models::{LimitEvalError, LimitEvalResult, LimitJsonResponse};
use cas_formatter::DisplayExpr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimitCommandInput<'a> {
    pub expr: &'a str,
    pub var: &'a str,
    pub approach: cas_solver::Approach,
    pub presimplify: cas_solver::PreSimplifyMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitCommandEvalError {
    EmptyInput,
    Parse(String),
    Limit(String),
}

#[derive(Debug, Clone)]
pub struct LimitCommandEvalOutput {
    pub var: String,
    pub approach: cas_solver::Approach,
    pub result: String,
    pub warning: Option<String>,
}

/// Output payload for CLI-style `limit` subcommand execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitSubcommandEvalOutput {
    Json(String),
    Text {
        result: String,
        warning: Option<String>,
    },
}

/// Error payload for CLI-style `limit` subcommand execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LimitSubcommandEvalError {
    Parse(String),
    Limit(String),
}

fn split_by_comma_ignoring_parens(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut balance = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '(' | '[' | '{' => balance += 1,
            ')' | ']' | '}' => balance -= 1,
            ',' if balance == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }

    if start < s.len() {
        parts.push(&s[start..]);
    }

    parts
}

fn eval_limit_from_str(
    expr: &str,
    var: &str,
    approach: cas_solver::Approach,
    presimplify: cas_solver::PreSimplifyMode,
) -> Result<LimitEvalResult, LimitEvalError> {
    let mut ctx = cas_ast::Context::new();
    let parsed = cas_parser::parse(expr, &mut ctx)
        .map_err(|e| LimitEvalError::Parse(format!("Parse error: {}", e)))?;

    let var_id = ctx.var(var);
    let mut budget = cas_solver::Budget::new();
    let opts = cas_solver::LimitOptions {
        presimplify,
        ..Default::default()
    };

    match cas_solver::limit(&mut ctx, parsed, var_id, approach, &opts, &mut budget) {
        Ok(limit_result) => {
            let result = DisplayExpr {
                context: &ctx,
                id: limit_result.expr,
            }
            .to_string();
            Ok(LimitEvalResult {
                result,
                warning: limit_result.warning,
            })
        }
        Err(e) => Err(LimitEvalError::Limit(e.to_string())),
    }
}

fn limit_str_to_json(
    expr: &str,
    var: &str,
    approach: cas_solver::Approach,
    presimplify: cas_solver::PreSimplifyMode,
    pretty: bool,
) -> String {
    let response = match eval_limit_from_str(expr, var, approach, presimplify) {
        Ok(limit_result) => LimitJsonResponse::ok(limit_result.result, limit_result.warning),
        Err(LimitEvalError::Parse(message)) => LimitJsonResponse::parse_error(message),
        Err(LimitEvalError::Limit(message)) => LimitJsonResponse::limit_error(message),
    };

    response.to_json_with_pretty(pretty)
}

pub fn parse_limit_command_input(rest: &str) -> LimitCommandInput<'_> {
    let parts = split_by_comma_ignoring_parens(rest);
    let expr = parts.first().copied().unwrap_or("").trim();
    let var = parts.get(1).copied().unwrap_or("x").trim();
    let dir = parts.get(2).copied().unwrap_or("infinity").trim();
    let mode = parts.get(3).copied().unwrap_or("off").trim();

    let approach = if dir.contains("-infinity") || dir.contains("-inf") {
        cas_solver::Approach::NegInfinity
    } else {
        cas_solver::Approach::PosInfinity
    };
    let presimplify = if mode.eq_ignore_ascii_case("safe") {
        cas_solver::PreSimplifyMode::Safe
    } else {
        cas_solver::PreSimplifyMode::Off
    };

    LimitCommandInput {
        expr,
        var,
        approach,
        presimplify,
    }
}

pub fn evaluate_limit_command_input(
    input: &str,
) -> Result<LimitCommandEvalOutput, LimitCommandEvalError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(LimitCommandEvalError::EmptyInput);
    }

    let parsed = parse_limit_command_input(trimmed);
    match eval_limit_from_str(parsed.expr, parsed.var, parsed.approach, parsed.presimplify) {
        Ok(limit_result) => Ok(LimitCommandEvalOutput {
            var: parsed.var.to_string(),
            approach: parsed.approach,
            result: limit_result.result,
            warning: limit_result.warning,
        }),
        Err(LimitEvalError::Parse(message)) => Err(LimitCommandEvalError::Parse(message)),
        Err(LimitEvalError::Limit(message)) => Err(LimitCommandEvalError::Limit(message)),
    }
}

pub fn evaluate_limit_subcommand_output(
    expr: &str,
    var: &str,
    approach: cas_solver::Approach,
    presimplify: cas_solver::PreSimplifyMode,
    json_output: bool,
) -> Result<LimitSubcommandEvalOutput, LimitSubcommandEvalError> {
    if json_output {
        return Ok(LimitSubcommandEvalOutput::Json(limit_str_to_json(
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
        Err(LimitEvalError::Parse(message)) => Err(LimitSubcommandEvalError::Parse(message)),
        Err(LimitEvalError::Limit(message)) => Err(LimitSubcommandEvalError::Limit(message)),
    }
}

pub fn format_limit_subcommand_error(error: &LimitSubcommandEvalError) -> String {
    match error {
        LimitSubcommandEvalError::Parse(message) => message.clone(),
        LimitSubcommandEvalError::Limit(message) => format!("Error: {message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_limit_command_input, evaluate_limit_subcommand_output,
        format_limit_subcommand_error, parse_limit_command_input, LimitCommandEvalError,
        LimitSubcommandEvalError, LimitSubcommandEvalOutput,
    };

    #[test]
    fn parse_limit_command_input_defaults_var_and_direction() {
        let parsed = parse_limit_command_input("x^2");
        assert_eq!(parsed.expr, "x^2");
        assert_eq!(parsed.var, "x");
        assert_eq!(parsed.approach, cas_solver::Approach::PosInfinity);
        assert_eq!(parsed.presimplify, cas_solver::PreSimplifyMode::Off);
    }

    #[test]
    fn parse_limit_command_input_reads_neg_infinity_and_safe() {
        let parsed = parse_limit_command_input("(x^2+1)/(2*x^2-3), t, -infinity, safe");
        assert_eq!(parsed.expr, "(x^2+1)/(2*x^2-3)");
        assert_eq!(parsed.var, "t");
        assert_eq!(parsed.approach, cas_solver::Approach::NegInfinity);
        assert_eq!(parsed.presimplify, cas_solver::PreSimplifyMode::Safe);
    }

    #[test]
    fn evaluate_limit_command_input_rejects_empty_input() {
        let err = evaluate_limit_command_input("  ").expect_err("expected empty-input error");
        assert_eq!(err, LimitCommandEvalError::EmptyInput);
    }

    #[test]
    fn evaluate_limit_command_input_computes_basic_limit() {
        let out = evaluate_limit_command_input("x^2, x, infinity").expect("limit eval");
        assert_eq!(out.var, "x");
        assert_eq!(out.approach, cas_solver::Approach::PosInfinity);
        assert!(!out.result.is_empty());
    }

    #[test]
    fn evaluate_limit_subcommand_output_json_mode_returns_payload() {
        let out = evaluate_limit_subcommand_output(
            "(x^2+1)/(2*x^2-3)",
            "x",
            cas_solver::Approach::PosInfinity,
            cas_solver::PreSimplifyMode::Off,
            true,
        )
        .expect("json output");

        match out {
            LimitSubcommandEvalOutput::Json(payload) => {
                let json: serde_json::Value = serde_json::from_str(&payload).expect("json");
                assert_eq!(json["ok"], true);
            }
            _ => panic!("expected json payload"),
        }
    }

    #[test]
    fn evaluate_limit_subcommand_output_parse_error_is_typed() {
        let err = evaluate_limit_subcommand_output(
            "sin(",
            "x",
            cas_solver::Approach::PosInfinity,
            cas_solver::PreSimplifyMode::Off,
            false,
        )
        .expect_err("parse error");

        match err {
            LimitSubcommandEvalError::Parse(message) => {
                assert!(message.starts_with("Parse error:"));
            }
            _ => panic!("expected parse error"),
        }
    }

    #[test]
    fn format_limit_subcommand_error_matches_cli_contract() {
        let parse = format_limit_subcommand_error(&LimitSubcommandEvalError::Parse(
            "Parse error: bad input".to_string(),
        ));
        assert_eq!(parse, "Parse error: bad input");

        let limit = format_limit_subcommand_error(&LimitSubcommandEvalError::Limit(
            "cannot compute".to_string(),
        ));
        assert_eq!(limit, "Error: cannot compute");
    }
}
