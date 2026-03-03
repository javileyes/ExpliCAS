#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimitCommandInput<'a> {
    pub expr: &'a str,
    pub var: &'a str,
    pub approach: crate::Approach,
    pub presimplify: crate::PreSimplifyMode,
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
    pub approach: crate::Approach,
    pub result: String,
    pub warning: Option<String>,
}

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

pub fn parse_limit_command_input(rest: &str) -> LimitCommandInput<'_> {
    let parts = split_by_comma_ignoring_parens(rest);
    let expr = parts.first().copied().unwrap_or("").trim();
    let var = parts.get(1).copied().unwrap_or("x").trim();
    let dir = parts.get(2).copied().unwrap_or("infinity").trim();
    let mode = parts.get(3).copied().unwrap_or("off").trim();

    let approach = if dir.contains("-infinity") || dir.contains("-inf") {
        crate::Approach::NegInfinity
    } else {
        crate::Approach::PosInfinity
    };
    let presimplify = if mode.eq_ignore_ascii_case("safe") {
        crate::PreSimplifyMode::Safe
    } else {
        crate::PreSimplifyMode::Off
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
    match crate::json::eval_limit_from_str(
        parsed.expr,
        parsed.var,
        parsed.approach,
        parsed.presimplify,
    ) {
        Ok(limit_result) => Ok(LimitCommandEvalOutput {
            var: parsed.var.to_string(),
            approach: parsed.approach,
            result: limit_result.result,
            warning: limit_result.warning,
        }),
        Err(crate::json::LimitEvalError::Parse(message)) => {
            Err(LimitCommandEvalError::Parse(message))
        }
        Err(crate::json::LimitEvalError::Limit(message)) => {
            Err(LimitCommandEvalError::Limit(message))
        }
    }
}

/// Evaluate `limit` subcommand and return typed output for frontend printing.
pub fn evaluate_limit_subcommand_output(
    expr: &str,
    var: &str,
    approach: crate::Approach,
    presimplify: crate::PreSimplifyMode,
    json_output: bool,
) -> Result<LimitSubcommandOutput, LimitSubcommandError> {
    if json_output {
        return Ok(LimitSubcommandOutput::Json(crate::json::limit_str_to_json(
            expr,
            var,
            approach,
            presimplify,
            false,
        )));
    }

    match crate::json::eval_limit_from_str(expr, var, approach, presimplify) {
        Ok(limit_result) => Ok(LimitSubcommandOutput::Text {
            result: limit_result.result,
            warning: limit_result.warning,
        }),
        Err(crate::json::LimitEvalError::Parse(message)) => {
            Err(LimitSubcommandError::Parse(message))
        }
        Err(crate::json::LimitEvalError::Limit(message)) => {
            Err(LimitSubcommandError::Limit(message))
        }
    }
}

/// Format a user-facing limit subcommand error message.
pub fn format_limit_subcommand_error(error: &LimitSubcommandError) -> String {
    match error {
        LimitSubcommandError::Parse(message) => message.clone(),
        LimitSubcommandError::Limit(message) => format!("Error: {message}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        evaluate_limit_command_input, evaluate_limit_subcommand_output,
        format_limit_subcommand_error, parse_limit_command_input, LimitCommandEvalError,
        LimitSubcommandError, LimitSubcommandOutput,
    };

    #[test]
    fn parse_limit_command_input_defaults_var_and_direction() {
        let parsed = parse_limit_command_input("x^2");
        assert_eq!(parsed.expr, "x^2");
        assert_eq!(parsed.var, "x");
        assert_eq!(parsed.approach, crate::Approach::PosInfinity);
        assert_eq!(parsed.presimplify, crate::PreSimplifyMode::Off);
    }

    #[test]
    fn parse_limit_command_input_reads_neg_infinity_and_safe() {
        let parsed = parse_limit_command_input("(x^2+1)/(2*x^2-3), t, -infinity, safe");
        assert_eq!(parsed.expr, "(x^2+1)/(2*x^2-3)");
        assert_eq!(parsed.var, "t");
        assert_eq!(parsed.approach, crate::Approach::NegInfinity);
        assert_eq!(parsed.presimplify, crate::PreSimplifyMode::Safe);
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
        assert_eq!(out.approach, crate::Approach::PosInfinity);
        assert!(!out.result.is_empty());
    }

    #[test]
    fn evaluate_limit_subcommand_output_json_mode_returns_payload() {
        let out = evaluate_limit_subcommand_output(
            "(x^2+1)/(2*x^2-3)",
            "x",
            crate::Approach::PosInfinity,
            crate::PreSimplifyMode::Off,
            true,
        )
        .expect("json output");

        match out {
            LimitSubcommandOutput::Json(payload) => {
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
            crate::Approach::PosInfinity,
            crate::PreSimplifyMode::Off,
            false,
        )
        .expect_err("parse error");

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
