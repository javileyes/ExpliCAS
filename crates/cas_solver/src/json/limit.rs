use crate::{limit, Approach, Budget, LimitOptions, PreSimplifyMode};
use cas_formatter::DisplayExpr;

#[derive(Debug, Clone)]
pub struct LimitEvalResult {
    pub result: String,
    pub warning: Option<String>,
}

#[derive(Debug, Clone)]
pub enum LimitEvalError {
    Parse(String),
    Limit(String),
}

fn serialize_limit_json(value: &serde_json::Value, pretty: bool) -> String {
    if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .unwrap_or_else(|e| {
        let fallback = serde_json::json!({
            "ok": false,
            "error": format!("JSON serialization failed: {}", e),
            "code": "INTERNAL_ERROR"
        });
        serde_json::to_string(&fallback).unwrap_or_else(|_| {
            "{\"ok\":false,\"error\":\"JSON serialization failed\",\"code\":\"INTERNAL_ERROR\"}"
                .to_string()
        })
    })
}

/// Evaluate a limit from string input using parser + limit engine.
pub fn eval_limit_from_str(
    expr: &str,
    var: &str,
    approach: Approach,
    presimplify: PreSimplifyMode,
) -> Result<LimitEvalResult, LimitEvalError> {
    let mut ctx = cas_ast::Context::new();

    let parsed = cas_parser::parse(expr, &mut ctx)
        .map_err(|e| LimitEvalError::Parse(format!("Parse error: {}", e)))?;

    let var_id = ctx.var(var);
    let mut budget = Budget::new();
    let opts = LimitOptions {
        presimplify,
        ..Default::default()
    };

    match limit(&mut ctx, parsed, var_id, approach, &opts, &mut budget) {
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

/// Evaluate a limit and return canonical JSON used by CLI/FFI adapters.
///
/// Preserves existing `cas_cli` contract:
/// - success: `{ ok: true, result, warning? }`
/// - parse failure: `{ ok: false, error: \"Parse error: ...\", code: \"PARSE_ERROR\" }`
/// - limit failure: `{ ok: false, error: \"...\", code: \"LIMIT_ERROR\" }`
pub fn limit_str_to_json(
    expr: &str,
    var: &str,
    approach: Approach,
    presimplify: PreSimplifyMode,
    pretty: bool,
) -> String {
    match eval_limit_from_str(expr, var, approach, presimplify) {
        Ok(limit_result) => {
            let mut out = serde_json::json!({
                "ok": true,
                "result": limit_result.result,
            });
            if let Some(warning) = limit_result.warning {
                out["warning"] = serde_json::Value::String(warning);
            }
            serialize_limit_json(&out, pretty)
        }
        Err(LimitEvalError::Parse(message)) => serialize_limit_json(
            &serde_json::json!({
                "ok": false,
                "error": message,
                "code": "PARSE_ERROR"
            }),
            pretty,
        ),
        Err(LimitEvalError::Limit(message)) => serialize_limit_json(
            &serde_json::json!({
                "ok": false,
                "error": message,
                "code": "LIMIT_ERROR"
            }),
            pretty,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_str_to_json_parse_error_contract() {
        let out = limit_str_to_json(
            "sin(",
            "x",
            Approach::PosInfinity,
            PreSimplifyMode::Off,
            false,
        );
        let v: serde_json::Value = serde_json::from_str(&out).expect("json");
        assert_eq!(v["ok"], false);
        assert_eq!(v["code"], "PARSE_ERROR");
        assert!(v["error"]
            .as_str()
            .unwrap_or_default()
            .starts_with("Parse error:"));
    }

    #[test]
    fn limit_str_to_json_success_contract() {
        let out = limit_str_to_json(
            "(x^2+1)/(2*x^2-3)",
            "x",
            Approach::PosInfinity,
            PreSimplifyMode::Off,
            false,
        );
        let v: serde_json::Value = serde_json::from_str(&out).expect("json");
        assert_eq!(v["ok"], true);
        assert!(v.get("result").is_some());
    }

    #[test]
    fn eval_limit_from_str_parse_error_contract() {
        let out = eval_limit_from_str("sin(", "x", Approach::PosInfinity, PreSimplifyMode::Off);
        match out {
            Err(LimitEvalError::Parse(message)) => {
                assert!(message.starts_with("Parse error:"));
            }
            _ => panic!("expected parse error"),
        }
    }
}
