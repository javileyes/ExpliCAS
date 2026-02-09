use serde::{Deserialize, Serialize};

use super::response::*;

/// Options for substitute JSON operation.
#[derive(Deserialize, Debug)]
pub struct SubstituteJsonOptions {
    /// Substitution mode: "exact" or "power" (default: "power")
    #[serde(default = "default_substitute_mode")]
    pub mode: String,

    /// Include substitution steps in output
    #[serde(default)]
    pub steps: bool,

    /// Pretty-print JSON output
    #[serde(default)]
    pub pretty: bool,
}

impl Default for SubstituteJsonOptions {
    fn default() -> Self {
        Self {
            mode: "power".into(),
            steps: false,
            pretty: false,
        }
    }
}

fn default_substitute_mode() -> String {
    "power".into()
}

/// Substitute JSON response with request echo and options.
#[derive(Serialize, Debug)]
pub struct SubstituteJsonResponse {
    /// Schema version for API stability
    pub schema_version: u8,

    /// True if operation succeeded
    pub ok: bool,

    /// Result expression (success only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,

    /// Error details (failure only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<EngineJsonError>,

    /// Request echo for reproducibility
    pub request: SubstituteRequestEcho,

    /// Options used
    pub options: SubstituteOptionsJson,

    /// Substitution steps (if requested)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<EngineJsonSubstep>,
}

impl SubstituteJsonResponse {
    /// Serialize to JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|e| {
            format!(r#"{{"schema_version":1,"ok":false,"error":{{"kind":"InternalError","code":"E_INTERNAL","message":"JSON serialization failed: {}"}}}}"#, e)
        })
    }

    /// Serialize to pretty JSON string.
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| {
            format!(r#"{{"schema_version":1,"ok":false,"error":{{"kind":"InternalError","code":"E_INTERNAL","message":"JSON serialization failed: {}"}}}}"#, e)
        })
    }
}

/// Request echo for substitute.
#[derive(Serialize, Debug)]
pub struct SubstituteRequestEcho {
    pub expr: String,
    pub target: String,
    #[serde(rename = "with")]
    pub with_expr: String,
}

/// Options echo for substitute.
#[derive(Serialize, Debug)]
pub struct SubstituteOptionsJson {
    pub substitute: SubstituteOptionsInner,
}

#[derive(Serialize, Debug)]
pub struct SubstituteOptionsInner {
    pub mode: String,
    pub steps: bool,
}

/// Substitute an expression and return JSON response.
///
/// This is the **canonical entry point** for all JSON-returning substitution.
/// Both CLI and FFI should use this to ensure consistent behavior.
///
/// # Arguments
/// * `expr_str` - Expression string to substitute in
/// * `target_str` - Target expression to replace
/// * `with_str` - Replacement expression
/// * `opts_json` - Options JSON string (optional, see `SubstituteJsonOptions`)
///
/// # Returns
/// JSON string with `SubstituteJsonResponse` (schema v1).
/// Always returns valid JSON, even on errors.
pub fn substitute_str_to_json(
    expr_str: &str,
    target_str: &str,
    with_str: &str,
    opts_json: Option<&str>,
) -> String {
    // Parse options (with defaults)
    let opts: SubstituteJsonOptions = match opts_json {
        Some(json) => serde_json::from_str(json).unwrap_or_default(),
        None => SubstituteJsonOptions::default(),
    };

    let request = SubstituteRequestEcho {
        expr: expr_str.to_string(),
        target: target_str.to_string(),
        with_expr: with_str.to_string(),
    };

    let options = SubstituteOptionsJson {
        substitute: SubstituteOptionsInner {
            mode: opts.mode.clone(),
            steps: opts.steps,
        },
    };

    // Create context
    let mut ctx = cas_ast::Context::new();

    // Parse expressions
    let expr = match cas_parser::parse(expr_str, &mut ctx) {
        Ok(id) => id,
        Err(e) => {
            let resp = SubstituteJsonResponse {
                schema_version: SCHEMA_VERSION,
                ok: false,
                result: None,
                error: Some(EngineJsonError {
                    kind: "ParseError",
                    code: "E_PARSE",
                    message: format!("Failed to parse expression: {}", e),
                    span: e.span().map(SpanJson::from),
                    details: serde_json::Value::Null,
                }),
                request,
                options,
                steps: vec![],
            };
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    let target = match cas_parser::parse(target_str, &mut ctx) {
        Ok(id) => id,
        Err(e) => {
            let resp = SubstituteJsonResponse {
                schema_version: SCHEMA_VERSION,
                ok: false,
                result: None,
                error: Some(EngineJsonError {
                    kind: "ParseError",
                    code: "E_PARSE",
                    message: format!("Failed to parse target: {}", e),
                    span: e.span().map(SpanJson::from),
                    details: serde_json::Value::Null,
                }),
                request,
                options,
                steps: vec![],
            };
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    let replacement = match cas_parser::parse(with_str, &mut ctx) {
        Ok(id) => id,
        Err(e) => {
            let resp = SubstituteJsonResponse {
                schema_version: SCHEMA_VERSION,
                ok: false,
                result: None,
                error: Some(EngineJsonError {
                    kind: "ParseError",
                    code: "E_PARSE",
                    message: format!("Failed to parse replacement: {}", e),
                    span: e.span().map(SpanJson::from),
                    details: serde_json::Value::Null,
                }),
                request,
                options,
                steps: vec![],
            };
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    // Build substitute options
    let sub_opts = match opts.mode.as_str() {
        "exact" => crate::substitute::SubstituteOptions::exact(),
        _ => crate::substitute::SubstituteOptions::power_aware_no_remainder(),
    };

    // Perform substitution
    let sub_result =
        crate::substitute::substitute_with_steps(&mut ctx, expr, target, replacement, sub_opts);

    // Strip __hold from result
    let clean_result = crate::engine::strip_all_holds(&mut ctx, sub_result.expr);
    let result_str = format!(
        "{}",
        cas_ast::display::DisplayExpr {
            context: &ctx,
            id: clean_result
        }
    );

    // Convert steps to JSON format
    let json_steps: Vec<EngineJsonSubstep> = if opts.steps {
        sub_result
            .steps
            .into_iter()
            .map(|s| EngineJsonSubstep {
                rule: s.rule,
                before: s.before,
                after: s.after,
                note: s.note,
            })
            .collect()
    } else {
        vec![]
    };

    let resp = SubstituteJsonResponse {
        schema_version: SCHEMA_VERSION,
        ok: true,
        result: Some(result_str),
        error: None,
        request,
        options,
        steps: json_steps,
    };

    if opts.pretty {
        resp.to_json_pretty()
    } else {
        resp.to_json()
    }
}
