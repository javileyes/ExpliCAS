//! Android JNI bridge for ExpliCAS engine.
//!
//! Provides a single JNI function `evalJson` that takes an expression string
//! and options JSON, returning the evaluation result as JSON (schema_version: 1).
//!
//! # Safety
//! - All panics are caught with `catch_unwind` to prevent crashes crossing FFI.
//! - Always returns valid JSON, even on errors.
//!
//! # Usage from Kotlin
//! ```kotlin
//! object CasNative {
//!     init { System.loadLibrary("cas_android_ffi") }
//!     external fun evalJson(expr: String, optsJson: String): String
//! }
//! ```

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use jni::objects::{JClass, JString};
use jni::sys::jstring;
use jni::JNIEnv;

use serde::{Deserialize, Serialize};

use cas_ast::Context;
use cas_engine::{Engine, EvalAction, EvalOutput, EvalRequest, EvalResult, SessionState};
use cas_parser::parse;

// ============================================================================
// JSON Types (matching CLI schema_version: 1)
// ============================================================================

/// Options JSON from Kotlin client
#[derive(Deserialize, Default)]
#[allow(dead_code)] // pretty field reserved for future use
struct OptsJson {
    #[serde(default)]
    budget: BudgetOptsJson,
    #[serde(default)]
    pretty: bool,
}

#[derive(Deserialize)]
struct BudgetOptsJson {
    #[serde(default = "default_preset")]
    preset: String,
    #[serde(default = "default_mode")]
    mode: String,
}

impl Default for BudgetOptsJson {
    fn default() -> Self {
        Self {
            preset: default_preset(),
            mode: default_mode(),
        }
    }
}

fn default_preset() -> String {
    "cli".to_string()
}
fn default_mode() -> String {
    "best-effort".to_string()
}

/// Response JSON to Kotlin client
#[derive(Serialize)]
struct ResponseJson {
    schema_version: u32,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result_truncated: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    steps_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    budget: Option<BudgetResponseJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<ErrorJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timings_us: Option<TimingsJson>,
}

#[derive(Serialize)]
struct BudgetResponseJson {
    preset: String,
    mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    exceeded: Option<BudgetExceededJson>,
}

#[derive(Serialize)]
struct BudgetExceededJson {
    operation: String,
    metric: String,
    used: u64,
    limit: u64,
}

#[derive(Serialize)]
struct ErrorJson {
    kind: String,
    message: String,
}

#[derive(Serialize)]
struct TimingsJson {
    parse_us: u64,
    simplify_us: u64,
    total_us: u64,
}

impl ResponseJson {
    fn success(
        input: String,
        result: String,
        budget_preset: String,
        budget_mode: String,
        steps_count: usize,
        timings: TimingsJson,
    ) -> Self {
        Self {
            schema_version: 1,
            ok: true,
            input: Some(input),
            result: Some(result),
            result_truncated: Some(false),
            steps_count: Some(steps_count),
            budget: Some(BudgetResponseJson {
                preset: budget_preset,
                mode: budget_mode,
                exceeded: None,
            }),
            error: None,
            timings_us: Some(timings),
        }
    }

    fn error(kind: &str, message: String) -> Self {
        Self {
            schema_version: 1,
            ok: false,
            input: None,
            result: None,
            result_truncated: None,
            steps_count: None,
            budget: Some(BudgetResponseJson {
                preset: "unknown".to_string(),
                mode: "unknown".to_string(),
                exceeded: None,
            }),
            error: Some(ErrorJson {
                kind: kind.to_string(),
                message,
            }),
            timings_us: None,
        }
    }

    fn error_with_budget(kind: &str, message: String, preset: String, mode: String) -> Self {
        Self {
            schema_version: 1,
            ok: false,
            input: None,
            result: None,
            result_truncated: None,
            steps_count: None,
            budget: Some(BudgetResponseJson {
                preset,
                mode,
                exceeded: None,
            }),
            error: Some(ErrorJson {
                kind: kind.to_string(),
                message,
            }),
            timings_us: None,
        }
    }

    #[allow(dead_code)] // useful fallback when budget info not yet available
    fn parse_error(message: String) -> Self {
        Self::error("ParseError", message)
    }

    fn parse_error_with_budget(message: String, preset: String, mode: String) -> Self {
        Self::error_with_budget("ParseError", message, preset, mode)
    }

    #[allow(dead_code)] // useful fallback when budget info not yet available
    fn eval_error(message: String) -> Self {
        Self::error("EvalError", message)
    }

    fn eval_error_with_budget(message: String, preset: String, mode: String) -> Self {
        Self::error_with_budget("EvalError", message, preset, mode)
    }

    fn internal_error(message: String) -> Self {
        Self::error("InternalError", message)
    }
}

// ============================================================================
// JNI Entry Points
// ============================================================================

/// ABI version for diagnostics
const ABI_VERSION: i32 = 1;

/// JNI function: Java_es_javiergimenez_explicas_CasNative_abiVersion
///
/// Returns the ABI version for diagnostics. Useful for detecting mismatches
/// between the Kotlin code and the native library.
#[no_mangle]
pub extern "system" fn Java_es_javiergimenez_explicas_CasNative_abiVersion(
    _env: JNIEnv,
    _class: JClass,
) -> jni::sys::jint {
    ABI_VERSION
}

/// JNI function: Java_es_javiergimenez_explicas_CasNative_evalJson
///
/// # Arguments
/// * `expr` - Expression string (e.g., "2+x^2/(sqrt(2)+3)")
/// * `opts_json` - Options JSON (e.g., {"budget":{"preset":"cli","mode":"best-effort"}})
///
/// # Returns
/// JSON string with schema_version: 1
#[no_mangle]
pub extern "system" fn Java_es_javiergimenez_explicas_CasNative_evalJson<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    expr: JString<'local>,
    opts_json: JString<'local>,
) -> jstring {
    // Wrap everything in catch_unwind to prevent panics crossing FFI
    let result = catch_unwind(AssertUnwindSafe(|| {
        eval_json_inner(&mut env, expr, opts_json)
    }));

    match result {
        Ok(json_str) => {
            // Successfully got a JSON string (may be success or error)
            match env.new_string(&json_str) {
                Ok(s) => s.into_raw(),
                Err(_) => create_fallback_error(&mut env),
            }
        }
        Err(_) => {
            // Panic was caught - return stable error JSON
            let fallback = ResponseJson::internal_error("panic caught in native code".to_string());
            let json = serde_json::to_string(&fallback).unwrap_or_else(|_| {
                r#"{"schema_version":1,"ok":false,"error":{"kind":"InternalError","message":"panic caught"}}"#.to_string()
            });
            match env.new_string(&json) {
                Ok(s) => s.into_raw(),
                Err(_) => create_fallback_error(&mut env),
            }
        }
    }
}

/// Creates a hardcoded fallback error when everything else fails
fn create_fallback_error(env: &mut JNIEnv) -> jstring {
    let fallback = r#"{"schema_version":1,"ok":false,"error":{"kind":"InternalError","message":"JNI string allocation failed"}}"#;
    env.new_string(fallback)
        .map(|s| s.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

/// Inner evaluation function - may return error JSON but should not panic
fn eval_json_inner(env: &mut JNIEnv, expr: JString, opts_json: JString) -> String {
    let total_start = Instant::now();

    // 1. Extract strings from JNI
    let expr_str: String = match env.get_string(&expr) {
        Ok(s) => s.into(),
        Err(e) => {
            return serde_json::to_string(&ResponseJson::internal_error(format!(
                "Failed to read expr: {}",
                e
            )))
            .unwrap();
        }
    };

    let opts_str: String = match env.get_string(&opts_json) {
        Ok(s) => s.into(),
        Err(e) => {
            return serde_json::to_string(&ResponseJson::internal_error(format!(
                "Failed to read opts: {}",
                e
            )))
            .unwrap();
        }
    };

    // 2. Parse options JSON
    let opts: OptsJson = serde_json::from_str(&opts_str).unwrap_or_default();
    let budget_preset = opts.budget.preset.clone();
    let budget_mode = opts.budget.mode.clone();

    // 3. Create engine
    // Note: Budget presets are tracked for reporting but enforcement is at orchestrator level
    let mut engine = Engine::new();
    let mut state = SessionState::new();

    // 4. Parse expression
    let parse_start = Instant::now();
    let parsed = match parse(&expr_str, &mut engine.simplifier.context) {
        Ok(id) => id,
        Err(e) => {
            return serde_json::to_string(&ResponseJson::parse_error_with_budget(
                e.to_string(),
                budget_preset,
                budget_mode,
            ))
            .unwrap();
        }
    };
    let parse_us = parse_start.elapsed().as_micros() as u64;

    // 5. Build eval request
    let req = EvalRequest {
        raw_input: expr_str.clone(),
        parsed,
        kind: cas_engine::EntryKind::Expr(parsed),
        action: EvalAction::Simplify,
        auto_store: false,
    };

    // 6. Evaluate
    let simplify_start = Instant::now();
    let output: EvalOutput = match engine.eval(&mut state, req) {
        Ok(o) => o,
        Err(e) => {
            return serde_json::to_string(&ResponseJson::eval_error_with_budget(
                e.to_string(),
                budget_preset,
                budget_mode,
            ))
            .unwrap();
        }
    };
    let simplify_us = simplify_start.elapsed().as_micros() as u64;
    let total_us = total_start.elapsed().as_micros() as u64;

    // 7. Format result
    let result_str = match &output.result {
        EvalResult::Expr(e) => format_expr(&engine.simplifier.context, *e),
        EvalResult::Set(v) if !v.is_empty() => format_expr(&engine.simplifier.context, v[0]),
        EvalResult::Bool(b) => b.to_string(),
        _ => "(no result)".to_string(),
    };

    // 8. Build success response
    let response = ResponseJson::success(
        expr_str,
        result_str,
        budget_preset,
        budget_mode,
        output.steps.len(),
        TimingsJson {
            parse_us,
            simplify_us,
            total_us,
        },
    );

    serde_json::to_string(&response).unwrap_or_else(|e| {
        serde_json::to_string(&ResponseJson::internal_error(format!(
            "JSON serialization failed: {}",
            e
        )))
        .unwrap()
    })
}

/// Format an expression ID to string
fn format_expr(ctx: &Context, id: cas_ast::ExprId) -> String {
    use cas_ast::display::DisplayExpr;
    format!("{}", DisplayExpr { context: ctx, id })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_json_success() {
        let resp = ResponseJson::success(
            "2+2".to_string(),
            "4".to_string(),
            "cli".to_string(),
            "best-effort".to_string(),
            0,
            TimingsJson {
                parse_us: 100,
                simplify_us: 200,
                total_us: 300,
            },
        );
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"schema_version\":1"));
        assert!(json.contains("\"ok\":true"));
        assert!(json.contains("\"result\":\"4\""));
    }

    #[test]
    fn test_response_json_error() {
        let resp = ResponseJson::parse_error("unexpected token".to_string());
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"schema_version\":1"));
        assert!(json.contains("\"ok\":false"));
        assert!(json.contains("\"kind\":\"ParseError\""));
    }

    #[test]
    fn test_opts_json_default() {
        let opts: OptsJson = serde_json::from_str("{}").unwrap();
        assert_eq!(opts.budget.preset, "cli");
        assert_eq!(opts.budget.mode, "best-effort");
    }

    #[test]
    fn test_opts_json_custom() {
        let opts: OptsJson =
            serde_json::from_str(r#"{"budget":{"preset":"small","mode":"strict"}}"#).unwrap();
        assert_eq!(opts.budget.preset, "small");
        assert_eq!(opts.budget.mode, "strict");
    }
}
