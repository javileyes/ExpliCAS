//! Android JNI bridge for ExpliCAS engine.
//!
//! Uses the canonical `cas_engine::json` module for all JSON responses.
//! Schema version: 1
//!
//! # Safety
//! - All panics are caught with `catch_unwind` to prevent crashes crossing FFI.
//! - Always returns valid JSON, even on errors.
//!
//! # Usage from Kotlin
//! ```kotlin
//! object CasNative {
//!     init { System.loadLibrary("cas_android_ffi") }
//!     external fun abiVersion(): Int
//!     external fun evalJson(expr: String, optsJson: String): String
//! }
//! ```

use std::panic::{catch_unwind, AssertUnwindSafe};

use jni::objects::{JClass, JString};
use jni::sys::jstring;
use jni::JNIEnv;

// Use the canonical JSON API from cas_engine
use cas_engine::{
    eval_str_to_json, BudgetJsonInfo, EngineJsonError, EngineJsonResponse, SCHEMA_VERSION,
};

// ============================================================================
// JNI Entry Points
// ============================================================================

/// ABI version for diagnostics
const ABI_VERSION: i32 = 2; // Bumped for schema v1 migration

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
///
/// Uses `cas_engine::eval_str_to_json` which is the canonical entry point.
#[no_mangle]
pub extern "system" fn Java_es_javiergimenez_explicas_CasNative_evalJson<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    expr: JString<'local>,
    opts_json: JString<'local>,
) -> jstring {
    // Wrap everything in catch_unwind to prevent panics crossing FFI
    let result = catch_unwind(AssertUnwindSafe(|| {
        eval_json_core(&mut env, expr, opts_json)
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
            let json = internal_error_json("panic caught in native code");
            match env.new_string(&json) {
                Ok(s) => s.into_raw(),
                Err(_) => create_fallback_error(&mut env),
            }
        }
    }
}

/// Creates a hardcoded fallback error when everything else fails
fn create_fallback_error(env: &mut JNIEnv) -> jstring {
    let fallback = r#"{"schema_version":1,"ok":false,"error":{"kind":"InternalError","code":"E_INTERNAL","message":"JNI string allocation failed"},"budget":{"preset":"unknown","mode":"strict"}}"#;
    env.new_string(fallback)
        .map(|s| s.into_raw())
        .unwrap_or(std::ptr::null_mut())
}

/// Create an internal error JSON response
fn internal_error_json(message: &str) -> String {
    let error = EngineJsonError::simple("InternalError", "E_INTERNAL", message);
    let budget = BudgetJsonInfo::new("unknown", true);
    let resp = EngineJsonResponse {
        schema_version: SCHEMA_VERSION,
        ok: false,
        result: None,
        error: Some(error),
        steps: vec![],
        warnings: vec![],
        assumptions: vec![],
        budget,
    };
    resp.to_json()
}

/// Core evaluation function - uses canonical cas_engine::eval_str_to_json
///
/// This is the testable inner function that doesn't require JNI.
pub fn eval_json_core(env: &mut JNIEnv, expr: JString, opts_json: JString) -> String {
    // 1. Extract strings from JNI
    let expr_str: String = match env.get_string(&expr) {
        Ok(s) => s.into(),
        Err(e) => {
            return internal_error_json(&format!("Failed to read expr: {}", e));
        }
    };

    let opts_str: String = match env.get_string(&opts_json) {
        Ok(s) => s.into(),
        Err(e) => {
            return internal_error_json(&format!("Failed to read opts: {}", e));
        }
    };

    // 2. Use canonical eval function
    eval_str_to_json(&expr_str, &opts_str)
}

/// JNI function: Java_es_javiergimenez_explicas_CasNative_substituteJson
///
/// # Arguments
/// * `expr` - Expression to substitute in
/// * `target` - Target expression to replace
/// * `with_expr` - Replacement expression
/// * `opts_json` - Options JSON (e.g., {"mode":"power","steps":true})
///
/// # Returns
/// JSON string with schema_version: 1
///
/// Uses `cas_engine::substitute_str_to_json` which is the canonical entry point.
#[no_mangle]
pub extern "system" fn Java_es_javiergimenez_explicas_CasNative_substituteJson<'local>(
    mut env: JNIEnv<'local>,
    _class: JClass<'local>,
    expr: JString<'local>,
    target: JString<'local>,
    with_expr: JString<'local>,
    opts_json: JString<'local>,
) -> jstring {
    let result = catch_unwind(AssertUnwindSafe(|| {
        substitute_json_core(&mut env, expr, target, with_expr, opts_json)
    }));

    match result {
        Ok(json_str) => match env.new_string(&json_str) {
            Ok(s) => s.into_raw(),
            Err(_) => create_fallback_error(&mut env),
        },
        Err(_) => {
            let json = internal_error_json("panic caught in substitute");
            match env.new_string(&json) {
                Ok(s) => s.into_raw(),
                Err(_) => create_fallback_error(&mut env),
            }
        }
    }
}

/// Core substitute function - uses canonical cas_engine::substitute_str_to_json
pub fn substitute_json_core(
    env: &mut JNIEnv,
    expr: JString,
    target: JString,
    with_expr: JString,
    opts_json: JString,
) -> String {
    let expr_str: String = match env.get_string(&expr) {
        Ok(s) => s.into(),
        Err(e) => return internal_error_json(&format!("Failed to read expr: {}", e)),
    };

    let target_str: String = match env.get_string(&target) {
        Ok(s) => s.into(),
        Err(e) => return internal_error_json(&format!("Failed to read target: {}", e)),
    };

    let with_str: String = match env.get_string(&with_expr) {
        Ok(s) => s.into(),
        Err(e) => return internal_error_json(&format!("Failed to read with: {}", e)),
    };

    let opts_str: String = match env.get_string(&opts_json) {
        Ok(s) => s.into(),
        Err(_) => String::new(), // Empty opts is OK
    };

    let opts = if opts_str.is_empty() {
        None
    } else {
        Some(opts_str.as_str())
    };

    cas_engine::substitute_str_to_json(&expr_str, &target_str, &with_str, opts)
}

// ============================================================================
// Tests (without JNI - using direct function calls)
// ============================================================================

#[cfg(test)]
mod tests {
    use cas_engine::eval_str_to_json;
    use serde_json::Value;

    fn parse_json(s: &str) -> Value {
        serde_json::from_str(s).expect("valid JSON")
    }

    #[test]
    fn test_eval_success() {
        let json = eval_str_to_json("x + x", "{}");
        let v = parse_json(&json);

        assert_eq!(v["schema_version"], 1);
        assert_eq!(v["ok"], true);
        assert!(v["result"].is_string());
        assert!(v["budget"].is_object());
    }

    #[test]
    fn test_eval_parse_error() {
        let json = eval_str_to_json("(", "{}");
        let v = parse_json(&json);

        assert_eq!(v["schema_version"], 1);
        assert_eq!(v["ok"], false);
        assert_eq!(v["error"]["kind"], "ParseError");
        assert_eq!(v["error"]["code"], "E_PARSE");
    }

    #[test]
    fn test_eval_invalid_opts() {
        let json = eval_str_to_json("x+1", "{invalid");
        let v = parse_json(&json);

        assert_eq!(v["schema_version"], 1);
        assert_eq!(v["ok"], false);
        assert_eq!(v["error"]["kind"], "InvalidInput");
        assert_eq!(v["error"]["code"], "E_INVALID_INPUT");
        assert!(v["error"]["details"]["error"].is_string());
    }

    #[test]
    fn test_opts_defaults() {
        // Empty opts should use defaults
        let json = eval_str_to_json("2+2", "{}");
        let v = parse_json(&json);

        assert_eq!(v["ok"], true);
        assert_eq!(v["budget"]["preset"], "cli");
        assert_eq!(v["budget"]["mode"], "best-effort");
    }

    #[test]
    fn test_opts_custom() {
        let json = eval_str_to_json("2+2", r#"{"budget":{"preset":"small","mode":"strict"}}"#);
        let v = parse_json(&json);

        assert_eq!(v["ok"], true);
        assert_eq!(v["budget"]["preset"], "small");
        assert_eq!(v["budget"]["mode"], "strict");
    }

    #[test]
    fn test_no_hold_leak() {
        // Expression that might internally use __hold
        let json = eval_str_to_json("(x+1)^2 - (x+1)^2", "{}");
        let v = parse_json(&json);

        if let Some(result) = v["result"].as_str() {
            assert!(
                !result.contains("__hold"),
                "Result contains __hold: {}",
                result
            );
        }
    }

    #[test]
    fn test_steps_mode() {
        let json = eval_str_to_json("x + x", r#"{"steps":true}"#);
        let v = parse_json(&json);

        assert_eq!(v["ok"], true);
        // steps should be present (array, possibly empty)
        assert!(v["steps"].is_array());
    }
}
