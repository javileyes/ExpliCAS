use super::response::*;

/// Evaluate an expression and return JSON response.
///
/// This is the **canonical entry point** for all JSON-returning evaluation.
/// Both CLI and FFI should use this to ensure consistent behavior.
///
/// # Arguments
/// * `expr` - Expression string to evaluate
/// * `opts_json` - Options JSON string (see `JsonRunOptions`)
///
/// # Returns
/// JSON string with `EngineJsonResponse` (schema v1).
/// Always returns valid JSON, even on errors.
///
/// # Example
/// ```
/// use cas_engine::json::eval_str_to_json;
///
/// let json = eval_str_to_json("x + x", r#"{"budget":{"preset":"cli"}}"#);
/// assert!(json.contains("\"ok\":true"));
/// ```
pub fn eval_str_to_json(expr: &str, opts_json: &str) -> String {
    // Parse options (with defaults)
    let opts: JsonRunOptions = match serde_json::from_str(opts_json) {
        Ok(o) => o,
        Err(e) => {
            // Invalid optsJson -> return error response
            let budget = BudgetJsonInfo::new("unknown", true);
            let error = EngineJsonError {
                kind: "InvalidInput",
                code: "E_INVALID_INPUT",
                message: format!("Invalid options JSON: {}", e),
                span: None,
                details: serde_json::json!({ "error": e.to_string() }),
            };
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
            return if opts_json.contains("\"pretty\":true") {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    let strict = opts.budget.mode == "strict";
    let budget_info = BudgetJsonInfo::new(&opts.budget.preset, strict);

    // Create engine and explicit eval components (stateless-friendly API)
    let mut engine = crate::eval::Engine::new();
    let mut store = crate::eval::SessionStore::new();
    let env = cas_session_core::env::Environment::new();
    let options = crate::options::EvalOptions::default();
    let mut profile_cache = crate::profile_cache::ProfileCache::new();

    // Parse expression
    let parsed = match cas_parser::parse(expr, &mut engine.simplifier.context) {
        Ok(id) => id,
        Err(e) => {
            let error = EngineJsonError {
                kind: "ParseError",
                code: "E_PARSE",
                message: e.to_string(),
                span: e.span().map(SpanJson::from),
                details: serde_json::Value::Null,
            };
            let resp = EngineJsonResponse {
                schema_version: SCHEMA_VERSION,
                ok: false,
                result: None,
                error: Some(error),
                steps: vec![],
                warnings: vec![],
                assumptions: vec![],
                budget: budget_info,
            };
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    // Build eval request
    let req = crate::eval::EvalRequest {
        raw_input: expr.to_string(),
        parsed,
        kind: cas_session_core::types::EntryKind::Expr(parsed),
        action: crate::eval::EvalAction::Simplify,
        auto_store: false,
    };

    // Evaluate
    let output =
        match engine.eval_with_components(&mut store, &env, &options, &mut profile_cache, req) {
            Ok(o) => o,
            Err(e) => {
                // anyhow::Error - create generic error
                let error = EngineJsonError::simple("InternalError", "E_INTERNAL", e.to_string());
                let resp = EngineJsonResponse {
                    schema_version: SCHEMA_VERSION,
                    ok: false,
                    result: None,
                    error: Some(error),
                    steps: vec![],
                    warnings: vec![],
                    assumptions: vec![],
                    budget: budget_info,
                };
                return if opts.pretty {
                    resp.to_json_pretty()
                } else {
                    resp.to_json()
                };
            }
        };

    // Format result
    let result_str = match &output.result {
        crate::eval::EvalResult::Expr(e) => {
            let clean = crate::engine::strip_all_holds(&mut engine.simplifier.context, *e);
            format!(
                "{}",
                cas_ast::display::DisplayExpr {
                    context: &engine.simplifier.context,
                    id: clean
                }
            )
        }
        crate::eval::EvalResult::Set(v) if !v.is_empty() => {
            let clean = crate::engine::strip_all_holds(&mut engine.simplifier.context, v[0]);
            format!(
                "{}",
                cas_ast::display::DisplayExpr {
                    context: &engine.simplifier.context,
                    id: clean
                }
            )
        }
        crate::eval::EvalResult::Bool(b) => b.to_string(),
        _ => "(no result)".to_string(),
    };

    // Build steps (if requested)
    let steps = if opts.steps {
        output
            .steps
            .iter()
            .map(|s| {
                let before_str = s.global_before.map(|id| {
                    let clean = crate::engine::strip_all_holds(&mut engine.simplifier.context, id);
                    format!(
                        "{}",
                        cas_ast::display::DisplayExpr {
                            context: &engine.simplifier.context,
                            id: clean
                        }
                    )
                });
                let after_str = s.global_after.map(|id| {
                    let clean = crate::engine::strip_all_holds(&mut engine.simplifier.context, id);
                    format!(
                        "{}",
                        cas_ast::display::DisplayExpr {
                            context: &engine.simplifier.context,
                            id: clean
                        }
                    )
                });
                EngineJsonStep {
                    phase: "Simplify".to_string(),
                    rule: s.rule_name.clone(),
                    before: before_str.unwrap_or_default(),
                    after: after_str.unwrap_or_default(),
                    substeps: vec![],
                }
            })
            .collect()
    } else {
        vec![]
    };

    let resp = EngineJsonResponse::ok_with_steps(result_str, steps, budget_info);
    if opts.pretty {
        resp.to_json_pretty()
    } else {
        resp.to_json()
    }
}
