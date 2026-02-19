use super::mappers::{
    map_domain_warnings_to_engine_warnings, map_solver_assumptions_to_api_records,
};
use cas_api_models::{
    BudgetJsonInfo, EngineJsonError, EngineJsonResponse, EngineJsonStep, JsonRunOptions, SpanJson,
};
use cas_engine::{strip_all_holds, Engine, EvalAction, EvalRequest, EvalResult};

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
/// use cas_solver::eval_str_to_json;
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
            let error = EngineJsonError::invalid_input(
                format!("Invalid options JSON: {}", e),
                serde_json::json!({ "error": e.to_string() }),
            );
            let resp = EngineJsonResponse::err(error, budget);
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
    let mut engine = Engine::new();
    let mut session = cas_session::SessionState::new();

    // Parse expression
    let parsed = match cas_parser::parse(expr, &mut engine.simplifier.context) {
        Ok(id) => id,
        Err(e) => {
            let error = EngineJsonError::parse(
                e.to_string(),
                e.span().map(|s| SpanJson {
                    start: s.start,
                    end: s.end,
                }),
            );
            let resp = EngineJsonResponse::err(error, budget_info);
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    // Build eval request
    let req = EvalRequest {
        raw_input: expr.to_string(),
        parsed,
        action: EvalAction::Simplify,
        auto_store: false,
    };

    // Evaluate
    let output = match engine.eval(&mut session, req) {
        Ok(o) => o,
        Err(e) => {
            // anyhow::Error - create generic error
            let error = EngineJsonError::simple("InternalError", "E_INTERNAL", e.to_string());
            let resp = EngineJsonResponse::err(error, budget_info);
            return if opts.pretty {
                resp.to_json_pretty()
            } else {
                resp.to_json()
            };
        }
    };

    // Format result
    let result_str = match &output.result {
        EvalResult::Expr(e) => {
            let clean = strip_all_holds(&mut engine.simplifier.context, *e);
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &engine.simplifier.context,
                    id: clean
                }
            )
        }
        EvalResult::Set(v) if !v.is_empty() => {
            let clean = strip_all_holds(&mut engine.simplifier.context, v[0]);
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &engine.simplifier.context,
                    id: clean
                }
            )
        }
        EvalResult::Bool(b) => b.to_string(),
        _ => "(no result)".to_string(),
    };

    // Build steps (if requested)
    let steps = if opts.steps {
        output
            .steps
            .iter()
            .map(|s| {
                let before_str = s.global_before.map(|id| {
                    let clean = strip_all_holds(&mut engine.simplifier.context, id);
                    format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: &engine.simplifier.context,
                            id: clean
                        }
                    )
                });
                let after_str = s.global_after.map(|id| {
                    let clean = strip_all_holds(&mut engine.simplifier.context, id);
                    format!(
                        "{}",
                        cas_formatter::DisplayExpr {
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

    let mut resp = EngineJsonResponse::ok_with_steps(result_str, steps, budget_info);
    resp.warnings = map_domain_warnings_to_engine_warnings(&output.domain_warnings);
    resp.assumptions = map_solver_assumptions_to_api_records(&output.solver_assumptions);

    if opts.pretty {
        resp.to_json_pretty()
    } else {
        resp.to_json()
    }
}
