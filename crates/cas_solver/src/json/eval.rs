use super::mappers::{
    map_domain_warnings_to_engine_warnings, map_solver_assumptions_to_api_records,
};
use super::stateless_eval::evaluate_prepared_stateless_request;
use crate::{Engine, EvalOptions, EvalResult};
use cas_api_models::{
    BudgetJsonInfo, EngineJsonError, EngineJsonResponse, EngineJsonStep, JsonRunOptions, SpanJson,
};
use cas_ast::hold::strip_all_holds;

/// Evaluate an expression and return JSON response.
///
/// This is the **solver-level canonical entry point** for JSON-returning
/// stateless evaluation. Frontends should normally go through
/// `cas_session::evaluate_eval_json_canonical`.
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
            let resp = EngineJsonResponse::invalid_options_json(e.to_string());
            return resp.to_json_with_pretty(JsonRunOptions::requested_pretty(opts_json));
        }
    };

    let strict = opts.budget.mode == "strict";
    let budget_info = BudgetJsonInfo::new(&opts.budget.preset, strict);

    // Create engine
    let mut engine = Engine::new();

    // Parse input into a solver-owned prepared request.
    let prepared =
        match crate::build_eval_json_request_for_input(expr, &mut engine.simplifier.context, false)
        {
            Ok(request) => request,
            Err(e) => {
                let error = EngineJsonError::parse(e, Option::<SpanJson>::None);
                let resp = EngineJsonResponse::err(error, budget_info);
                return resp.to_json_with_pretty(opts.pretty);
            }
        };

    // Evaluate in stateless mode.
    let output_view =
        match evaluate_prepared_stateless_request(&mut engine, EvalOptions::default(), prepared) {
            Ok(view) => view,
            Err(e) => {
                let error = EngineJsonError::from_eval_runtime_error(e.to_string());
                let resp = EngineJsonResponse::err(error, budget_info);
                return resp.to_json_with_pretty(opts.pretty);
            }
        };

    // Format result
    let result_str = match &output_view.result {
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
        EvalResult::SolutionSet(solution_set) => {
            crate::display_solution_set(&engine.simplifier.context, solution_set)
        }
        EvalResult::Bool(b) => b.to_string(),
        _ => "(no result)".to_string(),
    };

    // Build steps (if requested)
    let steps = if opts.steps {
        output_view
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
    resp.warnings = map_domain_warnings_to_engine_warnings(&output_view.domain_warnings);
    resp.assumptions = map_solver_assumptions_to_api_records(&output_view.solver_assumptions);

    resp.to_json_with_pretty(opts.pretty)
}

#[cfg(test)]
mod tests;
