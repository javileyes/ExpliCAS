use super::mappers::{
    map_domain_warnings_to_engine_warnings, map_solver_assumptions_to_api_records,
};
use super::stateless_eval::evaluate_prepared_stateless_request;
use crate::{Engine, EvalOptions};
use cas_api_models::{EngineJsonError, EngineJsonResponse};

mod options;
mod render;
mod request;
mod success;

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
    let opts = match options::parse_json_run_options(opts_json) {
        Ok(opts) => opts,
        Err(resp) => return resp,
    };
    let budget_info = options::build_budget_info(&opts);

    let mut engine = Engine::new();

    let prepared = match request::build_prepared_eval_json_request(expr, &mut engine) {
        Ok(request) => request,
        Err(error) => {
            let resp = EngineJsonResponse::err(error, budget_info);
            return resp.to_json_with_pretty(opts.pretty);
        }
    };

    let output_view =
        match evaluate_prepared_stateless_request(&mut engine, EvalOptions::default(), prepared) {
            Ok(view) => view,
            Err(e) => {
                let error = EngineJsonError::from_eval_runtime_error(e.to_string());
                let resp = EngineJsonResponse::err(error, budget_info);
                return resp.to_json_with_pretty(opts.pretty);
            }
        };

    success::build_success_json(
        &mut engine,
        &output_view,
        &opts,
        budget_info,
        map_domain_warnings_to_engine_warnings,
        map_solver_assumptions_to_api_records,
        render::render_eval_result,
        render::build_engine_json_steps,
    )
}

#[cfg(test)]
mod tests;
