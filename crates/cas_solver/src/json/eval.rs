use super::mappers::{
    map_domain_warnings_to_engine_warnings, map_solver_assumptions_to_api_records,
};
use super::stateless_eval::evaluate_prepared_stateless_request;
use crate::EvalOptions;
use cas_api_models::{EngineJsonError, EngineJsonResponse};

mod options;
mod prepare;
mod render;
mod request;
mod success;

/// Evaluate an expression and return JSON response.
///
/// This is the **solver-level canonical entry point** for JSON-returning
/// stateless evaluation. Frontends should normally go through
/// `cas_session::evaluate_eval_canonical`.
///
/// # Arguments
/// * `expr` - Expression string to evaluate
/// * `opts_json` - Options JSON string (see `EvalRunOptions`)
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
    let (opts, budget_info, mut engine, prepared) =
        match prepare::prepare_stateless_eval_request(expr, opts_json) {
            Ok(state) => state,
            Err(resp) => return resp,
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
        render::build_engine_wire_steps,
    )
}

#[cfg(test)]
mod tests;
