use super::mappers::{
    map_domain_warnings_to_engine_warnings, map_solver_assumptions_to_api_records,
};
use super::stateless_eval::evaluate_prepared_stateless_request;
use crate::EvalOptions;
use cas_api_models::{EngineWireError, EngineWireResponse};

mod options;
mod prepare;
mod render;
mod request;
mod success;

/// Evaluate an expression and return wire response.
///
/// This is the **solver-level direct entry point** for wire-returning
/// stateless evaluation. Frontends should normally go through
/// Stateless wire entrypoint for eval-style callers.
///
/// # Arguments
/// * `expr` - Expression string to evaluate
/// * `opts_json` - Options JSON string (see `EvalRunOptions`)
///
/// # Returns
/// Wire payload string with `EngineWireResponse` (schema v1).
/// Always returns valid JSON, even on errors.
///
/// # Example
/// ```
/// use cas_solver::eval_str_to_wire;
///
/// let wire = eval_str_to_wire("x + x", r#"{"budget":{"preset":"cli"}}"#);
/// assert!(wire.contains("\"ok\":true"));
/// ```
pub fn eval_str_to_wire(expr: &str, opts_json: &str) -> String {
    let (opts, budget_info, mut engine, prepared) =
        match prepare::prepare_stateless_eval_request(expr, opts_json) {
            Ok(state) => state,
            Err(resp) => return resp,
        };

    let output_view =
        match evaluate_prepared_stateless_request(&mut engine, EvalOptions::default(), prepared) {
            Ok(view) => view,
            Err(e) => {
                let error = EngineWireError::from_eval_runtime_error(e.to_string());
                let resp = EngineWireResponse::err(error, budget_info);
                return resp.to_json_with_pretty(opts.pretty);
            }
        };

    success::build_success_wire(
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
