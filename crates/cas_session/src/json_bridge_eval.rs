//! Canonical stateless eval-json bridge for external frontends.

use crate::json_bridge_eval_mapping::{
    map_domain_warnings_to_engine_warnings, map_solver_assumptions_to_api_records,
};
use crate::json_bridge_eval_render::{render_eval_result, render_eval_steps};
use cas_api_models::{
    BudgetJsonInfo, EngineJsonError as ApiEngineJsonError, EngineJsonResponse, JsonRunOptions,
    SpanJson as ApiSpanJson,
};

/// Stateless canonical eval JSON entry point.
pub fn evaluate_eval_json_canonical(expr: &str, opts_json: &str) -> String {
    let opts: JsonRunOptions = match serde_json::from_str(opts_json) {
        Ok(o) => o,
        Err(e) => {
            let resp = EngineJsonResponse::invalid_options_json(e.to_string());
            return resp.to_json_with_pretty(JsonRunOptions::requested_pretty(opts_json));
        }
    };

    let strict = opts.budget.mode == "strict";
    let budget_info = BudgetJsonInfo::new(&opts.budget.preset, strict);

    let mut engine = cas_solver::Engine::new();
    let parsed = match cas_parser::parse(expr, &mut engine.simplifier.context) {
        Ok(id) => id,
        Err(e) => {
            let error = ApiEngineJsonError::parse(
                e.to_string(),
                e.span().map(|s| ApiSpanJson {
                    start: s.start,
                    end: s.end,
                }),
            );
            let resp = EngineJsonResponse::err(error, budget_info);
            return resp.to_json_with_pretty(opts.pretty);
        }
    };

    let req = cas_solver::EvalRequest {
        raw_input: expr.to_string(),
        parsed,
        action: cas_solver::EvalAction::Simplify,
        auto_store: false,
    };

    let output = match engine.eval_stateless(cas_solver::EvalOptions::default(), req) {
        Ok(output) => output,
        Err(e) => {
            let error = ApiEngineJsonError::from_eval_runtime_error(e.to_string());
            let resp = EngineJsonResponse::err(error, budget_info);
            return resp.to_json_with_pretty(opts.pretty);
        }
    };
    let output_view = cas_solver::eval_output_view(&output);

    let result = render_eval_result(&mut engine.simplifier.context, &output_view.result);

    let steps = if opts.steps {
        render_eval_steps(&mut engine.simplifier.context, &output_view.steps)
    } else {
        vec![]
    };

    let mut resp = EngineJsonResponse::ok_with_steps(result, steps, budget_info);
    resp.warnings = map_domain_warnings_to_engine_warnings(&output_view.domain_warnings);
    resp.assumptions = map_solver_assumptions_to_api_records(&output_view.solver_assumptions);
    resp.to_json_with_pretty(opts.pretty)
}
