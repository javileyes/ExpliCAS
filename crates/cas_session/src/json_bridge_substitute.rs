//! Canonical stateless substitute-json bridge for external frontends.

use crate::json_bridge_substitute_eval::eval_substitute_impl;
use cas_api_models::{
    EngineJsonSubstep, SubstituteJsonOptions, SubstituteJsonResponse, SubstituteOptionsInner,
    SubstituteOptionsJson, SubstituteRequestEcho,
};

/// Stateless canonical substitute JSON entry point.
pub fn evaluate_substitute_json_canonical(
    expr: &str,
    target: &str,
    replacement: &str,
    opts_json: Option<&str>,
) -> String {
    let opts = SubstituteJsonOptions::parse_optional_json(opts_json);
    let request = SubstituteRequestEcho {
        expr: expr.to_string(),
        target: target.to_string(),
        with_expr: replacement.to_string(),
    };
    let options = SubstituteOptionsJson {
        substitute: SubstituteOptionsInner {
            mode: opts.mode.clone(),
            steps: opts.steps,
        },
    };

    let eval = match eval_substitute_impl(expr, target, replacement, &opts.mode, opts.steps) {
        Ok(eval) => eval,
        Err(issue) => {
            let resp = SubstituteJsonResponse::err(
                issue.to_json_error(),
                request.clone(),
                options.clone(),
            );
            return resp.to_json_with_pretty(opts.pretty);
        }
    };

    let json_steps: Vec<EngineJsonSubstep> = eval
        .steps
        .into_iter()
        .map(|step| EngineJsonSubstep {
            rule: step.rule,
            before: step.before,
            after: step.after,
            note: step.note,
        })
        .collect();
    let resp = SubstituteJsonResponse::ok(eval.result, request, options, json_steps);
    resp.to_json_with_pretty(opts.pretty)
}
