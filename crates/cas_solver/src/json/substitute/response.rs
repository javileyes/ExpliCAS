use cas_api_models::{
    EngineJsonSubstep, SubstituteJsonOptions, SubstituteJsonResponse, SubstituteOptionsInner,
    SubstituteOptionsJson, SubstituteRequestEcho,
};

use super::eval::eval_substitute_impl;

pub fn substitute_str_to_json_impl(
    expr_str: &str,
    target_str: &str,
    with_str: &str,
    opts: SubstituteJsonOptions,
) -> String {
    let request = SubstituteRequestEcho {
        expr: expr_str.to_string(),
        target: target_str.to_string(),
        with_expr: with_str.to_string(),
    };

    let options = SubstituteOptionsJson {
        substitute: SubstituteOptionsInner {
            mode: opts.mode.clone(),
            steps: opts.steps,
        },
    };

    let eval = match eval_substitute_impl(expr_str, target_str, with_str, &opts.mode, opts.steps) {
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
        .map(|s| EngineJsonSubstep {
            rule: s.rule,
            before: s.before,
            after: s.after,
            note: s.note,
        })
        .collect();

    let resp = SubstituteJsonResponse::ok(eval.result, request, options, json_steps);
    resp.to_json_with_pretty(opts.pretty)
}
