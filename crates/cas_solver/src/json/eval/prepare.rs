use crate::Engine;
use cas_api_models::{BudgetJsonInfo, EngineJsonResponse, JsonRunOptions};

pub(super) type PreparedEvalJsonState = (
    JsonRunOptions,
    BudgetJsonInfo,
    Engine,
    crate::EvalJsonPreparedRequest,
);

pub(super) fn prepare_eval_json_request(
    expr: &str,
    opts_json: &str,
) -> Result<PreparedEvalJsonState, String> {
    let opts = super::options::parse_json_run_options(opts_json)?;
    let budget_info = super::options::build_budget_info(&opts);

    let mut engine = Engine::new();
    let prepared = match super::request::build_prepared_eval_json_request(expr, &mut engine) {
        Ok(request) => request,
        Err(error) => {
            let resp = EngineJsonResponse::err(error, budget_info.clone());
            return Err(resp.to_json_with_pretty(opts.pretty));
        }
    };

    Ok((opts, budget_info, engine, prepared))
}
