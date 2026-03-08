use cas_api_models::{BudgetJsonInfo, EngineJsonResponse, JsonRunOptions};

pub(super) fn parse_json_run_options(opts_json: &str) -> Result<JsonRunOptions, String> {
    match serde_json::from_str(opts_json) {
        Ok(opts) => Ok(opts),
        Err(e) => {
            let resp = EngineJsonResponse::invalid_options_json(e.to_string());
            Err(resp.to_json_with_pretty(JsonRunOptions::requested_pretty(opts_json)))
        }
    }
}

pub(super) fn build_budget_info(opts: &JsonRunOptions) -> BudgetJsonInfo {
    let strict = opts.budget.mode == "strict";
    BudgetJsonInfo::new(&opts.budget.preset, strict)
}
