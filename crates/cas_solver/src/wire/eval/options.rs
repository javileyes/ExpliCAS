use cas_api_models::{BudgetWireInfo, EngineWireResponse, EvalRunOptions};

pub(super) fn parse_eval_run_options(opts_json: &str) -> Result<EvalRunOptions, String> {
    match serde_json::from_str(opts_json) {
        Ok(opts) => Ok(opts),
        Err(e) => {
            let resp = EngineWireResponse::invalid_options_json(e.to_string());
            Err(resp.to_json_with_pretty(EvalRunOptions::requested_pretty(opts_json)))
        }
    }
}

pub(super) fn build_budget_info(opts: &EvalRunOptions) -> BudgetWireInfo {
    let strict = opts.budget.mode == "strict";
    BudgetWireInfo::new(&opts.budget.preset, strict)
}
