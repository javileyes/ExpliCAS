use cas_api_models::{
    AssumptionRecord, BudgetJsonInfo, EngineJsonResponse, EngineJsonStep, EngineJsonWarning,
    JsonRunOptions,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn build_success_json(
    engine: &mut crate::Engine,
    output_view: &crate::EvalOutputView,
    opts: &JsonRunOptions,
    budget_info: BudgetJsonInfo,
    map_warnings: fn(&[crate::DomainWarning]) -> Vec<EngineJsonWarning>,
    map_assumptions: fn(&[crate::AssumptionRecord]) -> Vec<AssumptionRecord>,
    render_result: fn(&mut cas_ast::Context, &crate::EvalResult) -> String,
    render_steps: fn(&mut cas_ast::Context, &crate::DisplayEvalSteps, bool) -> Vec<EngineJsonStep>,
) -> String {
    let result_str = render_result(&mut engine.simplifier.context, &output_view.result);
    let steps = render_steps(
        &mut engine.simplifier.context,
        &output_view.steps,
        opts.steps,
    );

    let mut resp = EngineJsonResponse::ok_with_steps(result_str, steps, budget_info);
    resp.warnings = map_warnings(&output_view.domain_warnings);
    resp.assumptions = map_assumptions(&output_view.solver_assumptions);
    resp.to_json_with_pretty(opts.pretty)
}
