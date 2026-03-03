//! Canonical stateless eval-json bridge for external frontends.

use cas_api_models::{
    AssumptionRecord as ApiAssumptionRecord, BudgetJsonInfo, EngineJsonError as ApiEngineJsonError,
    EngineJsonResponse, EngineJsonStep, EngineJsonWarning, JsonRunOptions, SpanJson as ApiSpanJson,
};
use cas_ast::hold::strip_all_holds;

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

    let result = match &output_view.result {
        cas_solver::EvalResult::Expr(expr_id) => {
            let clean = strip_all_holds(&mut engine.simplifier.context, *expr_id);
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &engine.simplifier.context,
                    id: clean
                }
            )
        }
        cas_solver::EvalResult::Set(values) if !values.is_empty() => {
            let clean = strip_all_holds(&mut engine.simplifier.context, values[0]);
            format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &engine.simplifier.context,
                    id: clean
                }
            )
        }
        cas_solver::EvalResult::Bool(flag) => flag.to_string(),
        _ => "(no result)".to_string(),
    };

    let steps = if opts.steps {
        output_view
            .steps
            .iter()
            .map(|step| {
                let before = step.global_before.map(|id| {
                    let clean = strip_all_holds(&mut engine.simplifier.context, id);
                    format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: &engine.simplifier.context,
                            id: clean
                        }
                    )
                });
                let after = step.global_after.map(|id| {
                    let clean = strip_all_holds(&mut engine.simplifier.context, id);
                    format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: &engine.simplifier.context,
                            id: clean
                        }
                    )
                });
                EngineJsonStep {
                    phase: "Simplify".to_string(),
                    rule: step.rule_name.clone(),
                    before: before.unwrap_or_default(),
                    after: after.unwrap_or_default(),
                    substeps: vec![],
                }
            })
            .collect()
    } else {
        vec![]
    };

    let mut resp = EngineJsonResponse::ok_with_steps(result, steps, budget_info);
    resp.warnings = map_domain_warnings_to_engine_warnings(&output_view.domain_warnings);
    resp.assumptions = map_solver_assumptions_to_api_records(&output_view.solver_assumptions);
    resp.to_json_with_pretty(opts.pretty)
}

fn map_domain_warnings_to_engine_warnings(
    warnings: &[cas_solver::DomainWarning],
) -> Vec<EngineJsonWarning> {
    warnings
        .iter()
        .map(|warning| EngineJsonWarning {
            kind: "domain_assumption".to_string(),
            message: format!("{} (rule: {})", warning.message, warning.rule_name),
        })
        .collect()
}

fn map_solver_assumptions_to_api_records(
    assumptions: &[cas_solver::AssumptionRecord],
) -> Vec<ApiAssumptionRecord> {
    assumptions
        .iter()
        .map(|assumption| ApiAssumptionRecord {
            kind: assumption.kind.clone(),
            expr: assumption.expr.clone(),
            message: assumption.message.clone(),
            count: assumption.count,
        })
        .collect()
}
