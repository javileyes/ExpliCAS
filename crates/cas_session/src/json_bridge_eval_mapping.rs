use cas_api_models::{AssumptionRecord as ApiAssumptionRecord, EngineJsonWarning};

pub(crate) fn map_domain_warnings_to_engine_warnings(
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

pub(crate) fn map_solver_assumptions_to_api_records(
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
