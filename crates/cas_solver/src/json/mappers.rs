use crate::{AssumptionRecord, DomainWarning};
use cas_api_models::{AssumptionRecord as ApiAssumptionRecord, EngineJsonWarning};

pub fn map_domain_warnings_to_engine_warnings(
    warnings: &[DomainWarning],
) -> Vec<EngineJsonWarning> {
    warnings
        .iter()
        .map(|w| EngineJsonWarning {
            kind: "domain_assumption".to_string(),
            message: format!("{} (rule: {})", w.message, w.rule_name),
        })
        .collect()
}

pub fn map_solver_assumptions_to_api_records(
    assumptions: &[AssumptionRecord],
) -> Vec<ApiAssumptionRecord> {
    assumptions
        .iter()
        .map(|a| ApiAssumptionRecord {
            kind: a.kind.clone(),
            expr: a.expr.clone(),
            message: a.message.clone(),
            count: a.count,
        })
        .collect()
}
