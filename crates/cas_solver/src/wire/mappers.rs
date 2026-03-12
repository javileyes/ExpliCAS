use crate::{AssumptionRecord, DomainWarning};
use cas_api_models::{AssumptionRecord as ApiAssumptionRecord, EngineWireWarning};

pub(super) fn map_domain_warnings_to_engine_warnings(
    warnings: &[DomainWarning],
) -> Vec<EngineWireWarning> {
    warnings
        .iter()
        .map(|w| EngineWireWarning {
            kind: "domain_assumption".into(),
            message: format!("{} (rule: {})", w.message, w.rule_name),
        })
        .collect()
}

pub(super) fn map_solver_assumptions_to_api_records(
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
