use crate::{AssumptionRecord, DomainWarning};
use cas_api_models::AssumptionDto;

pub(super) fn map_assumptions_used(
    assumptions: &[AssumptionRecord],
    warnings: &[DomainWarning],
) -> Vec<AssumptionDto> {
    let mut out: Vec<AssumptionDto> = assumptions
        .iter()
        .map(|a| AssumptionDto {
            kind: a.kind.clone(),
            display: a.message.clone(),
            expr_canonical: a.expr.clone(),
            rule: "solver".to_string(),
        })
        .collect();
    out.extend(warnings.iter().map(|w| AssumptionDto {
        kind: "domain_warning".to_string(),
        display: w.message.clone(),
        expr_canonical: String::new(),
        rule: w.rule_name.clone(),
    }));
    out
}
