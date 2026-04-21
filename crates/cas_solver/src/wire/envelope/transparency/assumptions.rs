use crate::{AssumptionRecord, DomainWarning};
use cas_api_models::AssumptionDto;
use std::collections::HashSet;

pub(super) fn map_assumptions_used(
    assumptions: &[AssumptionRecord],
    warnings: &[DomainWarning],
    steps: &[crate::Step],
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
    let mut seen: HashSet<(String, String)> = out
        .iter()
        .map(|assumption| (assumption.rule.clone(), assumption.display.clone()))
        .collect();
    for step in steps {
        for event in step.assumption_events() {
            if !matches!(
                event.kind,
                cas_solver_core::assumption_model::AssumptionKind::HeuristicAssumption
            ) {
                continue;
            }
            let key = (step.rule_name.to_string(), event.message.clone());
            if !seen.insert(key.clone()) {
                continue;
            }
            out.push(AssumptionDto {
                kind: "domain_warning".to_string(),
                display: key.1,
                expr_canonical: event.expr_display.clone(),
                rule: key.0,
            });
        }
    }
    out
}
