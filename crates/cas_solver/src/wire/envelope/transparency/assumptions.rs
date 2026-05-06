use crate::{AssumptionRecord, DomainWarning};
use cas_api_models::AssumptionDto;
use std::collections::HashSet;

pub(super) fn map_assumptions_used(
    assumptions: &[AssumptionRecord],
    warnings: &[DomainWarning],
    steps: &[crate::Step],
) -> Vec<AssumptionDto> {
    let mut out = Vec::new();
    let mut seen: HashSet<(String, String, String)> = HashSet::new();

    for assumption in assumptions {
        let kind = assumption.kind.clone();
        let expr_canonical = assumption.expr.clone();
        let display = assumption.message.clone();
        if !seen.insert((kind.clone(), expr_canonical.clone(), display.clone())) {
            continue;
        }
        out.push(AssumptionDto {
            kind,
            display,
            expr_canonical,
            rule: "solver".to_string(),
        });
    }

    for step in steps {
        for event in step.assumption_events() {
            if !matches!(
                event.kind,
                cas_solver_core::assumption_model::AssumptionKind::HeuristicAssumption
            ) {
                continue;
            }

            let kind = event.key.kind().to_string();
            let display = event.message.clone();
            let expr_canonical = event.expr_display.clone();
            if !seen.insert((kind.clone(), expr_canonical.clone(), display.clone())) {
                continue;
            }
            out.push(AssumptionDto {
                kind,
                display,
                expr_canonical,
                rule: step.rule_name.to_string(),
            });
        }
    }

    let assumed_displays: HashSet<String> = out
        .iter()
        .map(|assumption| assumption.display.clone())
        .collect();
    out.extend(
        warnings
            .iter()
            .filter(|warning| !assumed_displays.contains(&warning.message))
            .map(|warning| AssumptionDto {
                kind: "domain_warning".to_string(),
                display: warning.message.clone(),
                expr_canonical: String::new(),
                rule: warning.rule_name.clone(),
            }),
    );

    out
}
