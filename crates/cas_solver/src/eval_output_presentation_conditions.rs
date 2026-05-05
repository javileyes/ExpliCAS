use cas_api_models::{AssumptionDto, RequiredConditionWire, WarningWire};
use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_solver_core::domain_normalization::normalize_and_dedupe_conditions;
use std::collections::HashSet;

pub(crate) fn collect_output_warnings(
    domain_warnings: &[crate::DomainWarning],
) -> Vec<WarningWire> {
    domain_warnings
        .iter()
        .map(|w| WarningWire {
            rule: w.rule_name.clone(),
            assumption: w.message.clone(),
        })
        .collect()
}

pub(crate) fn collect_output_required_conditions(
    required_conditions: &[crate::ImplicitCondition],
    ctx: &mut Context,
    assumed_display: &HashSet<String>,
) -> Vec<RequiredConditionWire> {
    normalize_and_dedupe_conditions(ctx, required_conditions)
        .iter()
        .filter(|cond| !assumed_display.contains(&cond.display(ctx)))
        .map(|cond| {
            let (kind, expr_id) = match cond {
                crate::ImplicitCondition::NonNegative(e) => ("NonNegative", *e),
                crate::ImplicitCondition::Positive(e) => ("Positive", *e),
                crate::ImplicitCondition::NonZero(e) => ("NonZero", *e),
            };
            let expr_str = DisplayExpr {
                context: ctx,
                id: expr_id,
            }
            .to_string();
            RequiredConditionWire {
                kind: kind.to_string(),
                expr_display: expr_str.clone(),
                expr_canonical: expr_str,
            }
        })
        .collect()
}

pub(crate) fn collect_output_required_display(
    required_conditions: &[crate::ImplicitCondition],
    ctx: &mut Context,
    assumed_display: &HashSet<String>,
) -> Vec<String> {
    normalize_and_dedupe_conditions(ctx, required_conditions)
        .iter()
        .map(|cond| cond.display(ctx))
        .filter(|display| !assumed_display.contains(display))
        .collect()
}

pub(crate) fn collect_output_assumptions_used(steps: &[crate::Step]) -> Vec<AssumptionDto> {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut assumptions = Vec::new();

    for step in steps {
        for event in step.assumption_events() {
            if !matches!(
                event.kind,
                cas_solver_core::assumption_model::AssumptionKind::HeuristicAssumption
            ) {
                continue;
            }

            let rule = step.rule_name.to_string();
            let display = event.message.clone();
            if !seen.insert((rule.clone(), display.clone())) {
                continue;
            }

            assumptions.push(AssumptionDto {
                kind: event.key.kind().to_string(),
                display,
                expr_canonical: event.expr_display.clone(),
                rule,
            });
        }
    }

    assumptions
}

pub(crate) fn collect_output_assumption_display_set(
    assumptions: &[AssumptionDto],
) -> HashSet<String> {
    assumptions
        .iter()
        .map(|assumption| assumption.display.clone())
        .collect()
}
