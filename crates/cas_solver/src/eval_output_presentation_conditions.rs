use cas_api_models::{AssumptionDto, RequiredConditionWire, WarningWire};
use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_solver_core::domain_normalization::normalize_and_dedupe_conditions;
use std::collections::HashSet;

use crate::eval_output_condition_filter::AssumedConditionFilter;

pub(crate) fn collect_output_warnings(
    domain_warnings: &[crate::DomainWarning],
    assumptions_used: &[AssumptionDto],
) -> Vec<WarningWire> {
    let assumed_display: HashSet<&str> = assumptions_used
        .iter()
        .map(|assumption| assumption.display.as_str())
        .collect();
    domain_warnings
        .iter()
        .filter(|w| !assumed_display.contains(w.message.as_str()))
        .map(|w| WarningWire {
            rule: w.rule_name.clone(),
            assumption: w.message.clone(),
        })
        .collect()
}

pub(crate) fn collect_output_required_conditions(
    required_conditions: &[crate::ImplicitCondition],
    ctx: &mut Context,
    assumptions_used: &[AssumptionDto],
) -> Vec<RequiredConditionWire> {
    let assumed_filter = AssumedConditionFilter::from_assumptions(assumptions_used);
    normalize_and_dedupe_conditions(ctx, required_conditions)
        .iter()
        .filter(|cond| !assumed_filter.covers_required_condition(ctx, cond))
        .map(|cond| {
            let (kind, expr_id) = match cond {
                crate::ImplicitCondition::NonNegative(e) => ("NonNegative", *e),
                crate::ImplicitCondition::Positive(e) => ("Positive", *e),
                crate::ImplicitCondition::NonZero(e) => ("NonZero", *e),
            };
            let expr_str = expr_display(ctx, expr_id);
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
    assumptions_used: &[AssumptionDto],
) -> Vec<String> {
    let assumed_filter = AssumedConditionFilter::from_assumptions(assumptions_used);
    normalize_and_dedupe_conditions(ctx, required_conditions)
        .iter()
        .filter(|cond| !assumed_filter.covers_required_condition(ctx, cond))
        .map(|cond| cond.display(ctx))
        .collect()
}

pub(crate) fn collect_output_assumptions_used(steps: &[crate::Step]) -> Vec<AssumptionDto> {
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    let mut assumptions = Vec::new();

    for step in steps {
        for event in step.assumption_events() {
            if !matches!(
                event.kind,
                cas_solver_core::assumption_model::AssumptionKind::HeuristicAssumption
            ) {
                continue;
            }

            let kind = event.key.kind().to_string();
            let rule = step.rule_name.to_string();
            let display = event.message.clone();
            let expr_canonical = event.expr_display.clone();
            if !seen.insert((kind.clone(), expr_canonical.clone(), display.clone())) {
                continue;
            }

            assumptions.push(AssumptionDto {
                kind,
                display,
                expr_canonical,
                rule,
            });
        }
    }

    assumptions
}

fn expr_display(ctx: &Context, expr_id: cas_ast::ExprId) -> String {
    DisplayExpr {
        context: ctx,
        id: expr_id,
    }
    .to_string()
}
