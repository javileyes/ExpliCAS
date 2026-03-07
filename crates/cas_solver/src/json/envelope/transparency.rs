use crate::{AssumptionRecord, BlockedHint, DomainWarning, ImplicitCondition};
use cas_api_models::{AssumptionDto, BlockedHintDto, ConditionDto, TransparencyDto};

use super::common::display_expr;

fn map_required_conditions(
    required_conditions: &[ImplicitCondition],
    ctx: &cas_ast::Context,
) -> Vec<ConditionDto> {
    required_conditions
        .iter()
        .map(|cond| {
            let (kind, expr_id) = match cond {
                ImplicitCondition::NonNegative(e) => ("NonNegative", *e),
                ImplicitCondition::Positive(e) => ("Positive", *e),
                ImplicitCondition::NonZero(e) => ("NonZero", *e),
            };
            let expr_display = display_expr(ctx, expr_id);
            ConditionDto {
                kind: kind.to_string(),
                display: cond.display(ctx),
                expr_display: expr_display.clone(),
                expr_canonical: expr_display,
            }
        })
        .collect()
}

fn map_assumptions_used(
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

fn map_blocked_hints(blocked_hints: &[BlockedHint]) -> Vec<BlockedHintDto> {
    blocked_hints
        .iter()
        .map(|h| BlockedHintDto {
            rule: h.rule.clone(),
            requires: vec![h.key.condition_display().to_string()],
            tip: h.suggestion.to_string(),
        })
        .collect()
}

pub fn build_transparency(
    required_conditions: &[ImplicitCondition],
    solver_assumptions: &[AssumptionRecord],
    domain_warnings: &[DomainWarning],
    blocked_hints: &[BlockedHint],
    ctx: &cas_ast::Context,
) -> TransparencyDto {
    TransparencyDto {
        required_conditions: map_required_conditions(required_conditions, ctx),
        assumptions_used: map_assumptions_used(solver_assumptions, domain_warnings),
        blocked_hints: map_blocked_hints(blocked_hints),
    }
}
