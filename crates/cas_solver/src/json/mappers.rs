use cas_api_models::{
    AssumptionDto, AssumptionRecord as ApiAssumptionRecord, BlockedHintDto, ConditionDto,
    EngineJsonWarning,
};
use cas_formatter::DisplayExpr;

pub(super) fn map_domain_warnings_to_engine_warnings(
    warnings: &[cas_engine::eval::DomainWarning],
) -> Vec<EngineJsonWarning> {
    warnings
        .iter()
        .map(|w| EngineJsonWarning {
            kind: "domain_assumption".to_string(),
            message: format!("{} (rule: {})", w.message, w.rule_name),
        })
        .collect()
}

pub(super) fn map_solver_assumptions_to_api_records(
    assumptions: &[cas_engine::assumptions::AssumptionRecord],
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

pub(super) fn map_required_conditions(
    required_conditions: &[cas_engine::implicit_domain::ImplicitCondition],
    ctx: &cas_ast::Context,
) -> Vec<ConditionDto> {
    use cas_engine::implicit_domain::ImplicitCondition;

    required_conditions
        .iter()
        .map(|cond| {
            let (kind, expr_id) = match cond {
                ImplicitCondition::NonNegative(e) => ("NonNegative", *e),
                ImplicitCondition::Positive(e) => ("Positive", *e),
                ImplicitCondition::NonZero(e) => ("NonZero", *e),
            };
            let expr_display = DisplayExpr {
                context: ctx,
                id: expr_id,
            }
            .to_string();
            ConditionDto {
                kind: kind.to_string(),
                display: cond.display(ctx),
                expr_display: expr_display.clone(),
                expr_canonical: expr_display,
            }
        })
        .collect()
}

pub(super) fn map_assumptions_used(
    assumptions: &[cas_engine::assumptions::AssumptionRecord],
    warnings: &[cas_engine::eval::DomainWarning],
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

pub(super) fn map_blocked_hints(
    blocked_hints: &[cas_engine::domain::BlockedHint],
) -> Vec<BlockedHintDto> {
    blocked_hints
        .iter()
        .map(|h| BlockedHintDto {
            rule: h.rule.clone(),
            requires: vec![h.key.condition_display().to_string()],
            tip: h.suggestion.to_string(),
        })
        .collect()
}
