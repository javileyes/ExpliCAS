use cas_api_models::{RequiredConditionJson, WarningJson};
use cas_ast::Context;
use cas_formatter::DisplayExpr;

pub(crate) fn collect_warnings_eval_json(
    domain_warnings: &[crate::DomainWarning],
) -> Vec<WarningJson> {
    domain_warnings
        .iter()
        .map(|w| WarningJson {
            rule: w.rule_name.clone(),
            assumption: w.message.clone(),
        })
        .collect()
}

pub(crate) fn collect_required_conditions_eval_json(
    required_conditions: &[crate::ImplicitCondition],
    ctx: &Context,
) -> Vec<RequiredConditionJson> {
    required_conditions
        .iter()
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
            RequiredConditionJson {
                kind: kind.to_string(),
                expr_display: expr_str.clone(),
                expr_canonical: expr_str,
            }
        })
        .collect()
}

pub(crate) fn collect_required_display_eval_json(
    required_conditions: &[crate::ImplicitCondition],
    ctx: &Context,
) -> Vec<String> {
    required_conditions
        .iter()
        .map(|cond| cond.display(ctx))
        .collect()
}
