use cas_api_models::{RequiredConditionWire, WarningWire};
use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_solver_core::domain_normalization::normalize_and_dedupe_conditions;

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
) -> Vec<RequiredConditionWire> {
    normalize_and_dedupe_conditions(ctx, required_conditions)
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
) -> Vec<String> {
    normalize_and_dedupe_conditions(ctx, required_conditions)
        .iter()
        .map(|cond| cond.display(ctx))
        .collect()
}
