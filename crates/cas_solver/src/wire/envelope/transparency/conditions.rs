use crate::ImplicitCondition;
use cas_api_models::{AssumptionDto, ConditionDto};
use cas_solver_core::domain_normalization::normalize_and_dedupe_conditions;

use super::super::common::display_expr;
use crate::eval_output_condition_filter::AssumedConditionFilter;

pub(super) fn map_required_conditions(
    required_conditions: &[ImplicitCondition],
    ctx: &cas_ast::Context,
    assumptions_used: &[AssumptionDto],
) -> Vec<ConditionDto> {
    let mut normalized_ctx = ctx.clone();
    let assumed_filter = AssumedConditionFilter::from_assumptions(assumptions_used);

    normalize_and_dedupe_conditions(&mut normalized_ctx, required_conditions)
        .iter()
        .filter(|cond| !assumed_filter.covers_required_condition(&normalized_ctx, cond))
        .map(|cond| {
            let (kind, expr_id) = match cond {
                ImplicitCondition::NonNegative(e) => ("NonNegative", *e),
                ImplicitCondition::Positive(e) => ("Positive", *e),
                ImplicitCondition::NonZero(e) => ("NonZero", *e),
            };
            let expr_display = display_expr(&normalized_ctx, expr_id);
            ConditionDto {
                kind: kind.to_string(),
                display: cond.display(&normalized_ctx),
                expr_display: expr_display.clone(),
                expr_canonical: expr_display,
            }
        })
        .collect()
}
