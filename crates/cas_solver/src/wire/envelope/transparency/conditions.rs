use crate::ImplicitCondition;
use cas_api_models::ConditionDto;

use super::super::common::display_expr;

pub(super) fn map_required_conditions(
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
