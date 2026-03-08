//! Assumption classification against implicit-domain requirements.

use crate::assumption_model::{
    assumption_condition_kind, classify_assumption_with_condition, AssumptionConditionKind,
    AssumptionEvent, AssumptionKind,
};
use crate::domain_condition::ImplicitCondition;
use crate::domain_context::DomainContext;
use cas_ast::Context;

/// Convert an assumption event into an implicit condition, when applicable.
pub fn assumption_to_condition(event: &AssumptionEvent) -> Option<ImplicitCondition> {
    assumption_condition_kind(event).map(|(kind, expr_id)| match kind {
        AssumptionConditionKind::NonZero => ImplicitCondition::NonZero(expr_id),
        AssumptionConditionKind::Positive => ImplicitCondition::Positive(expr_id),
        AssumptionConditionKind::NonNegative => ImplicitCondition::NonNegative(expr_id),
    })
}

/// Classify a single assumption event against known global/introduced requires.
pub fn classify_assumption(
    ctx: &Context,
    dc: &DomainContext,
    event: &AssumptionEvent,
) -> (AssumptionKind, Option<ImplicitCondition>) {
    classify_assumption_with_condition(event, assumption_to_condition(event), |cond| {
        dc.is_condition_implied(ctx, cond)
    })
}

/// Reclassify a batch of events in place and accumulate newly introduced conditions.
pub fn classify_assumptions_in_place(
    ctx: &Context,
    dc: &mut DomainContext,
    events: &mut [AssumptionEvent],
) {
    for event in events.iter_mut() {
        let (new_kind, new_cond) = classify_assumption(ctx, dc, event);
        event.kind = new_kind;
        if let Some(cond) = new_cond {
            dc.add_introduced(cond);
        }
    }
}
