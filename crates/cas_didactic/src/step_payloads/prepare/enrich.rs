use cas_ast::{Context, ExprId};
use cas_solver::Step;

pub(super) fn enrich_step_payloads(
    context: &Context,
    original_expr: ExprId,
    filtered: Vec<Step>,
    enrich_steps: fn(&Context, ExprId, Vec<Step>) -> Vec<crate::didactic::EnrichedStep>,
) -> Vec<crate::didactic::EnrichedStep> {
    enrich_steps(context, original_expr, filtered)
}
