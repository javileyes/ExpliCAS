use super::StepVisibility;
use crate::cas_solver::Step;
use cas_ast::ExprId;

pub(super) fn clone_steps_matching_visibility(
    steps: &[Step],
    visibility: StepVisibility,
    step_matches_visibility: fn(&Step, StepVisibility) -> bool,
) -> Vec<Step> {
    steps
        .iter()
        .filter(|step| step_matches_visibility(step, visibility))
        .cloned()
        .collect()
}

pub(super) fn infer_original_expr_for_steps(steps: &[Step]) -> Option<ExprId> {
    steps
        .first()
        .map(|step| step.global_before.unwrap_or(step.before))
}
