use crate::cas_solver::Step;
use cas_ast::ExprId;

pub(super) fn filter_step_payloads(
    steps: &[Step],
    clone_steps_matching_visibility: fn(&[Step], crate::didactic::StepVisibility) -> Vec<Step>,
) -> Vec<Step> {
    clone_steps_matching_visibility(steps, crate::didactic::StepVisibility::MediumOrHigher)
}

pub(super) fn infer_original_expr_for_filtered_steps(
    filtered: &[Step],
    infer_original_expr_for_steps: fn(&[Step]) -> Option<ExprId>,
) -> Option<ExprId> {
    infer_original_expr_for_steps(filtered)
}
