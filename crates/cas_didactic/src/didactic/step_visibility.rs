mod classify;
mod clone;
mod importance;
mod matches;
mod simplify;
mod types;

use super::display_policy::StepDisplayMode;
use crate::cas_solver::Step;
use cas_ast::ExprId;

pub(crate) use types::StepVisibility;

pub(crate) fn step_matches_visibility(step: &Step, visibility: StepVisibility) -> bool {
    matches::step_matches_visibility(
        step,
        visibility,
        classify::step_matches_visibility,
        is_medium_or_higher_step,
        is_high_or_higher_step,
    )
}

pub(crate) fn clone_steps_matching_visibility(
    steps: &[Step],
    visibility: StepVisibility,
) -> Vec<Step> {
    clone::clone_steps_matching_visibility(steps, visibility, step_matches_visibility)
}

pub(crate) fn infer_original_expr_for_steps(steps: &[Step]) -> Option<ExprId> {
    clone::infer_original_expr_for_steps(steps)
}

pub(crate) fn should_show_simplify_step(step: &Step, mode: StepDisplayMode) -> bool {
    simplify::should_show_simplify_step(step, mode, step_matches_visibility)
}

pub fn is_medium_or_higher_step(step: &Step) -> bool {
    importance::is_medium_or_higher_step(step)
}

pub fn is_high_or_higher_step(step: &Step) -> bool {
    importance::is_high_or_higher_step(step)
}
