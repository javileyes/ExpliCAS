use super::display_policy::StepDisplayMode;
use cas_ast::ExprId;
use cas_solver::{ImportanceLevel, Step};

/// Shared visibility policy for step-oriented didactic frontends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepVisibility {
    All,
    MediumOrHigher,
    HighOrHigher,
}

pub(crate) fn step_matches_visibility(step: &Step, visibility: StepVisibility) -> bool {
    match visibility {
        StepVisibility::All => true,
        StepVisibility::MediumOrHigher => is_medium_or_higher_step(step),
        StepVisibility::HighOrHigher => is_high_or_higher_step(step),
    }
}

pub(crate) fn clone_steps_matching_visibility(
    steps: &[Step],
    visibility: StepVisibility,
) -> Vec<Step> {
    steps
        .iter()
        .filter(|step| step_matches_visibility(step, visibility))
        .cloned()
        .collect()
}

pub(crate) fn infer_original_expr_for_steps(steps: &[Step]) -> Option<ExprId> {
    steps
        .first()
        .map(|step| step.global_before.unwrap_or(step.before))
}

pub(crate) fn should_show_simplify_step(step: &Step, mode: StepDisplayMode) -> bool {
    match mode {
        StepDisplayMode::None => false,
        StepDisplayMode::Verbose => step_matches_visibility(step, StepVisibility::All),
        StepDisplayMode::Succinct | StepDisplayMode::Normal => {
            if !step_matches_visibility(step, StepVisibility::MediumOrHigher) {
                return false;
            }
            if let (Some(before), Some(after)) = (step.global_before, step.global_after) {
                if before == after {
                    return false;
                }
            }
            true
        }
    }
}

pub fn is_medium_or_higher_step(step: &Step) -> bool {
    step.get_importance() >= ImportanceLevel::Medium
}

pub fn is_high_or_higher_step(step: &Step) -> bool {
    step.get_importance() >= ImportanceLevel::High
}
