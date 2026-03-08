use super::StepVisibility;
use crate::didactic::display_policy::StepDisplayMode;
use cas_solver::Step;

pub(super) fn should_show_simplify_step(
    step: &Step,
    mode: StepDisplayMode,
    step_matches_visibility: fn(&Step, StepVisibility) -> bool,
) -> bool {
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
