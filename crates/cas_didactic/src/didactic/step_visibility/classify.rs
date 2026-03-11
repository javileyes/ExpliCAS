use super::StepVisibility;
use crate::cas_solver::Step;

pub(super) fn step_matches_visibility(
    step: &Step,
    visibility: StepVisibility,
    is_medium_or_higher_step: fn(&Step) -> bool,
    is_high_or_higher_step: fn(&Step) -> bool,
) -> bool {
    match visibility {
        StepVisibility::All => true,
        StepVisibility::MediumOrHigher => is_medium_or_higher_step(step),
        StepVisibility::HighOrHigher => is_high_or_higher_step(step),
    }
}
