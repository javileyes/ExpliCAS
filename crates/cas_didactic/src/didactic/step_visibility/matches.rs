use super::StepVisibility;
use crate::runtime::Step;

#[allow(clippy::type_complexity)]
pub(super) fn step_matches_visibility(
    step: &Step,
    visibility: StepVisibility,
    classify_step_matches_visibility: fn(
        &Step,
        StepVisibility,
        fn(&Step) -> bool,
        fn(&Step) -> bool,
    ) -> bool,
    is_medium_or_higher_step: fn(&Step) -> bool,
    is_high_or_higher_step: fn(&Step) -> bool,
) -> bool {
    classify_step_matches_visibility(
        step,
        visibility,
        is_medium_or_higher_step,
        is_high_or_higher_step,
    )
}
