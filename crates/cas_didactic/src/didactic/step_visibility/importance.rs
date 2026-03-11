use crate::cas_solver::{ImportanceLevel, Step};

pub(super) fn is_medium_or_higher_step(step: &Step) -> bool {
    step.get_importance() >= ImportanceLevel::Medium
}

pub(super) fn is_high_or_higher_step(step: &Step) -> bool {
    step.get_importance() >= ImportanceLevel::High
}
