pub(super) fn is_unchanged_global_step(step: &crate::Step) -> bool {
    matches!(
        (step.global_before, step.global_after),
        (Some(before), Some(after)) if before == after
    )
}
