use cas_api_models::SubstituteRenderMode;

pub(super) fn should_render_substitute_step(
    step: &crate::Step,
    mode: SubstituteRenderMode,
) -> bool {
    match mode {
        SubstituteRenderMode::None => false,
        SubstituteRenderMode::Verbose => true,
        SubstituteRenderMode::Succinct | SubstituteRenderMode::Normal => {
            if step.get_importance() < crate::ImportanceLevel::Medium {
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
