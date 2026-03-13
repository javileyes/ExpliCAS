mod unchanged;

use super::FullSimplifyDisplayMode;

pub(super) fn should_show_simplify_step(step: &crate::Step, mode: FullSimplifyDisplayMode) -> bool {
    match mode {
        FullSimplifyDisplayMode::None => false,
        FullSimplifyDisplayMode::Verbose => true,
        FullSimplifyDisplayMode::Succinct | FullSimplifyDisplayMode::Normal => {
            if step.get_importance() < crate::ImportanceLevel::Medium {
                return false;
            }
            if unchanged::is_unchanged_global_step(step) {
                return false;
            }
            true
        }
    }
}
