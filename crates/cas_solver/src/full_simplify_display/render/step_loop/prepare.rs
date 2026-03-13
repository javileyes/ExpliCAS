use super::super::super::FullSimplifyDisplayMode;

pub(super) fn prepare_step_lines(
    lines: &mut Vec<String>,
    steps: &[crate::Step],
    mode: FullSimplifyDisplayMode,
) -> bool {
    if mode == FullSimplifyDisplayMode::None {
        return false;
    }

    if steps.is_empty() {
        if mode != FullSimplifyDisplayMode::Succinct {
            lines.push("No simplification steps needed.".to_string());
        }
        return false;
    }

    if mode != FullSimplifyDisplayMode::Succinct {
        lines.push("Steps (Aggressive Mode):".to_string());
    }

    true
}
