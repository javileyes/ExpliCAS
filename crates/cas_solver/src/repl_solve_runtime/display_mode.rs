use crate::SetDisplayMode;

pub(super) fn map_solve_display_mode(mode: SetDisplayMode) -> crate::SolveDisplayMode {
    match mode {
        SetDisplayMode::None => crate::SolveDisplayMode::None,
        SetDisplayMode::Succinct => crate::SolveDisplayMode::Succinct,
        SetDisplayMode::Normal => crate::SolveDisplayMode::Normal,
        SetDisplayMode::Verbose => crate::SolveDisplayMode::Verbose,
    }
}

pub(super) fn map_full_simplify_display_mode(
    mode: SetDisplayMode,
) -> crate::FullSimplifyDisplayMode {
    match mode {
        SetDisplayMode::None => crate::FullSimplifyDisplayMode::None,
        SetDisplayMode::Succinct => crate::FullSimplifyDisplayMode::Succinct,
        SetDisplayMode::Normal => crate::FullSimplifyDisplayMode::Normal,
        SetDisplayMode::Verbose => crate::FullSimplifyDisplayMode::Verbose,
    }
}
