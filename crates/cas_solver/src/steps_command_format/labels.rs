use cas_solver_core::steps_command_types::StepsDisplayMode;

pub(super) fn steps_mode_label(mode: crate::StepsMode) -> &'static str {
    match mode {
        crate::StepsMode::On => "on",
        crate::StepsMode::Off => "off",
        crate::StepsMode::Compact => "compact",
    }
}

pub(super) fn steps_display_mode_label(mode: StepsDisplayMode) -> &'static str {
    match mode {
        StepsDisplayMode::None => "none",
        StepsDisplayMode::Succinct => "succinct",
        StepsDisplayMode::Normal => "normal",
        StepsDisplayMode::Verbose => "verbose",
    }
}
