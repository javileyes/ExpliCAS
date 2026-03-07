use crate::SetDisplayMode;

pub(super) fn on_off(enabled: bool) -> &'static str {
    if enabled {
        "on"
    } else {
        "off"
    }
}

pub(super) fn steps_mode_label(mode: crate::StepsMode) -> &'static str {
    match mode {
        crate::StepsMode::On => "on",
        crate::StepsMode::Off => "off",
        crate::StepsMode::Compact => "compact",
    }
}

pub(super) fn display_mode_label(mode: SetDisplayMode) -> &'static str {
    match mode {
        SetDisplayMode::None => "none",
        SetDisplayMode::Succinct => "succinct",
        SetDisplayMode::Normal => "normal",
        SetDisplayMode::Verbose => "verbose",
    }
}
