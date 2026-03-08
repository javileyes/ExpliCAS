use crate::steps_command_types::StepsDisplayMode;

use super::labels::{steps_display_mode_label, steps_mode_label};

/// Format current steps collection/display status.
pub fn format_steps_current_message(
    steps_mode: crate::StepsMode,
    display_mode: StepsDisplayMode,
) -> String {
    format!(
        "Steps collection: {}\n\
         Steps display: {}\n\
           (use 'steps on|off|compact' for collection)\n\
           (use 'steps verbose|succinct|normal|none' for display)",
        steps_mode_label(steps_mode),
        steps_display_mode_label(display_mode)
    )
}

/// Format feedback for collection-mode updates.
pub fn format_steps_collection_set_message(mode: crate::StepsMode) -> &'static str {
    match mode {
        crate::StepsMode::On => "Steps: on (full collection, normal display)",
        crate::StepsMode::Off => {
            "Steps: off\n  ⚡ Steps disabled (faster). Warnings still enabled."
        }
        crate::StepsMode::Compact => "Steps: compact (no before/after snapshots)",
    }
}

/// Format feedback for display-mode updates.
pub fn format_steps_display_set_message(mode: StepsDisplayMode) -> &'static str {
    match mode {
        StepsDisplayMode::Verbose => "Steps: verbose (all rules, full detail)",
        StepsDisplayMode::Succinct => "Steps: succinct (compact 1-line per step)",
        StepsDisplayMode::Normal => "Steps: normal (default display)",
        StepsDisplayMode::None => {
            "Steps display: none (collection still active)\n  Use 'steps off' to also disable collection."
        }
    }
}

/// Format unknown-mode error for `steps`.
pub fn format_steps_unknown_mode_message(mode: &str) -> String {
    format!(
        "Unknown steps mode: '{}'\n\
             Usage: steps [on | off | compact | verbose | succinct | normal | none]\n\
               Collection modes:\n\
                 on      - Full steps with snapshots (default)\n\
                 off     - No steps (fastest, warnings preserved)\n\
                 compact - Minimal steps (no snapshots)\n\
               Display modes:\n\
                 verbose - Show all rules, full detail\n\
                 succinct- Compact 1-line per step\n\
                 normal  - Standard display (default)\n\
                 none    - Hide steps output (collection still active)",
        mode
    )
}
