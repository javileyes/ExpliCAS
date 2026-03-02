/// CLI-facing display mode for step rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepsDisplayMode {
    None,
    Succinct,
    Normal,
    Verbose,
}

/// Runtime state needed to evaluate a `steps` command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepsCommandState {
    pub steps_mode: crate::StepsMode,
    pub display_mode: StepsDisplayMode,
}

/// Parsed input for the `steps` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepsCommandInput {
    ShowCurrent,
    SetCollectionMode(crate::StepsMode),
    SetDisplayMode(StepsDisplayMode),
    UnknownMode(String),
}

/// Normalized result for `steps` command handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepsCommandResult {
    ShowCurrent {
        message: String,
    },
    Update {
        set_steps_mode: Option<crate::StepsMode>,
        set_display_mode: Option<StepsDisplayMode>,
        message: String,
    },
    Invalid {
        message: String,
    },
}

/// Side-effects from applying a `steps` command update to runtime options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepsCommandApplyEffects {
    pub set_steps_mode: Option<crate::StepsMode>,
    pub set_display_mode: Option<StepsDisplayMode>,
}

/// Parse raw `steps ...` command input.
pub fn parse_steps_command_input(line: &str) -> StepsCommandInput {
    let args: Vec<&str> = line.split_whitespace().collect();
    match args.get(1) {
        None => StepsCommandInput::ShowCurrent,
        Some(&"on") => StepsCommandInput::SetCollectionMode(crate::StepsMode::On),
        Some(&"off") => StepsCommandInput::SetCollectionMode(crate::StepsMode::Off),
        Some(&"compact") => StepsCommandInput::SetCollectionMode(crate::StepsMode::Compact),
        Some(&"verbose") => StepsCommandInput::SetDisplayMode(StepsDisplayMode::Verbose),
        Some(&"succinct") => StepsCommandInput::SetDisplayMode(StepsDisplayMode::Succinct),
        Some(&"normal") => StepsCommandInput::SetDisplayMode(StepsDisplayMode::Normal),
        Some(&"none") => StepsCommandInput::SetDisplayMode(StepsDisplayMode::None),
        Some(other) => StepsCommandInput::UnknownMode((*other).to_string()),
    }
}

/// Evaluate a `steps` command into state changes + message.
pub fn evaluate_steps_command_input(line: &str, state: StepsCommandState) -> StepsCommandResult {
    match parse_steps_command_input(line) {
        StepsCommandInput::ShowCurrent => StepsCommandResult::ShowCurrent {
            message: format_steps_current_message(state.steps_mode, state.display_mode),
        },
        StepsCommandInput::SetCollectionMode(mode) => {
            let display = match mode {
                crate::StepsMode::On => Some(StepsDisplayMode::Normal),
                crate::StepsMode::Off => Some(StepsDisplayMode::None),
                crate::StepsMode::Compact => None,
            };
            StepsCommandResult::Update {
                set_steps_mode: Some(mode),
                set_display_mode: display,
                message: format_steps_collection_set_message(mode).to_string(),
            }
        }
        StepsCommandInput::SetDisplayMode(mode) => {
            let steps_mode = match mode {
                StepsDisplayMode::Verbose
                | StepsDisplayMode::Succinct
                | StepsDisplayMode::Normal => Some(crate::StepsMode::On),
                StepsDisplayMode::None => None,
            };
            StepsCommandResult::Update {
                set_steps_mode: steps_mode,
                set_display_mode: Some(mode),
                message: format_steps_display_set_message(mode).to_string(),
            }
        }
        StepsCommandInput::UnknownMode(mode) => StepsCommandResult::Invalid {
            message: format_steps_unknown_mode_message(&mode),
        },
    }
}

/// Apply `steps` update fields into eval options and return external effects.
pub fn apply_steps_command_update(
    set_steps_mode: Option<crate::StepsMode>,
    set_display_mode: Option<StepsDisplayMode>,
    eval_options: &mut crate::EvalOptions,
) -> StepsCommandApplyEffects {
    let mut changed_steps_mode = None;
    if let Some(mode) = set_steps_mode {
        if eval_options.steps_mode != mode {
            eval_options.steps_mode = mode;
            changed_steps_mode = Some(mode);
        }
    }

    StepsCommandApplyEffects {
        set_steps_mode: changed_steps_mode,
        set_display_mode,
    }
}

/// Format current steps collection/display status.
pub fn format_steps_current_message(
    steps_mode: crate::StepsMode,
    display_mode: StepsDisplayMode,
) -> String {
    let mode_str = match steps_mode {
        crate::StepsMode::On => "on",
        crate::StepsMode::Off => "off",
        crate::StepsMode::Compact => "compact",
    };
    let display_str = match display_mode {
        StepsDisplayMode::None => "none",
        StepsDisplayMode::Succinct => "succinct",
        StepsDisplayMode::Normal => "normal",
        StepsDisplayMode::Verbose => "verbose",
    };
    format!(
        "Steps collection: {}\n\
         Steps display: {}\n\
           (use 'steps on|off|compact' for collection)\n\
           (use 'steps verbose|succinct|normal|none' for display)",
        mode_str, display_str
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

#[cfg(test)]
mod tests {
    use super::{
        apply_steps_command_update, evaluate_steps_command_input, format_steps_current_message,
        format_steps_unknown_mode_message, parse_steps_command_input, StepsCommandApplyEffects,
        StepsCommandInput, StepsCommandResult, StepsCommandState, StepsDisplayMode,
    };

    #[test]
    fn parse_steps_command_input_reads_compact() {
        assert_eq!(
            parse_steps_command_input("steps compact"),
            StepsCommandInput::SetCollectionMode(crate::StepsMode::Compact)
        );
    }

    #[test]
    fn parse_steps_command_input_reads_verbose_display_mode() {
        assert_eq!(
            parse_steps_command_input("steps verbose"),
            StepsCommandInput::SetDisplayMode(StepsDisplayMode::Verbose)
        );
    }

    #[test]
    fn format_steps_current_message_reports_modes() {
        let text = format_steps_current_message(crate::StepsMode::On, StepsDisplayMode::Normal);
        assert!(text.contains("Steps collection: on"));
        assert!(text.contains("Steps display: normal"));
    }

    #[test]
    fn format_steps_unknown_mode_message_mentions_usage() {
        let text = format_steps_unknown_mode_message("oops");
        assert!(text.contains("Unknown steps mode: 'oops'"));
        assert!(text.contains("Usage: steps"));
    }

    #[test]
    fn evaluate_steps_command_input_off_disables_collection_and_display() {
        let state = StepsCommandState {
            steps_mode: crate::StepsMode::On,
            display_mode: StepsDisplayMode::Normal,
        };
        let out = evaluate_steps_command_input("steps off", state);
        assert_eq!(
            out,
            StepsCommandResult::Update {
                set_steps_mode: Some(crate::StepsMode::Off),
                set_display_mode: Some(StepsDisplayMode::None),
                message: "Steps: off\n  ⚡ Steps disabled (faster). Warnings still enabled."
                    .to_string(),
            }
        );
    }

    #[test]
    fn evaluate_steps_command_input_show_current_renders_message() {
        let state = StepsCommandState {
            steps_mode: crate::StepsMode::Compact,
            display_mode: StepsDisplayMode::Succinct,
        };
        let out = evaluate_steps_command_input("steps", state);
        match out {
            StepsCommandResult::ShowCurrent { message } => {
                assert!(message.contains("Steps collection: compact"));
                assert!(message.contains("Steps display: succinct"));
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn apply_steps_command_update_sets_mode_when_changed() {
        let mut eval_options = crate::EvalOptions {
            steps_mode: crate::StepsMode::On,
            ..crate::EvalOptions::default()
        };
        let effects = apply_steps_command_update(
            Some(crate::StepsMode::Compact),
            Some(StepsDisplayMode::Succinct),
            &mut eval_options,
        );
        assert_eq!(eval_options.steps_mode, crate::StepsMode::Compact);
        assert_eq!(
            effects,
            StepsCommandApplyEffects {
                set_steps_mode: Some(crate::StepsMode::Compact),
                set_display_mode: Some(StepsDisplayMode::Succinct),
            }
        );
    }

    #[test]
    fn apply_steps_command_update_skips_mode_when_unchanged() {
        let mut eval_options = crate::EvalOptions {
            steps_mode: crate::StepsMode::On,
            ..crate::EvalOptions::default()
        };
        let effects = apply_steps_command_update(
            Some(crate::StepsMode::On),
            Some(StepsDisplayMode::Verbose),
            &mut eval_options,
        );
        assert_eq!(
            effects,
            StepsCommandApplyEffects {
                set_steps_mode: None,
                set_display_mode: Some(StepsDisplayMode::Verbose),
            }
        );
    }
}
