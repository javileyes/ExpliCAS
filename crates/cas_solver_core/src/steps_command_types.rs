use crate::eval_option_axes::StepsMode;

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
    pub steps_mode: StepsMode,
    pub display_mode: StepsDisplayMode,
}

/// Parsed input for the `steps` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepsCommandInput {
    ShowCurrent,
    SetCollectionMode(StepsMode),
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
        set_steps_mode: Option<StepsMode>,
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
    pub set_steps_mode: Option<StepsMode>,
    pub set_display_mode: Option<StepsDisplayMode>,
}
