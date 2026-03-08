/// Parsed input for the `profile` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileCommandInput {
    ShowReport,
    Enable,
    Disable,
    Clear,
    Invalid,
}

/// Normalized result for `profile` command handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileCommandResult {
    ShowReport,
    SetEnabled { enabled: bool, message: String },
    Clear { message: String },
    Invalid { message: String },
}
