/// Parsed input for `config ...` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigCommandInput {
    List,
    Save,
    Restore,
    SetRule { rule: String, enable: bool },
    MissingRuleArg { action: String },
    InvalidUsage,
    UnknownSubcommand { subcommand: String },
}

/// Evaluated result for `config ...` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigCommandResult {
    ShowList {
        message: String,
    },
    SaveRequested,
    RestoreRequested,
    ApplyToggleConfig {
        toggles: crate::SimplifierToggleConfig,
        message: String,
    },
    Error {
        message: String,
    },
}
