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

/// Applied result for `config ...` command against a mutable [`crate::CasConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigCommandApplyOutput {
    pub message: String,
    pub sync_simplifier: bool,
}
