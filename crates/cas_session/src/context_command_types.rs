/// Parsed input for the `context` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextCommandInput {
    ShowCurrent,
    SetMode(cas_solver::ContextMode),
    UnknownMode(String),
}

/// Normalized result for `context` command handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContextCommandResult {
    ShowCurrent {
        message: String,
    },
    SetMode {
        mode: cas_solver::ContextMode,
        message: String,
    },
    Invalid {
        message: String,
    },
}

/// Result from evaluating + applying a `context` command to runtime options.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextCommandApplyOutput {
    pub message: String,
    pub rebuild_simplifier: bool,
}
