use crate::SetDisplayMode;

/// Message kind for REPL `set` command output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplSetMessageKind {
    Output,
    Info,
}

/// Fully-evaluated REPL `set` command output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplSetCommandOutput {
    pub message_kind: ReplSetMessageKind,
    pub message: String,
    pub set_display_mode: Option<SetDisplayMode>,
}
