//! Output types for ReplCore - structured messages instead of direct printing.
//!
//! This allows the REPL logic to be decoupled from I/O, making it testable
//! and reusable for web/TUI/API contexts.

use cas_didactic::TimelineCliAction;
use std::path::PathBuf;

/// Structured message returned by ReplCore operations.
#[derive(Debug, Clone)]
pub enum ReplMsg {
    /// Informational message (general feedback)
    Info(String),
    /// Warning message (non-fatal issue)
    Warn(String),
    /// Error message (operation failed)
    Error(String),
    /// Main output/result (what the user asked for)
    Output(String),
    /// Step-by-step computation trace
    Steps(String),
    /// Debug mode output
    Debug(String),
    /// Action: write content to a file (executed by shell, not core)
    WriteFile { path: PathBuf, contents: String },
    /// Action: open a file with host OS default app (executed by shell, not core)
    OpenFile { path: PathBuf },
}

impl ReplMsg {
    /// Create an Output message
    pub fn output(s: impl Into<String>) -> Self {
        ReplMsg::Output(s.into())
    }

    /// Create an Info message
    pub fn info(s: impl Into<String>) -> Self {
        ReplMsg::Info(s.into())
    }

    /// Create a Warn message
    pub fn warn(s: impl Into<String>) -> Self {
        ReplMsg::Warn(s.into())
    }

    /// Create an Error message
    pub fn error(s: impl Into<String>) -> Self {
        ReplMsg::Error(s.into())
    }

    /// Create a Steps message
    pub fn steps(s: impl Into<String>) -> Self {
        ReplMsg::Steps(s.into())
    }

    /// Create a Debug message
    pub fn debug(s: impl Into<String>) -> Self {
        ReplMsg::Debug(s.into())
    }
}

/// Collection of messages returned by a ReplCore operation
pub type ReplReply = Vec<ReplMsg>;

/// Create a new empty ReplReply
pub fn reply() -> ReplReply {
    Vec::new()
}

/// Create a ReplReply with a single Output message
pub fn reply_output(s: impl Into<String>) -> ReplReply {
    vec![ReplMsg::output(s)]
}

/// Create a `ReplReply` from output lines.
pub fn reply_output_lines<I, S>(lines: I) -> ReplReply
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    lines.into_iter().map(ReplMsg::output).collect()
}

/// Convert timeline CLI actions into a REPL reply payload.
pub fn timeline_cli_actions_to_reply(actions: Vec<TimelineCliAction>) -> ReplReply {
    let mut reply = ReplReply::new();
    for action in actions {
        match action {
            TimelineCliAction::Output(line) => reply.push(ReplMsg::output(line)),
            TimelineCliAction::WriteFile { path, contents } => {
                reply.push(ReplMsg::WriteFile {
                    path: PathBuf::from(path),
                    contents,
                });
            }
            TimelineCliAction::OpenFile { path } => {
                reply.push(ReplMsg::OpenFile {
                    path: PathBuf::from(path),
                });
            }
        }
    }
    reply
}

/// Convert a visualize command output into REPL actions.
pub fn visualize_output_to_reply(output: cas_session::VisualizeCommandOutput) -> ReplReply {
    let mut reply = vec![ReplMsg::WriteFile {
        path: PathBuf::from(output.file_name),
        contents: output.dot_source,
    }];
    reply.extend(output.hint_lines.into_iter().map(ReplMsg::output));
    reply
}

/// Extension trait for ReplReply to add helper methods
pub trait ReplReplyExt {
    fn push_output(&mut self, s: impl Into<String>);
    fn push_info(&mut self, s: impl Into<String>);
    fn push_warn(&mut self, s: impl Into<String>);
    fn push_error(&mut self, s: impl Into<String>);
    fn push_steps(&mut self, s: impl Into<String>);
    fn push_debug(&mut self, s: impl Into<String>);
}

impl ReplReplyExt for ReplReply {
    fn push_output(&mut self, s: impl Into<String>) {
        self.push(ReplMsg::output(s));
    }

    fn push_info(&mut self, s: impl Into<String>) {
        self.push(ReplMsg::info(s));
    }

    fn push_warn(&mut self, s: impl Into<String>) {
        self.push(ReplMsg::warn(s));
    }

    fn push_error(&mut self, s: impl Into<String>) {
        self.push(ReplMsg::error(s));
    }

    fn push_steps(&mut self, s: impl Into<String>) {
        self.push(ReplMsg::steps(s));
    }

    fn push_debug(&mut self, s: impl Into<String>) {
        self.push(ReplMsg::debug(s));
    }
}

use super::Verbosity;

/// Delta of UI state changes produced by a core operation.
/// The Repl wrapper applies these changes after processing the reply.
#[derive(Debug, Clone, Default)]
pub struct UiDelta {
    /// New verbosity level (if changed by command like `set steps`)
    pub verbosity: Option<Verbosity>,
}

/// Result from a core operation: reply messages + optional UI state changes.
#[derive(Debug, Clone)]
pub struct CoreResult {
    /// Messages to display
    pub reply: ReplReply,
    /// UI state changes to apply
    pub ui_delta: UiDelta,
}

impl CoreResult {
    /// Create a CoreResult with only reply, no UI changes
    pub fn reply_only(reply: ReplReply) -> Self {
        Self {
            reply,
            ui_delta: UiDelta::default(),
        }
    }

    /// Create a CoreResult with reply and UI delta
    pub fn with_delta(reply: ReplReply, ui_delta: UiDelta) -> Self {
        Self { reply, ui_delta }
    }
}

impl From<ReplReply> for CoreResult {
    fn from(reply: ReplReply) -> Self {
        Self::reply_only(reply)
    }
}

use super::Repl;

impl Repl {
    /// Apply UI delta mutations (e.g. verbosity) emitted by core handlers.
    pub(crate) fn apply_ui_delta(&mut self, ui_delta: UiDelta) {
        if let Some(verbosity) = ui_delta.verbosity {
            self.verbosity = verbosity;
        }
    }

    /// Apply UI delta and return the reply payload for printing.
    pub(crate) fn finalize_core_result(&mut self, result: CoreResult) -> ReplReply {
        self.apply_ui_delta(result.ui_delta);
        result.reply
    }
}
