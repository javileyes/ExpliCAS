//! Output types for ReplCore - structured messages instead of direct printing.
//!
//! This allows the REPL logic to be decoupled from I/O, making it testable
//! and reusable for web/TUI/API contexts.

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
