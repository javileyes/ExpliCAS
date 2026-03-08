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
