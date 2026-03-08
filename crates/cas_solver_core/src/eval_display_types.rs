/// Lightweight message kind for rendering eval output in frontends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalDisplayMessageKind {
    Output,
    Warn,
    Info,
}

/// Frontend-agnostic message line emitted by eval render planning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalDisplayMessage {
    pub kind: EvalDisplayMessageKind,
    pub text: String,
}

/// Formatted final result line for eval output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalResultLine {
    pub line: String,
    /// When true, caller should stop rendering extra metadata sections.
    pub terminal: bool,
}

/// Formatted eval metadata lines grouped by display phase.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvalMetadataLines {
    pub warning_lines: Vec<String>,
    pub requires_lines: Vec<String>,
    pub hint_lines: Vec<String>,
    pub assumption_lines: Vec<String>,
}
