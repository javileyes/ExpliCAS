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
