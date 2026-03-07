/// Formatted final result line for eval output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalResultLine {
    pub line: String,
    /// When true, caller should stop rendering extra metadata sections.
    pub terminal: bool,
}
