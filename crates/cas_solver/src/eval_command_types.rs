use cas_ast::ExprId;
use cas_formatter::root_style::ParseStyleSignals;

/// Evaluated payload for REPL `eval` rendering.
#[derive(Debug, Clone)]
pub struct EvalCommandOutput {
    pub resolved_expr: ExprId,
    pub style_signals: ParseStyleSignals,
    pub steps: crate::DisplayEvalSteps,
    pub stored_entry_line: Option<String>,
    pub metadata: EvalMetadataLines,
    pub result_line: Option<EvalResultLine>,
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

/// Ordered plan for rendering eval output in a frontend.
#[derive(Debug, Clone)]
pub struct EvalCommandRenderPlan {
    pub pre_messages: Vec<EvalDisplayMessage>,
    pub render_steps: bool,
    pub resolved_expr: ExprId,
    pub style_signals: ParseStyleSignals,
    pub steps: crate::DisplayEvalSteps,
    pub result_message: Option<EvalDisplayMessage>,
    pub result_terminal: bool,
    pub post_messages: Vec<EvalDisplayMessage>,
}

/// Errors while evaluating REPL `eval` command.
#[derive(Debug, Clone)]
pub enum EvalCommandError {
    Parse(cas_parser::ParseError),
    Eval(String),
}

#[derive(Debug, Clone)]
pub(crate) struct EvalCommandEvalView {
    pub(crate) stored_id: Option<u64>,
    pub(crate) parsed: ExprId,
    pub(crate) resolved: ExprId,
    pub(crate) result: crate::EvalResult,
    pub(crate) diagnostics: crate::Diagnostics,
    pub(crate) steps: crate::DisplayEvalSteps,
    pub(crate) domain_warnings: Vec<crate::DomainWarning>,
    pub(crate) blocked_hints: Vec<crate::BlockedHint>,
}
