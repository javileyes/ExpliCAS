//! Eval command entrypoints exposed for CLI/frontends.

use cas_ast::ExprId;
use cas_formatter::root_style::ParseStyleSignals;

pub use crate::eval_command_eval::evaluate_eval_command_output;
pub use crate::eval_command_render::build_eval_command_render_plan;
pub use crate::eval_command_text::evaluate_eval_text_simplify_with_session;
pub use cas_solver_core::eval_display_types::{
    EvalDisplayMessage, EvalDisplayMessageKind, EvalMetadataLines, EvalResultLine,
};

/// Errors while evaluating REPL `eval` command.
#[derive(Debug, Clone)]
pub enum EvalCommandError {
    Parse(cas_parser::ParseError),
    Eval(String),
}

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
