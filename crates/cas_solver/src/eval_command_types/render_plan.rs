use cas_ast::ExprId;
use cas_formatter::root_style::ParseStyleSignals;

use super::EvalDisplayMessage;

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
