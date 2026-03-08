use cas_ast::ExprId;
use cas_formatter::root_style::ParseStyleSignals;
use cas_solver_core::eval_display_types::EvalMetadataLines;

use super::EvalResultLine;

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
