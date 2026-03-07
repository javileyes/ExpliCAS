use cas_ast::ExprId;
use cas_formatter::root_style::ParseStyleSignals;

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

/// Formatted eval metadata lines grouped by display phase.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EvalMetadataLines {
    pub warning_lines: Vec<String>,
    pub requires_lines: Vec<String>,
    pub hint_lines: Vec<String>,
    pub assumption_lines: Vec<String>,
}
