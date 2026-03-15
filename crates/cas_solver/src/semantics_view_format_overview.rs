mod evaluation;
mod reporting;

use crate::SemanticsViewState;

/// Format all semantic axis settings.
pub fn format_semantics_overview_lines(state: &SemanticsViewState) -> Vec<String> {
    let mut lines = vec!["Semantics:".to_string()];
    lines.extend(evaluation::format_evaluation_overview_lines(state));
    lines.extend(reporting::format_reporting_overview_lines(state));
    lines
}
