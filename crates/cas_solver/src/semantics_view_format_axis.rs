mod evaluation;
mod reporting;

use cas_solver_core::semantics_view_types::SemanticsViewState;

/// Format one semantic axis description and values.
pub fn format_semantics_axis_lines(state: &SemanticsViewState, axis: &str) -> Vec<String> {
    match axis {
        "domain" | "value" | "branch" | "inv_trig" | "const_fold" => {
            evaluation::format_evaluation_axis_lines(state, axis)
        }
        "assumptions" | "assume_scope" | "requires" => {
            reporting::format_reporting_axis_lines(state, axis)
        }
        _ => vec![format!("Unknown axis: {}", axis)],
    }
}
