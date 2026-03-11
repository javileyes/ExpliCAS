use crate::Step;

/// Collect assumptions used from simplification steps.
///
/// Returns `(condition_text, rule_name)` items, deduplicated by
/// `(assumption_kind, expr_fingerprint)` to avoid cascades.
pub fn collect_assumed_conditions_from_steps(steps: &[Step]) -> Vec<(String, String)> {
    cas_solver_core::assumption_usage::collect_assumed_conditions_from_steps(steps)
}

/// Format "assumptions used" report lines for REPL display.
pub fn format_assumed_conditions_report_lines(conditions: &[(String, String)]) -> Vec<String> {
    cas_solver_core::assumption_usage::format_assumed_conditions_report_lines(conditions)
}
