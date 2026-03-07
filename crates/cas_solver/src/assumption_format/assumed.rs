mod collect;
mod format;
mod group;

use crate::Step;

/// Collect assumptions used from simplification steps.
///
/// Returns `(condition_text, rule_name)` items, deduplicated by
/// `(assumption_kind, expr_fingerprint)` to avoid cascades.
pub fn collect_assumed_conditions_from_steps(steps: &[Step]) -> Vec<(String, String)> {
    collect::collect_assumed_conditions_from_steps(steps)
}

/// Group `(condition, rule)` assumed-condition pairs by rule name.
///
/// Conditions are sorted and deduplicated inside each rule group.
pub fn group_assumed_conditions_by_rule(
    conditions: &[(String, String)],
) -> Vec<(String, Vec<String>)> {
    group::group_assumed_conditions_by_rule(conditions)
}

/// Format "assumptions used" report lines for REPL display.
pub fn format_assumed_conditions_report_lines(conditions: &[(String, String)]) -> Vec<String> {
    format::format_assumed_conditions_report_lines(conditions)
}
