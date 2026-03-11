use crate::assumption_model::{assumption_condition_text, assumption_key_dedupe_fingerprint};
use crate::step_model::Step;
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// Collect assumptions used from simplification steps.
///
/// Returns `(condition_text, rule_name)` items, deduplicated by
/// `(assumption_kind, expr_fingerprint)` to avoid cascades.
pub fn collect_assumed_conditions_from_steps(steps: &[Step]) -> Vec<(String, String)> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for step in steps {
        for event in step.assumption_events() {
            let fp = assumption_key_dedupe_fingerprint(&event.key);
            if seen.insert(fp) {
                result.push((
                    assumption_condition_text(&event.key, &event.expr_display),
                    step.rule_name.to_string(),
                ));
            }
        }
    }

    result
}

/// Group `(condition, rule)` assumed-condition pairs by rule name.
///
/// Conditions are sorted and deduplicated inside each rule group.
pub fn group_assumed_conditions_by_rule(
    conditions: &[(String, String)],
) -> Vec<(String, Vec<String>)> {
    let mut grouped: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (condition, rule) in conditions {
        grouped
            .entry(rule.clone())
            .or_default()
            .insert(condition.clone());
    }

    grouped
        .into_iter()
        .map(|(rule, conditions)| (rule, conditions.into_iter().collect()))
        .collect()
}

/// Format "assumptions used" report lines for REPL display.
pub fn format_assumed_conditions_report_lines(conditions: &[(String, String)]) -> Vec<String> {
    if conditions.is_empty() {
        return Vec::new();
    }

    if conditions.len() == 1 {
        let (cond, rule) = &conditions[0];
        return vec![format!(
            "ℹ️  Assumptions used (assumed): {} [{}]",
            cond, rule
        )];
    }

    let mut lines = vec!["ℹ️  Assumptions used (assumed):".to_string()];
    for (rule, conds) in group_assumed_conditions_by_rule(conditions) {
        lines.push(format!("   - {} [{}]", conds.join(", "), rule));
    }
    lines
}
