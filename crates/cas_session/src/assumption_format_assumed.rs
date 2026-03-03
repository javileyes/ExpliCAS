use std::collections::{BTreeMap, BTreeSet, HashSet};

fn dedupe_fingerprint(key: &cas_solver::AssumptionKey) -> u64 {
    match key {
        cas_solver::AssumptionKey::NonZero { expr_fingerprint } => *expr_fingerprint,
        cas_solver::AssumptionKey::Positive { expr_fingerprint } => {
            expr_fingerprint.wrapping_add(1_000_000)
        }
        cas_solver::AssumptionKey::NonNegative { expr_fingerprint } => {
            expr_fingerprint.wrapping_add(2_000_000)
        }
        cas_solver::AssumptionKey::Defined { expr_fingerprint } => {
            expr_fingerprint.wrapping_add(3_000_000)
        }
        cas_solver::AssumptionKey::InvTrigPrincipalRange {
            arg_fingerprint, ..
        } => arg_fingerprint.wrapping_add(4_000_000),
        cas_solver::AssumptionKey::ComplexPrincipalBranch {
            arg_fingerprint, ..
        } => arg_fingerprint.wrapping_add(5_000_000),
    }
}

fn condition_text(key: &cas_solver::AssumptionKey, expr_display: &str) -> String {
    match key {
        cas_solver::AssumptionKey::NonZero { .. } => format!("{expr_display} ≠ 0"),
        cas_solver::AssumptionKey::Positive { .. } => format!("{expr_display} > 0"),
        cas_solver::AssumptionKey::NonNegative { .. } => format!("{expr_display} ≥ 0"),
        cas_solver::AssumptionKey::Defined { .. } => format!("{expr_display} is defined"),
        cas_solver::AssumptionKey::InvTrigPrincipalRange { func, .. } => {
            format!("{expr_display} in {func} principal range")
        }
        cas_solver::AssumptionKey::ComplexPrincipalBranch { func, .. } => {
            format!("{func}({expr_display}) principal branch")
        }
    }
}

/// Collect assumptions used from simplification steps.
///
/// Returns `(condition_text, rule_name)` items, deduplicated by
/// `(assumption_kind, expr_fingerprint)` to avoid cascades.
pub fn collect_assumed_conditions_from_steps(steps: &[cas_solver::Step]) -> Vec<(String, String)> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for step in steps {
        let assumption_events = cas_solver::assumption_events_from_step(step);
        for event in &assumption_events {
            let fp = dedupe_fingerprint(&event.key);
            if seen.insert(fp) {
                result.push((
                    condition_text(&event.key, &event.expr_display),
                    step.rule_name.clone(),
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

fn assumption_record_summary_item(record: &cas_solver::AssumptionRecord) -> String {
    if record.count > 1 {
        format!("{}({}) (×{})", record.kind, record.expr, record.count)
    } else {
        format!("{}({})", record.kind, record.expr)
    }
}

/// Format assumptions summary payload for REPL/UI.
///
/// Returns only the right side content (without the `⚠ Assumptions:` prefix).
pub fn format_assumption_records_summary(
    records: &[cas_solver::AssumptionRecord],
) -> Option<String> {
    if records.is_empty() {
        return None;
    }
    let items: Vec<String> = records.iter().map(assumption_record_summary_item).collect();
    Some(items.join(", "))
}

/// Format displayable assumption events into compact single-line strings.
///
/// Output format: `"<icon> <label>: <message>"`.
pub fn format_displayable_assumption_lines(events: &[cas_solver::AssumptionEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| {
            let kind = event.kind;
            if kind.should_display() {
                Some(format!(
                    "{} {}: {}",
                    kind.icon(),
                    kind.label(),
                    event.message
                ))
            } else {
                None
            }
        })
        .collect()
}
