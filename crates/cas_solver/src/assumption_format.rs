use std::collections::{BTreeMap, BTreeSet, HashSet};

use cas_ast::{Context, ExprId};

use crate::{AssumptionEvent, AssumptionKind, AssumptionRecord, Step};

/// Collect assumptions used from simplification steps.
///
/// Returns `(condition_text, rule_name)` items, deduplicated by
/// `(assumption_kind, expr_fingerprint)` to avoid cascades.
pub fn collect_assumed_conditions_from_steps(steps: &[Step]) -> Vec<(String, String)> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for step in steps {
        for event in step.assumption_events() {
            let fp = crate::assumption_key_dedupe_fingerprint(&event.key);
            if seen.insert(fp) {
                result.push((
                    crate::assumption_condition_text(&event.key, &event.expr_display),
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

fn assumption_record_summary_item(record: &AssumptionRecord) -> String {
    if record.count > 1 {
        format!("{}({}) (×{})", record.kind, record.expr, record.count)
    } else {
        format!("{}({})", record.kind, record.expr)
    }
}

/// Format assumptions summary payload for REPL/UI.
///
/// Returns only the right side content (without the `⚠ Assumptions:` prefix).
pub fn format_assumption_records_summary(records: &[AssumptionRecord]) -> Option<String> {
    if records.is_empty() {
        return None;
    }
    let items: Vec<String> = records.iter().map(assumption_record_summary_item).collect();
    Some(items.join(", "))
}

/// Format displayable assumption events into compact single-line strings.
///
/// Output format: `"<icon> <label>: <message>"`.
pub fn format_displayable_assumption_lines(events: &[AssumptionEvent]) -> Vec<String> {
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

/// Format displayable assumptions emitted by one step.
pub fn format_displayable_assumption_lines_for_step(step: &Step) -> Vec<String> {
    format_displayable_assumption_lines(step.assumption_events())
}

/// Group displayable assumption events by kind for compact timeline/HTML presentation.
///
/// Output format: one line per kind in stable order:
/// `Requires`, `Branch`, `Domain`, `Assumes`.
pub fn format_displayable_assumption_lines_grouped(events: &[AssumptionEvent]) -> Vec<String> {
    let mut requires = Vec::new();
    let mut branches = Vec::new();
    let mut domain_ext = Vec::new();
    let mut assumes = Vec::new();

    for event in events.iter().filter(|event| event.kind.should_display()) {
        match event.kind {
            AssumptionKind::RequiresIntroduced => requires.push(event.message.clone()),
            AssumptionKind::BranchChoice => branches.push(event.message.clone()),
            AssumptionKind::DomainExtension => domain_ext.push(event.message.clone()),
            AssumptionKind::HeuristicAssumption => assumes.push(event.message.clone()),
            AssumptionKind::DerivedFromRequires => {}
        }
    }

    let mut lines = Vec::new();
    if !requires.is_empty() {
        lines.push(format!(
            "{} {}: {}",
            AssumptionKind::RequiresIntroduced.icon(),
            AssumptionKind::RequiresIntroduced.label(),
            requires.join(", ")
        ));
    }
    if !branches.is_empty() {
        lines.push(format!(
            "{} {}: {}",
            AssumptionKind::BranchChoice.icon(),
            AssumptionKind::BranchChoice.label(),
            branches.join(", ")
        ));
    }
    if !domain_ext.is_empty() {
        lines.push(format!(
            "{} {}: {}",
            AssumptionKind::DomainExtension.icon(),
            AssumptionKind::DomainExtension.label(),
            domain_ext.join(", ")
        ));
    }
    if !assumes.is_empty() {
        lines.push(format!(
            "{} {}: {}",
            AssumptionKind::HeuristicAssumption.icon(),
            AssumptionKind::HeuristicAssumption.label(),
            assumes.join(", ")
        ));
    }
    lines
}

/// Group displayable assumptions emitted by one step.
pub fn format_displayable_assumption_lines_grouped_for_step(step: &Step) -> Vec<String> {
    format_displayable_assumption_lines_grouped(step.assumption_events())
}

/// Render required conditions as plain display lines with a custom prefix.
pub fn format_required_condition_lines(
    ctx: &Context,
    conditions: &[crate::ImplicitCondition],
    line_prefix: &str,
) -> Vec<String> {
    conditions
        .iter()
        .map(|cond| format!("{line_prefix}{}", cond.display(ctx)))
        .collect()
}

/// Render domain warnings as display lines with a custom prefix.
///
/// When `include_rule` is true, appends `(from <rule>)`.
pub fn format_domain_warning_lines(
    warnings: &[crate::DomainWarning],
    include_rule: bool,
    line_prefix: &str,
) -> Vec<String> {
    warnings
        .iter()
        .map(|warning| {
            if include_rule {
                format!(
                    "{line_prefix}{} (from {})",
                    warning.message, warning.rule_name
                )
            } else {
                format!("{line_prefix}{}", warning.message)
            }
        })
        .collect()
}

/// Render blocked hints as compact rule/suggestion lines using a line prefix.
pub fn format_blocked_hint_lines(hints: &[crate::BlockedHint], line_prefix: &str) -> Vec<String> {
    hints
        .iter()
        .map(|hint| format!("{line_prefix}{} (hint: {})", hint.rule, hint.suggestion))
        .collect()
}

/// Render normalized required conditions as REPL bullet lines.
pub fn format_normalized_condition_lines(
    ctx: &mut Context,
    conditions: &[crate::ImplicitCondition],
    debug_mode: bool,
) -> Vec<String> {
    let normalized_conditions = crate::normalize_and_dedupe_conditions(ctx, conditions);
    normalized_conditions
        .iter()
        .map(|cond| {
            if debug_mode {
                format!("  • {} (normalized)", cond.display(ctx))
            } else {
                format!("  • {}", cond.display(ctx))
            }
        })
        .collect()
}

/// Render display lines for `Diagnostics::requires` after witness filtering.
pub fn format_diagnostics_requires_lines(
    ctx: &mut Context,
    diagnostics: &crate::Diagnostics,
    result_expr: Option<ExprId>,
    display_level: crate::RequiresDisplayLevel,
    debug_mode: bool,
) -> Vec<String> {
    let filtered: Vec<_> = if let Some(result) = result_expr {
        diagnostics.filter_requires_for_display(ctx, result, display_level)
    } else {
        diagnostics.requires.iter().collect()
    };

    if filtered.is_empty() {
        return Vec::new();
    }

    let conditions: Vec<_> = filtered.iter().map(|item| item.cond.clone()).collect();
    format_normalized_condition_lines(ctx, &conditions, debug_mode)
}
