use cas_ast::Context;

use crate::assumption_format_blocked_support::{
    blocked_hint_suggestion, format_blocked_hint_condition,
};

fn assumption_record_condition(record: &cas_solver::AssumptionRecord) -> String {
    match record.kind.to_ascii_lowercase().as_str() {
        "positive" => format!("{} > 0", record.expr),
        "nonzero" => format!("{} ≠ 0", record.expr),
        "nonnegative" => format!("{} ≥ 0", record.expr),
        "defined" => format!("{} is defined", record.expr),
        _ => format!("{} ({})", record.expr, record.kind),
    }
}

fn format_assumption_records_conditions(records: &[cas_solver::AssumptionRecord]) -> Vec<String> {
    let mut items: Vec<String> = records.iter().map(assumption_record_condition).collect();
    items.sort();
    items.dedup();
    items
}

fn format_assumption_records_section_lines(
    records: &[cas_solver::AssumptionRecord],
    header: &str,
    line_prefix: &str,
) -> Vec<String> {
    if records.is_empty() {
        return Vec::new();
    }

    let mut lines = vec![header.to_string()];
    for cond in format_assumption_records_conditions(records) {
        lines.push(format!("{line_prefix}{cond}"));
    }
    lines
}

fn collect_blocked_hint_items(
    ctx: &Context,
    hints: &[cas_solver::BlockedHint],
) -> Vec<(String, String)> {
    let mut items: Vec<(String, String)> = hints
        .iter()
        .map(|hint| (format_blocked_hint_condition(ctx, hint), hint.rule.clone()))
        .collect();
    items.sort();
    items.dedup();
    items
}

fn format_blocked_simplifications_section_lines(
    ctx: &Context,
    hints: &[cas_solver::BlockedHint],
    domain_mode: cas_solver::DomainMode,
) -> Vec<String> {
    if hints.is_empty() {
        return Vec::new();
    }

    let mut lines = vec!["ℹ️ Blocked simplifications:".to_string()];
    for (cond, rule) in collect_blocked_hint_items(ctx, hints) {
        lines.push(format!("  - requires {}  [{}]", cond, rule));
    }
    lines.push(format!(
        "  tip: {}",
        blocked_hint_suggestion(domain_mode, false)
    ));
    lines
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SolveAssumptionSectionConfig {
    pub debug_mode: bool,
    pub hints_enabled: bool,
    pub domain_mode: cas_solver::DomainMode,
}

/// Render optional solve assumption/blocked sections according to CLI flags.
pub fn format_solve_assumption_and_blocked_sections(
    ctx: &Context,
    assumption_records: &[cas_solver::AssumptionRecord],
    blocked_hints: &[cas_solver::BlockedHint],
    config: SolveAssumptionSectionConfig,
) -> Vec<String> {
    let has_assumptions = !assumption_records.is_empty();
    let has_blocked = !blocked_hints.is_empty();

    if config.debug_mode && (has_assumptions || has_blocked) {
        let mut lines = vec![String::new()];
        if has_assumptions {
            lines.extend(format_assumption_records_section_lines(
                assumption_records,
                "ℹ️ Assumptions used:",
                "  - ",
            ));
        }
        if has_blocked {
            lines.extend(format_blocked_simplifications_section_lines(
                ctx,
                blocked_hints,
                config.domain_mode,
            ));
        }
        return lines;
    }

    if has_blocked && config.hints_enabled {
        let mut lines = vec![String::new()];
        lines.extend(format_blocked_simplifications_section_lines(
            ctx,
            blocked_hints,
            config.domain_mode,
        ));
        return lines;
    }

    Vec::new()
}
