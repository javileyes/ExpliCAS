//! Formatting helpers for history/show metadata sections.

use std::collections::BTreeMap;

/// Render required/assumed/blocked metadata sections for history entry display.
pub(crate) fn format_history_eval_metadata_sections(
    ctx: &cas_ast::Context,
    required_conditions: &[crate::ImplicitCondition],
    domain_warnings: &[crate::DomainWarning],
    blocked_hints: &[crate::BlockedHint],
) -> Vec<String> {
    let mut lines = Vec::new();

    if !required_conditions.is_empty() {
        lines.push("  ℹ️ Requires:".to_string());
        lines.extend(crate::assumption_format::format_required_condition_lines(
            ctx,
            required_conditions,
            "    - ",
        ));
    }

    if !domain_warnings.is_empty() {
        lines.push("  ⚠ Assumed:".to_string());
        lines.extend(crate::assumption_format::format_domain_warning_lines(
            domain_warnings,
            false,
            "    - ",
        ));
    }

    if !blocked_hints.is_empty() {
        lines.push("  🚫 Blocked:".to_string());
        lines.extend(format_history_blocked_hint_lines(ctx, blocked_hints));
    }

    lines
}

fn format_history_blocked_hint_lines(
    ctx: &cas_ast::Context,
    blocked_hints: &[crate::BlockedHint],
) -> Vec<String> {
    let mut grouped: BTreeMap<(String, &'static str), Vec<String>> = BTreeMap::new();
    for hint in blocked_hints {
        let condition = crate::format_blocked_hint_condition(ctx, hint);
        let rules = grouped.entry((condition, hint.suggestion)).or_default();
        if !rules.iter().any(|rule| rule == &hint.rule) {
            rules.push(hint.rule.clone());
        }
    }

    let mut lines = Vec::new();
    for ((condition, suggestion), rules) in grouped {
        lines.push(format!(
            "    - requires {} [{}]",
            condition,
            rules.join(", ")
        ));
        lines.push(format!("      tip: {suggestion}"));
    }
    lines
}
