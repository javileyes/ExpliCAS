//! Formatting helpers for history/show metadata sections.

/// Render required/assumed/blocked metadata sections for history entry display.
pub fn format_history_eval_metadata_sections(
    ctx: &cas_ast::Context,
    required_conditions: &[crate::ImplicitCondition],
    domain_warnings: &[crate::DomainWarning],
    blocked_hints: &[crate::BlockedHint],
) -> Vec<String> {
    let mut lines = Vec::new();

    if !required_conditions.is_empty() {
        lines.push("  ℹ️ Requires:".to_string());
        lines.extend(crate::format_required_condition_lines(
            ctx,
            required_conditions,
            "    - ",
        ));
    }

    if !domain_warnings.is_empty() {
        lines.push("  ⚠ Assumed:".to_string());
        lines.extend(crate::format_domain_warning_lines(
            domain_warnings,
            false,
            "    - ",
        ));
    }

    if !blocked_hints.is_empty() {
        lines.push("  🚫 Blocked:".to_string());
        lines.extend(crate::format_blocked_hint_lines(blocked_hints, "    - "));
    }

    lines
}
