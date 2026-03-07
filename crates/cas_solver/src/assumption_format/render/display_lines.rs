use cas_ast::Context;

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
