use cas_ast::{Context, ExprId};

/// Render required-conditions as plain display lines using a line prefix.
pub fn format_required_condition_lines(
    ctx: &Context,
    conditions: &[cas_solver::ImplicitCondition],
    line_prefix: &str,
) -> Vec<String> {
    conditions
        .iter()
        .map(|cond| format!("{line_prefix}{}", cond.display(ctx)))
        .collect()
}

/// Render domain warnings as display lines using a line prefix.
///
/// When `include_rule` is true, appends `(from <rule>)`.
pub fn format_domain_warning_lines(
    warnings: &[cas_solver::DomainWarning],
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
pub fn format_blocked_hint_lines(
    hints: &[cas_solver::BlockedHint],
    line_prefix: &str,
) -> Vec<String> {
    hints
        .iter()
        .map(|hint| format!("{line_prefix}{} (hint: {})", hint.rule, hint.suggestion))
        .collect()
}

/// Render normalized required-conditions as REPL bullet lines.
pub fn format_normalized_condition_lines(
    ctx: &mut Context,
    conditions: &[cas_solver::ImplicitCondition],
    debug_mode: bool,
) -> Vec<String> {
    let normalized_conditions = cas_solver::normalize_and_dedupe_conditions(ctx, conditions);
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
    diagnostics: &cas_solver::Diagnostics,
    result_expr: Option<ExprId>,
    display_level: cas_solver::RequiresDisplayLevel,
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
