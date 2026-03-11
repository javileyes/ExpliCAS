use crate::blocked_hint::BlockedHint;
use crate::diagnostics_model::{Diagnostics, RequiredItem};
use crate::domain_condition::{ImplicitCondition, RequiresDisplayLevel};
use crate::domain_normalization::normalize_and_dedupe_conditions;
use crate::domain_warning::DomainWarning;
use cas_ast::{Context, ExprId};

/// Render required conditions as plain display lines with a custom prefix.
pub fn format_required_condition_lines(
    ctx: &Context,
    conditions: &[ImplicitCondition],
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
    warnings: &[DomainWarning],
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
pub fn format_blocked_hint_lines(hints: &[BlockedHint], line_prefix: &str) -> Vec<String> {
    hints
        .iter()
        .map(|hint| format!("{line_prefix}{} (hint: {})", hint.rule, hint.suggestion))
        .collect()
}

/// Render normalized required conditions as REPL bullet lines.
pub fn format_normalized_condition_lines(
    ctx: &mut Context,
    conditions: &[ImplicitCondition],
    debug_mode: bool,
) -> Vec<String> {
    let normalized_conditions = normalize_and_dedupe_conditions(ctx, conditions);
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
    diagnostics: &Diagnostics,
    result_expr: Option<ExprId>,
    display_level: RequiresDisplayLevel,
    debug_mode: bool,
) -> Vec<String> {
    let filtered = filter_diagnostic_requires(ctx, diagnostics, result_expr, display_level);

    if filtered.is_empty() {
        return Vec::new();
    }

    let conditions: Vec<_> = filtered.iter().map(|item| item.cond.clone()).collect();
    format_normalized_condition_lines(ctx, &conditions, debug_mode)
}

fn filter_diagnostic_requires<'a>(
    ctx: &mut Context,
    diagnostics: &'a Diagnostics,
    result_expr: Option<ExprId>,
    display_level: RequiresDisplayLevel,
) -> Vec<&'a RequiredItem> {
    if let Some(result) = result_expr {
        diagnostics.filter_requires_for_display(ctx, result, display_level)
    } else {
        diagnostics.requires.iter().collect()
    }
}
