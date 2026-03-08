use cas_ast::Context;

/// Render blocked hints with eval-oriented messaging.
///
/// Uses a compact single-line format when there is only one hint.
pub fn format_eval_blocked_hints_lines(
    ctx: &Context,
    hints: &[crate::BlockedHint],
    domain_mode: crate::DomainMode,
) -> Vec<String> {
    if hints.is_empty() {
        return Vec::new();
    }

    let grouped = crate::group_blocked_hint_conditions_by_rule(ctx, hints);
    let suggestion = crate::blocked_hint_suggestion(domain_mode, true);

    if grouped.len() == 1 && hints.len() == 1 {
        let hint = &hints[0];
        return vec![
            format!(
                "ℹ️  Blocked: requires {} [{}]",
                crate::format_blocked_hint_condition(ctx, hint),
                hint.rule
            ),
            format!("   {suggestion}"),
        ];
    }

    let mut lines = vec!["ℹ️  Some simplifications were blocked:".to_string()];
    for (rule, conditions) in grouped {
        lines.push(format!(" - Requires {}  [{}]", conditions.join(", "), rule));
    }
    lines.push(format!("   Tip: {suggestion}"));
    lines
}
