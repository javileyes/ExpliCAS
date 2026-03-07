use cas_ast::Context;

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
