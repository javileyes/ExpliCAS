mod filter;

use cas_ast::{Context, ExprId};

use super::normalized::format_normalized_condition_lines;

/// Render display lines for `Diagnostics::requires` after witness filtering.
pub fn format_diagnostics_requires_lines(
    ctx: &mut Context,
    diagnostics: &crate::Diagnostics,
    result_expr: Option<ExprId>,
    display_level: crate::RequiresDisplayLevel,
    debug_mode: bool,
) -> Vec<String> {
    let filtered = filter::filter_diagnostic_requires(ctx, diagnostics, result_expr, display_level);

    if filtered.is_empty() {
        return Vec::new();
    }

    let conditions: Vec<_> = filtered.iter().map(|item| item.cond.clone()).collect();
    format_normalized_condition_lines(ctx, &conditions, debug_mode)
}
