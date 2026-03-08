use cas_ast::{Context, ExprId};

pub(super) fn filter_diagnostic_requires<'a>(
    ctx: &mut Context,
    diagnostics: &'a crate::Diagnostics,
    result_expr: Option<ExprId>,
    display_level: crate::RequiresDisplayLevel,
) -> Vec<&'a crate::RequiredItem> {
    if let Some(result) = result_expr {
        diagnostics.filter_requires_for_display(ctx, result, display_level)
    } else {
        diagnostics.requires.iter().collect()
    }
}
