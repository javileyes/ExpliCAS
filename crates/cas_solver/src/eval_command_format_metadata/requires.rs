mod result;

use crate::eval_command_types::EvalCommandEvalView;

pub(super) fn format_requires_lines(
    context: &mut cas_ast::Context,
    output: &EvalCommandEvalView,
    requires_display: crate::RequiresDisplayLevel,
    debug_mode: bool,
) -> Vec<String> {
    if output.diagnostics.requires.is_empty() {
        return Vec::new();
    }

    let result_expr = result::eval_result_expr(&output.result);
    let rendered = crate::assumption_format::format_diagnostics_requires_lines(
        context,
        &output.diagnostics,
        result_expr,
        requires_display,
        debug_mode,
    );

    if rendered.is_empty() {
        return Vec::new();
    }

    let mut requires_lines = vec!["ℹ️ Requires:".to_string()];
    requires_lines.extend(rendered);
    requires_lines
}
