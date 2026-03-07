mod iterate;
mod prepare;

use super::super::types::FullSimplifyDisplayMode;

pub(super) fn append_full_simplify_step_lines(
    lines: &mut Vec<String>,
    ctx: &mut cas_ast::Context,
    resolved_expr: cas_ast::ExprId,
    steps: &[crate::Step],
    mode: FullSimplifyDisplayMode,
    style_prefs: &cas_formatter::StylePreferences,
) {
    if !prepare::prepare_step_lines(lines, steps, mode) {
        return;
    }

    iterate::append_step_loop_lines(lines, ctx, resolved_expr, steps, mode, style_prefs);
}
