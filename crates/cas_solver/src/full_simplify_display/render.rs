mod prepare;
mod result;
mod step_loop;

use super::types::FullSimplifyDisplayMode;

/// Format full simplify output lines according to display mode.
pub fn format_full_simplify_eval_lines(
    ctx: &mut cas_ast::Context,
    expr_input: &str,
    resolved_expr: cas_ast::ExprId,
    simplified_expr: cas_ast::ExprId,
    steps: &[crate::Step],
    mode: FullSimplifyDisplayMode,
) -> Vec<String> {
    let (mut lines, style_prefs) =
        prepare::prepare_full_simplify_render(ctx, expr_input, resolved_expr);
    step_loop::append_full_simplify_step_lines(
        &mut lines,
        ctx,
        resolved_expr,
        steps,
        mode,
        &style_prefs,
    );
    result::append_full_simplify_result_line(&mut lines, ctx, simplified_expr, &style_prefs);
    lines
}
