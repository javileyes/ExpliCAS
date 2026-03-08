use super::super::super::steps::{
    next_step_root, push_detailed_step_lines, push_succinct_step_line,
};
use super::super::super::types::FullSimplifyDisplayMode;
use super::super::super::visibility::should_show_simplify_step;

pub(super) fn append_step_loop_lines(
    lines: &mut Vec<String>,
    ctx: &mut cas_ast::Context,
    resolved_expr: cas_ast::ExprId,
    steps: &[crate::Step],
    mode: FullSimplifyDisplayMode,
    style_prefs: &cas_formatter::StylePreferences,
) {
    let mut current_root = resolved_expr;
    let mut step_count = 0;

    for step in steps {
        if should_show_simplify_step(step, mode) {
            step_count += 1;
            current_root = append_visible_step_lines(
                lines,
                ctx,
                step,
                mode,
                style_prefs,
                step_count,
                current_root,
            );
        } else {
            current_root = next_step_root(ctx, current_root, step);
        }
    }
}

fn append_visible_step_lines(
    lines: &mut Vec<String>,
    ctx: &mut cas_ast::Context,
    step: &crate::Step,
    mode: FullSimplifyDisplayMode,
    style_prefs: &cas_formatter::StylePreferences,
    step_count: usize,
    current_root: cas_ast::ExprId,
) -> cas_ast::ExprId {
    if mode == FullSimplifyDisplayMode::Succinct {
        let next_root = next_step_root(ctx, current_root, step);
        push_succinct_step_line(lines, ctx, next_root);
        next_root
    } else {
        push_detailed_step_lines(lines, ctx, step, style_prefs, step_count, current_root)
    }
}
