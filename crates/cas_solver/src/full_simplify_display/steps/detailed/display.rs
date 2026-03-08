use cas_ast::{Context, ExprId};

use super::super::root::next_step_root;

pub(super) fn push_detailed_display_lines(
    lines: &mut Vec<String>,
    ctx: &mut Context,
    step: &crate::Step,
    style_prefs: &cas_formatter::StylePreferences,
    step_count: usize,
    current_root: ExprId,
) -> ExprId {
    lines.push(format!(
        "{}. {}  [{}]",
        step_count, step.description, step.rule_name
    ));

    let before_root = step.global_before.unwrap_or(current_root);
    lines.push(format!(
        "   Before: {}",
        cas_formatter::clean_display_string(&format!(
            "{}",
            cas_formatter::DisplayExprStyled::new(ctx, before_root, style_prefs)
        ))
    ));

    let (rule_before_id, rule_after_id) = match (step.before_local(), step.after_local()) {
        (Some(bl), Some(al)) => (bl, al),
        _ => (step.before, step.after),
    };

    let before_disp = cas_formatter::clean_display_string(&format!(
        "{}",
        cas_formatter::DisplayExprStyled::new(ctx, rule_before_id, style_prefs)
    ));
    let after_disp = cas_formatter::clean_display_string(&cas_formatter::render_with_rule_scope(
        ctx,
        rule_after_id,
        &step.rule_name,
        style_prefs,
    ));
    lines.push(format!("   Rule: {} -> {}", before_disp, after_disp));

    let next_root = next_step_root(ctx, current_root, step);
    lines.push(format!(
        "   After: {}",
        cas_formatter::clean_display_string(&format!(
            "{}",
            cas_formatter::DisplayExprStyled::new(ctx, next_root, style_prefs)
        ))
    ));

    next_root
}
