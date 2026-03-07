use cas_ast::{Context, ExprId};
use cas_formatter::DisplayExprStyled;
use cas_solver::Step;

pub(super) fn render_step_header(step_count: usize, step: &Step) -> String {
    format!("{}. {}  [{}]", step_count, step.description, step.rule_name)
}

pub(super) fn render_before_line(
    ctx: &mut Context,
    before_expr: ExprId,
    style_prefs: &cas_formatter::root_style::StylePreferences,
) -> String {
    format!(
        "   Before: {}",
        cas_formatter::clean_display_string(&format!(
            "{}",
            DisplayExprStyled::new(ctx, before_expr, style_prefs)
        ))
    )
}

pub(super) fn render_rule_with_scope_line(
    ctx: &mut Context,
    step: &Step,
    style_prefs: &cas_formatter::root_style::StylePreferences,
    local_rule_expr_ids: fn(&Step) -> (ExprId, ExprId),
) -> String {
    let (rule_before_id, rule_after_id) = local_rule_expr_ids(step);
    let before_disp = cas_formatter::clean_display_string(&format!(
        "{}",
        DisplayExprStyled::new(ctx, rule_before_id, style_prefs)
    ));
    let after_disp = cas_formatter::clean_display_string(&cas_formatter::render_with_rule_scope(
        ctx,
        rule_after_id,
        &step.rule_name,
        style_prefs,
    ));

    format!("   Rule: {} -> {}", before_disp, after_disp)
}

pub(super) fn render_after_line(
    ctx: &mut Context,
    after_expr: ExprId,
    style_prefs: &cas_formatter::root_style::StylePreferences,
) -> String {
    format!(
        "   After: {}",
        cas_formatter::clean_display_string(&format!(
            "{}",
            DisplayExprStyled::new(ctx, after_expr, style_prefs)
        ))
    )
}
