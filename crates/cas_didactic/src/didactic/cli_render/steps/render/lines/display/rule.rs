use cas_ast::{Context, ExprId};
use cas_solver::Step;

pub(super) fn render_rule_with_scope_line(
    ctx: &mut Context,
    step: &Step,
    style_prefs: &cas_formatter::root_style::StylePreferences,
    local_rule_expr_ids: fn(&Step) -> (ExprId, ExprId),
    render_expr: fn(&mut Context, ExprId, &cas_formatter::root_style::StylePreferences) -> String,
) -> String {
    let (rule_before_id, rule_after_id) = local_rule_expr_ids(step);
    let before_disp =
        cas_formatter::clean_display_string(&render_expr(ctx, rule_before_id, style_prefs));
    let after_disp = cas_formatter::clean_display_string(&cas_formatter::render_with_rule_scope(
        ctx,
        rule_after_id,
        &step.rule_name,
        style_prefs,
    ));

    format!("   Rule: {} -> {}", before_disp, after_disp)
}
