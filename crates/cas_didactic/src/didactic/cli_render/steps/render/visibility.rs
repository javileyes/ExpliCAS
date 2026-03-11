use super::local_rule_expr_ids;
use crate::runtime::Step;
use cas_ast::Context;
use cas_formatter::DisplayExprStyled;

pub(crate) fn render_step_visible_change(
    ctx: &mut Context,
    step: &Step,
    style_prefs: &cas_formatter::root_style::StylePreferences,
) -> bool {
    let before_disp = cas_formatter::clean_display_string(&format!(
        "{}",
        DisplayExprStyled::new(ctx, step.before, style_prefs)
    ));
    let after_disp = cas_formatter::clean_display_string(&format!(
        "{}",
        DisplayExprStyled::new(ctx, step.after, style_prefs)
    ));
    before_disp != after_disp
}

pub(crate) fn render_rule_visible_change(
    ctx: &mut Context,
    step: &Step,
    style_prefs: &cas_formatter::root_style::StylePreferences,
) -> bool {
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

    before_disp != after_disp
}
