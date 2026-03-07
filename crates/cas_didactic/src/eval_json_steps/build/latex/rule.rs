use cas_ast::Context;
use cas_solver::Step;

pub(super) fn render_step_rule_latex(context: &Context, step: &Step) -> String {
    let focus_before = step.before_local().unwrap_or(step.before);
    let focus_after = step.after_local().unwrap_or(step.after);
    crate::eval_json_render::render_local_rule_latex(context, focus_before, focus_after)
}
