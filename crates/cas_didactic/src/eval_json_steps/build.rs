mod expr;
mod latex;
mod substeps;

use cas_api_models::StepJson;
use cas_ast::Context;

pub(super) fn build_step_json(
    context: &Context,
    index: usize,
    enriched: &crate::didactic::EnrichedStep,
) -> StepJson {
    let step = &enriched.base_step;
    let rendered_exprs = expr::render_step_json_exprs(context, step);
    let rendered_latex = latex::render_step_json_latex(context, step);
    let substeps = substeps::collect_step_json_substeps(step, enriched);

    StepJson {
        index,
        rule: step.rule_name.clone(),
        rule_latex: rendered_latex.rule_latex,
        before: rendered_exprs.before,
        after: rendered_exprs.after,
        before_latex: rendered_latex.before_latex,
        after_latex: rendered_latex.after_latex,
        substeps,
    }
}
