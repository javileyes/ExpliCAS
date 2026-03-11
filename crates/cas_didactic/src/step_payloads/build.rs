mod expr;
mod latex;
mod substeps;

use cas_api_models::StepWire;
use cas_ast::Context;

pub(super) fn build_step_wire(
    context: &Context,
    index: usize,
    enriched: &crate::didactic::EnrichedStep,
) -> StepWire {
    let step = &enriched.base_step;
    let rendered_exprs = expr::render_step_wire_exprs(context, step);
    let rendered_latex = latex::render_step_wire_latex(context, step);
    let substeps = substeps::collect_step_wire_substeps(step, enriched);

    StepWire {
        index,
        rule: step.rule_name.to_string(),
        rule_latex: rendered_latex.rule_latex,
        before: rendered_exprs.before,
        after: rendered_exprs.after,
        before_latex: rendered_latex.before_latex,
        after_latex: rendered_latex.after_latex,
        substeps,
    }
}
