use crate::runtime::Step;
use cas_api_models::SubStepWire;

pub(super) fn collect_step_payload_substeps(
    step: &Step,
    enriched: &crate::didactic::EnrichedStep,
) -> Vec<SubStepWire> {
    let mut substeps: Vec<SubStepWire> = step
        .substeps()
        .iter()
        .map(|substep| SubStepWire {
            title: substep.title.clone(),
            lines: substep.lines.clone(),
            before_latex: None,
            after_latex: None,
        })
        .collect();

    for substep in &enriched.sub_steps {
        substeps.push(SubStepWire {
            title: substep.description.clone(),
            lines: vec![],
            before_latex: Some(render_substep_side(
                substep.before_latex.clone(),
                substep.before_expr.as_str(),
            )),
            after_latex: Some(render_substep_side(
                substep.after_latex.clone(),
                substep.after_expr.as_str(),
            )),
        });
    }

    substeps
}

fn render_substep_side(explicit_latex: Option<String>, fallback_expr: &str) -> String {
    explicit_latex.unwrap_or_else(|| format!("\\text{{{}}}", fallback_expr))
}
