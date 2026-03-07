use cas_api_models::SubStepJson;
use cas_solver::Step;

pub(super) fn collect_step_json_substeps(
    step: &Step,
    enriched: &crate::didactic::EnrichedStep,
) -> Vec<SubStepJson> {
    let mut substeps: Vec<SubStepJson> = step
        .substeps()
        .iter()
        .map(|substep| SubStepJson {
            title: substep.title.clone(),
            lines: substep.lines.clone(),
            before_latex: None,
            after_latex: None,
        })
        .collect();

    for substep in &enriched.sub_steps {
        substeps.push(SubStepJson {
            title: substep.description.clone(),
            lines: vec![],
            before_latex: Some(render_substep_side(
                substep.before_latex.clone(),
                &substep.before_expr,
            )),
            after_latex: Some(render_substep_side(
                substep.after_latex.clone(),
                &substep.after_expr,
            )),
        });
    }

    substeps
}

fn render_substep_side(explicit_latex: Option<String>, fallback_expr: &str) -> String {
    explicit_latex.unwrap_or_else(|| format!("\\text{{{}}}", fallback_expr))
}
