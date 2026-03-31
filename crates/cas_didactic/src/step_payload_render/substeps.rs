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
    explicit_latex.unwrap_or_else(|| render_substep_fallback_latex(fallback_expr))
}

fn render_substep_fallback_latex(fallback_expr: &str) -> String {
    if looks_like_math_latex(fallback_expr) {
        fallback_expr.to_string()
    } else {
        format!("\\text{{{}}}", fallback_expr)
    }
}

fn looks_like_math_latex(fallback_expr: &str) -> bool {
    let trimmed = fallback_expr.trim();
    if trimmed.is_empty() {
        return false;
    }

    trimmed.chars().any(|ch| matches!(ch, '\\' | '^' | '_'))
}

#[cfg(test)]
mod tests {
    use super::render_substep_side;

    #[test]
    fn render_substep_side_preserves_explicit_latex() {
        let rendered = render_substep_side(
            Some("\\frac{1}{\\sqrt{x} - 1}".to_string()),
            "1 / (sqrt(x) - 1)",
        );
        assert_eq!(rendered, "\\frac{1}{\\sqrt{x} - 1}");
    }

    #[test]
    fn render_substep_side_preserves_latexish_fallbacks() {
        let rendered = render_substep_side(None, "\\frac{1}{\\sqrt{x} - 1}");
        assert_eq!(rendered, "\\frac{1}{\\sqrt{x} - 1}");
    }

    #[test]
    fn render_substep_side_wraps_plain_text_fallbacks() {
        let rendered = render_substep_side(None, "Expandir y agrupar");
        assert_eq!(rendered, "\\text{Expandir y agrupar}");
    }
}
