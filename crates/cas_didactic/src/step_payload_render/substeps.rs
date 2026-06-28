use crate::runtime::Step;
use cas_api_models::SubStepWire;

pub(super) fn collect_step_payload_substeps(
    _step: &Step,
    enriched: &crate::didactic::EnrichedStep,
    language: cas_solver_core::eval_option_axes::Language,
) -> Vec<SubStepWire> {
    let mut substeps: Vec<SubStepWire> = Vec::new();

    for substep in &enriched.sub_steps {
        // A keyed sub-step renders its title in `language` from the locale table; an unkeyed one keeps
        // its Spanish `description` (sub-steps are migrated to keys incrementally).
        let title = match substep.desc_key {
            Some(key) => {
                let args: Vec<&str> = substep.desc_args.iter().map(String::as_str).collect();
                crate::didactic::locale::translate(key, &args, language)
            }
            None => match language {
                cas_solver_core::eval_option_axes::Language::En => {
                    crate::didactic::locale::description_en(&substep.description).to_string()
                }
                _ => substep.description.clone(),
            },
        };
        substeps.push(SubStepWire {
            title,
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
