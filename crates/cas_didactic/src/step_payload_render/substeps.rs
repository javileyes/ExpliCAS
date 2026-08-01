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
                    crate::didactic::locale::description_en(&substep.description).into_owned()
                }
                _ => substep.description.clone(),
            },
        };
        substeps.push(SubStepWire {
            title,
            before: substep.before_expr.clone(),
            after: substep.after_expr.clone(),
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

/// A substep side WITHOUT explicit LaTeX is plain display text, no matter how
/// LaTeX-ish it looks — so it is published as text, escaped.
///
/// The policy this replaces sniffed the string for `\`, `^` or `_` and, on a
/// hit, handed it to MathJax raw. The sniff cannot tell LaTeX from display
/// text, and display text is full of `^`: `y·2·x^(2 - 1)` was typeset as
/// `x^{(}`, i.e. the exponent became a lone parenthesis, in 11 rows of the
/// corpus the web serves. Guessing produced a WRONG rendering; the honest
/// fallback is an unstyled but truthful one, and the emitter that wants
/// typeset math declares it with `.with_before_latex` / `.with_after_latex`.
fn render_substep_fallback_latex(fallback_expr: &str) -> String {
    // ONE text-mode escape table for the whole engine: this used to keep a
    // second, identical copy, and both copies were wrong the same way — see
    // `cas_formatter::escape_for_text_mode` for the table MathJax 3 actually
    // implements and for the glyphs each candidate escape paints.
    format!(
        "\\text{{{}}}",
        cas_formatter::escape_for_text_mode(fallback_expr)
    )
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

    /// Replaces `render_substep_side_preserves_latexish_fallbacks`, which
    /// pinned the defect: a fallback that merely LOOKED like LaTeX was handed
    /// to MathJax raw. A fallback is display text and is published as text —
    /// an emitter that wants typeset math must declare it. Each backslash
    /// leaves `\text{…}` to be drawn in math mode, the one spelling MathJax
    /// paints as an actual backslash.
    #[test]
    fn render_substep_side_never_declares_a_fallback_to_be_latex() {
        let rendered = render_substep_side(None, "\\frac{1}{\\sqrt{x} - 1}");
        assert_eq!(
            rendered,
            "\\text{}\\unicode{x5C}\\text{frac\\{1\\}\\{}\\unicode{x5C}\\text{sqrt\\{x\\} - 1\\}}"
        );
    }

    /// The witness from the audit: display exponents used to be typeset with
    /// the opening parenthesis as the exponent (`x^{(}`). Wrapping in `\text{}`
    /// is what defuses the `^`; escaping it on TOP of that was the second
    /// defect, and it showed the reader `x\unicode{x5E}(2 - 1)`.
    #[test]
    fn render_substep_side_leaves_display_exponents_raw_inside_text_mode() {
        let rendered = render_substep_side(None, "y·2·x^(2 - 1)");
        assert_eq!(rendered, "\\text{y·2·x^(2 - 1)}");
    }

    /// An unbalanced brace does not degrade one line, it breaks the whole row.
    #[test]
    fn render_substep_side_keeps_braces_balanced() {
        let rendered = render_substep_side(None, "conjunto { 1, 2 } sin cerrar }");
        // Escape-aware, like the corpus gate's D9 detector: `\{` is a literal
        // brace, not a group delimiter.
        let mut depth: i32 = 0;
        let mut escaped = false;
        for ch in rendered.chars() {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
            assert!(depth >= 0, "closed too early: {rendered}");
        }
        assert_eq!(depth, 0, "unbalanced: {rendered}");
    }

    #[test]
    fn render_substep_side_wraps_plain_text_fallbacks() {
        let rendered = render_substep_side(None, "Expandir y agrupar");
        assert_eq!(rendered, "\\text{Expandir y agrupar}");
    }
}
