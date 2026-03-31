use super::super::{latex_to_plain_text, EnrichedStep};

pub(super) fn render_cli_enriched_substeps_lines(
    enriched_step: &EnrichedStep,
    header: Option<&'static str>,
) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(header) = header {
        lines.push(format!("   {}", header));
    }
    for sub in &enriched_step.sub_steps {
        lines.push(format!("      → {}", sub.description));
        let before = sub
            .before_latex
            .as_deref()
            .unwrap_or(sub.before_expr.as_str());
        let after = sub
            .after_latex
            .as_deref()
            .unwrap_or(sub.after_expr.as_str());
        if !before.is_empty() {
            lines.push(format!(
                "        {} → {}",
                latex_to_plain_text(before),
                latex_to_plain_text(after)
            ));
        }
    }

    lines
}
