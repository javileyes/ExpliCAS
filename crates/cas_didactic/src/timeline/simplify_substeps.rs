use cas_formatter::html_escape;
use cas_solver::Step;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct TimelineSubstepsRenderState {
    enriched_dedupe_shown: bool,
}

pub(super) fn render_timeline_enriched_substeps_html(
    enriched: &crate::didactic::EnrichedStep,
    state: &mut TimelineSubstepsRenderState,
) -> String {
    if enriched.sub_steps.is_empty() {
        return String::new();
    }

    let classification = crate::didactic::classify_sub_steps(&enriched.sub_steps);
    let render_plan = crate::didactic::build_timeline_substeps_render_plan(&enriched.sub_steps);
    let should_show = if classification.has_nested_fraction || classification.has_factorization {
        true
    } else {
        !state.enriched_dedupe_shown
    };

    if !should_show {
        return String::new();
    }

    if render_plan.dedupe_once {
        state.enriched_dedupe_shown = true;
    }

    let mut details_html = format!(
        r#"<details class="substeps-details">
                            <summary>{}</summary>
                            <div class="substeps-content">"#,
        render_plan.header
    );
    for sub in &enriched.sub_steps {
        details_html.push_str(&format!(
            r#"<div class="substep">
                                    <span class="substep-desc">{}</span>"#,
            html_escape(&sub.description)
        ));
        if !sub.before_expr.is_empty() {
            details_html.push_str(&format!(
                r#"<div class="substep-math">\[{} \rightarrow {}\]</div>"#,
                sub.before_expr, sub.after_expr
            ));
        }
        details_html.push_str("</div>");
    }
    details_html.push_str("</div></details>");
    details_html
}

pub(super) fn render_timeline_rule_substeps_html(step: &Step) -> String {
    if step.substeps().is_empty() {
        return String::new();
    }

    let mut details_html = String::from(
        r#"<details class="substeps-details" open>
                    <summary>Pasos didácticos</summary>
                    <div class="substeps-content">"#,
    );
    for substep in step.substeps() {
        details_html.push_str(&format!(
            r#"<div class="substep">
                            <strong>[{}]</strong>"#,
            html_escape(&substep.title)
        ));
        for line in &substep.lines {
            details_html.push_str(&format!(
                r#"<div class="substep-line">• {}</div>"#,
                html_escape(line)
            ));
        }
        details_html.push_str("</div>");
    }
    details_html.push_str("</div></details>");
    details_html
}

pub(super) fn render_timeline_domain_assumptions_html(step: &Step) -> String {
    let grouped_lines = cas_solver::format_displayable_assumption_lines_grouped_for_step(step);
    if grouped_lines.is_empty() {
        return String::new();
    }

    let parts: Vec<String> = grouped_lines.iter().map(|line| html_escape(line)).collect();
    format!(
        r#"                    <div class="domain-assumptions">{}</div>
"#,
        parts.join("<br/>")
    )
}
