use super::TimelineSubstepsRenderState;
use cas_formatter::html_escape;

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
        details_html.push_str(&render_enriched_substep(sub));
    }
    details_html.push_str("</div></details>");
    details_html
}

fn render_enriched_substep(sub: &crate::didactic::SubStep) -> String {
    let mut html = format!(
        r#"<div class="substep">
                                    <span class="substep-desc">{}</span>"#,
        html_escape(&sub.description)
    );
    if !sub.before_expr.is_empty() {
        html.push_str(&format!(
            r#"<div class="substep-math">\[{} \rightarrow {}\]</div>"#,
            sub.before_expr, sub.after_expr
        ));
    }
    html.push_str("</div>");
    html
}
