mod item;
mod policy;

use super::TimelineSubstepsRenderState;

pub(super) fn render_timeline_enriched_substeps_html(
    enriched: &crate::didactic::EnrichedStep,
    state: &mut TimelineSubstepsRenderState,
) -> String {
    if enriched.sub_steps.is_empty() {
        return String::new();
    }

    let render_plan = crate::didactic::build_timeline_substeps_render_plan(&enriched.sub_steps);
    if !policy::should_render_enriched_substeps(enriched, state) {
        return String::new();
    }

    policy::update_enriched_substeps_state(&render_plan, state);

    let mut details_html = format!(
        r#"<details class="substeps-details">
                            <summary>{}</summary>
                            <div class="substeps-content">"#,
        render_plan.header
    );
    for sub in &enriched.sub_steps {
        details_html.push_str(&item::render_enriched_substep(sub));
    }
    details_html.push_str("</div></details>");
    details_html
}
