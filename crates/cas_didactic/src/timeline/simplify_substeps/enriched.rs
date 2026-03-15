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
    let summary = render_plan.header;

    let mut content_html = String::new();
    for sub in &enriched.sub_steps {
        content_html.push_str(&item::render_enriched_substep(sub));
    }

    super::super::render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/simplify_render/substeps_details.html"
        )),
        &[
            ("__OPEN_ATTR__", ""),
            ("__SUMMARY__", summary),
            ("__CONTENT__", content_html.as_str()),
        ],
    )
}
