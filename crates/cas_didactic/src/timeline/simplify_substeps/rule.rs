use crate::runtime::Step;

pub(super) fn render_timeline_rule_substeps_html(_step: &Step) -> String {
    // Legacy engine substeps are free-form narrative lines.
    // Timeline rendering now standardizes on structured enriched substeps only.
    String::new()
}
