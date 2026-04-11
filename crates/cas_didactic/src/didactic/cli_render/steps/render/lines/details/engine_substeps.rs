use crate::runtime::Step;

pub(super) fn render_engine_substeps_lines(_step: &Step) -> Vec<String> {
    // Legacy engine substeps only carry free-form narrative lines.
    // User-facing displays now standardize on structured didactic substeps
    // with explicit before/after expressions.
    Vec::new()
}
