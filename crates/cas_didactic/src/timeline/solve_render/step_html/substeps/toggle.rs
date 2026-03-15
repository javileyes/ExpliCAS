pub(super) fn render_substeps_toggle_html(substep_id: &str, len: usize) -> String {
    let len_text = len.to_string();
    super::super::super::super::render_template::render_static_template(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/timeline/solve_render/substeps_toggle.html"
        )),
        &[
            ("__SUBSTEP_ID__", substep_id),
            ("__LEN__", len_text.as_str()),
        ],
    )
}
