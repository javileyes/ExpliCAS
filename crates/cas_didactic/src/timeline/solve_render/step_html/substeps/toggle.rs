pub(super) fn render_substeps_toggle_html(substep_id: &str, len: usize) -> String {
    format!(
        r#"            <div class="substeps-toggle" onclick="toggleSubsteps('{}')">
                <span class="arrow">▶</span>
                <span>Show derivation ({} steps)</span>
            </div>
            <div id="{}" class="substeps-container">
"#,
        substep_id, len, substep_id
    )
}
