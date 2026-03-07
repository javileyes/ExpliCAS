use super::super::super::*;

#[test]
fn timeline_cli_actions_from_render_html_emits_io_and_lines() {
    let actions = timeline_cli_actions_from_render(TimelineCliRender::Html {
        file_name: TIMELINE_HTML_FILE,
        html: "<html/>".to_string(),
        lines: vec!["line1".to_string(), "line2".to_string()],
    });
    assert!(matches!(
        actions.first(),
        Some(TimelineCliAction::WriteFile { .. })
    ));
    assert!(matches!(
        actions.get(1),
        Some(TimelineCliAction::OpenFile { .. })
    ));
    assert!(actions
        .iter()
        .any(|action| matches!(action, TimelineCliAction::Output(line) if line == "line1")));
}
