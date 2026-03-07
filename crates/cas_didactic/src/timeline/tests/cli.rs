use super::super::*;
use cas_ast::Context;
use cas_solver::to_display_steps;

#[test]
fn render_timeline_command_cli_output_simplify_no_steps() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let out = TimelineCommandOutput::Simplify(TimelineSimplifyCommandOutput {
        expr_input: "x".to_string(),
        use_aggressive: false,
        parsed_expr: x,
        simplified_expr: x,
        steps: to_display_steps(Vec::new()),
    });

    let render = render_timeline_command_cli_output(&mut ctx, &out, VerbosityLevel::Normal);
    match render {
        TimelineCliRender::NoSteps { lines } => assert!(!lines.is_empty()),
        TimelineCliRender::Html { .. } => panic!("expected no-steps render"),
    }
}

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

#[test]
fn extract_timeline_invocation_input_strips_prefix() {
    assert_eq!(extract_timeline_invocation_input("timeline x+1"), "x+1");
    assert_eq!(
        extract_timeline_invocation_input("timeline solve x+1=2,x"),
        "solve x+1=2,x"
    );
    assert_eq!(extract_timeline_invocation_input("x+1"), "x+1");
}

#[test]
fn evaluate_timeline_invocation_cli_actions_with_session_returns_actions() {
    let mut engine = cas_solver::Engine::new();
    let mut session = cas_session::SessionState::new();
    let options = cas_solver::EvalOptions::default();
    let actions = evaluate_timeline_invocation_cli_actions_with_session(
        &mut engine,
        &mut session,
        "timeline x+1",
        &options,
        VerbosityLevel::Normal,
    )
    .expect("timeline eval");
    assert!(!actions.is_empty());
}
