use super::*;
use cas_ast::{Context, Expr};
use cas_solver::{to_display_steps, Step};

#[test]
fn test_html_generation() {
    let mut ctx = Context::new();
    let two = ctx.num(2);
    let three = ctx.num(3);
    let add_expr = ctx.add(Expr::Add(two, three));
    let five = ctx.num(5);

    let steps = vec![Step::new(
        "2 + 3 = 5",
        "Combine Constants",
        add_expr,
        five,
        vec![],
        Some(&ctx),
    )];

    let mut timeline = TimelineHtml::new(&mut ctx, &steps, add_expr, VerbosityLevel::Verbose);
    let html = timeline.to_html();

    assert!(html.contains("<!DOCTYPE html"));
    assert!(html.contains("timeline"));
    assert!(html.contains("CAS Simplification"));
    assert!(html.contains("Combine Constants"));
}

#[test]
fn test_html_escape() {
    assert_eq!(html_escape("<script>"), "&lt;script&gt;");
    assert_eq!(html_escape("x & y"), "x &amp; y");
}

#[test]
fn render_simplify_timeline_helper_produces_document() {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let y = ctx.var("y");
    let add = ctx.add(cas_ast::Expr::Add(x, y));
    let steps = vec![];
    let html = render_simplify_timeline_html(
        &mut ctx,
        &steps,
        add,
        Some(add),
        VerbosityLevel::Normal,
        Some("x+y"),
    );
    assert!(html.contains("<!DOCTYPE html"));
}

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
