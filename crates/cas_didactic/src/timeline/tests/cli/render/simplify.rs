use super::super::super::super::*;
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
