use super::super::render_api::render_simplify_timeline_html;
use super::super::simplify::VerbosityLevel;
use super::super::types::{TimelineCliRender, TimelineSimplifyCommandOutput};
use super::{timeline_simplify_info_lines, TIMELINE_HTML_FILE, TIMELINE_NO_STEPS_MESSAGE};
use cas_ast::Context;

/// Build CLI render output for simplify timeline command.
pub fn render_simplify_timeline_cli_output(
    context: &mut Context,
    out: &TimelineSimplifyCommandOutput,
    verbosity: VerbosityLevel,
) -> TimelineCliRender {
    if out.steps.is_empty() {
        return TimelineCliRender::NoSteps {
            lines: vec![TIMELINE_NO_STEPS_MESSAGE.to_string()],
        };
    }

    let html = render_simplify_timeline_html(
        context,
        &out.steps,
        out.parsed_expr,
        Some(out.simplified_expr),
        verbosity,
        Some(out.expr_input.as_str()),
    );
    let lines = timeline_simplify_info_lines(out.use_aggressive);

    TimelineCliRender::Html {
        file_name: TIMELINE_HTML_FILE,
        html,
        lines,
    }
}
