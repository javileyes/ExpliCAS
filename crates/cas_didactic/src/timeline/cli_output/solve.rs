use super::super::render_api::render_solve_timeline_html;
use super::super::solve_result_display::{
    format_timeline_solve_no_steps_message, format_timeline_solve_result_line,
};
use super::super::types::{TimelineCliRender, TimelineSolveCommandOutput};
use super::{TIMELINE_HTML_FILE, TIMELINE_OPEN_HINT_MESSAGE};
use cas_ast::Context;

/// Build CLI render output for solve timeline command.
pub fn render_solve_timeline_cli_output(
    context: &mut Context,
    out: &TimelineSolveCommandOutput,
) -> TimelineCliRender {
    if out.display_steps.0.is_empty() {
        return TimelineCliRender::NoSteps {
            lines: vec![format_timeline_solve_no_steps_message(
                context,
                &out.solution_set,
            )],
        };
    }

    let html = render_solve_timeline_html(
        context,
        &out.display_steps.0,
        &out.equation,
        &out.solution_set,
        &out.var,
    );
    let lines = vec![
        format_timeline_solve_result_line(context, &out.solution_set),
        TIMELINE_OPEN_HINT_MESSAGE.to_string(),
    ];

    TimelineCliRender::Html {
        file_name: TIMELINE_HTML_FILE,
        html,
        lines,
    }
}
