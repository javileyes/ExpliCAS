mod shared;
mod simplify;
mod solve;

use super::simplify::VerbosityLevel;
use super::types::{TimelineCliRender, TimelineCommandOutput};

pub use self::simplify::render_simplify_timeline_cli_output;
pub use self::solve::render_solve_timeline_cli_output;

/// Canonical output file name used by timeline CLI render helpers.
pub const TIMELINE_HTML_FILE: &str = "timeline.html";
const TIMELINE_NO_STEPS_MESSAGE: &str = "No simplification steps to visualize.";
const TIMELINE_OPEN_HINT_MESSAGE: &str = "Open in browser to view interactive visualization.";

fn timeline_simplify_info_lines(use_aggressive: bool) -> Vec<String> {
    let mut lines = Vec::new();
    if use_aggressive {
        lines.push("(Aggressive simplification mode)".to_string());
    }
    lines.push(TIMELINE_OPEN_HINT_MESSAGE.to_string());
    lines
}

/// Build CLI render output for a full `timeline` command eval output.
pub fn render_timeline_command_cli_output(
    context: &mut cas_ast::Context,
    out: &TimelineCommandOutput,
    verbosity: VerbosityLevel,
) -> TimelineCliRender {
    match out {
        TimelineCommandOutput::Solve(solve_out) => {
            render_solve_timeline_cli_output(context, solve_out)
        }
        TimelineCommandOutput::Simplify(simplify_out) => {
            render_simplify_timeline_cli_output(context, simplify_out, verbosity)
        }
    }
}
