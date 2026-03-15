//! Timeline HTML generation module.
//!
//! Provides interactive HTML visualizations for:
//! - Expression simplification steps ([`TimelineHtml`])
//! - Equation solving steps ([`SolveTimelineHtml`])

#[path = "types/actions.rs"]
mod actions;
mod cli_actions;
mod cli_output;
#[path = "types/command.rs"]
mod command;
mod command_eval;
mod command_projection;
mod page_shell;
mod page_theme_css;
#[path = "types/render.rs"]
mod render;
mod render_api;
mod render_template;
mod simplify;
mod simplify_highlights;
mod simplify_init;
mod simplify_page;
mod simplify_render;
mod simplify_step_html;
mod simplify_substeps;
mod simplify_summary;
mod solve;
mod solve_init;
mod solve_page;
mod solve_render;
mod solve_result_display;
mod solve_solution_latex;
mod solve_timeline_render;

pub use actions::TimelineCliAction;
pub use cas_formatter::{html_escape, latex_escape};
pub use cli_actions::timeline_cli_actions_from_render;
pub use cli_output::{
    render_simplify_timeline_cli_output, render_solve_timeline_cli_output,
    render_timeline_command_cli_output, TIMELINE_HTML_FILE,
};
pub use command::{
    TimelineCommandOutput, TimelineSimplifyCommandOutput, TimelineSolveCommandOutput,
};
pub use command_eval::{
    evaluate_timeline_command_cli_render_with_session,
    evaluate_timeline_command_output_with_session,
    evaluate_timeline_invocation_cli_actions_with_session, extract_timeline_invocation_input,
};
pub use command_projection::timeline_command_output_from_solver;
pub use render::TimelineCliRender;
pub use render_api::{render_simplify_timeline_html, render_solve_timeline_html};
pub use simplify::{TimelineHtml, VerbosityLevel};
pub use solve::SolveTimelineHtml;

#[cfg(test)]
mod tests;
