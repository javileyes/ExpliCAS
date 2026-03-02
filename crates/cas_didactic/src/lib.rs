//! Didactic/timeline facade crate.
//!
//! During migration this crate centralizes didactic and timeline APIs so
//! frontends can consume them without importing `cas_engine` root exports.

pub mod didactic;
pub mod eval_json_steps;
pub mod timeline;

pub use cas_solver::to_display_steps;
pub use cas_solver::{pathsteps_to_expr_path, DisplayEvalSteps, ImportanceLevel, PathStep, Step};
pub use didactic::{
    build_cli_substeps_render_plan, enrich_steps, format_cli_simplification_steps,
    get_standalone_substeps, latex_to_plain_text, CliSubstepsRenderPlan, EnrichedStep,
    StepDisplayMode, SubStep,
};
pub use eval_json_steps::collect_eval_json_steps;
pub use timeline::{
    html_escape, latex_escape, render_simplify_timeline_cli_output, render_simplify_timeline_html,
    render_solve_timeline_cli_output, render_solve_timeline_html,
    render_timeline_command_cli_output, SolveTimelineHtml, TimelineCliRender,
    TimelineCommandOutput, TimelineHtml, TimelineSimplifyCommandOutput, TimelineSolveCommandOutput,
    VerbosityLevel, TIMELINE_HTML_FILE,
};
