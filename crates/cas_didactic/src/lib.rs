//! Didactic/timeline facade crate.
//!
//! During migration this crate centralizes didactic and timeline APIs so
//! frontends can consume them without importing `cas_engine` root exports.

pub mod didactic;
pub mod events;
mod step_payload_render;
mod step_payloads;
pub mod timeline;

pub use cas_solver::to_display_steps;
pub use cas_solver::{pathsteps_to_expr_path, DisplayEvalSteps, ImportanceLevel, PathStep, Step};
pub use didactic::{
    build_cli_substeps_render_plan, build_timeline_substeps_render_plan, enrich_steps,
    format_cli_simplification_steps, format_cli_simplification_steps_with_simplifier,
    get_standalone_substeps, is_high_or_higher_step, is_medium_or_higher_step, latex_to_plain_text,
    CliSubstepsRenderPlan, EnrichedStep, StepDisplayMode, SubStep, TimelineSubstepsRenderPlan,
};
pub use events::EngineEventCollector;
pub use step_payloads::{collect_step_payloads, collect_step_payloads_with_events};
pub use timeline::{
    evaluate_timeline_command_cli_render_with_session,
    evaluate_timeline_command_output_with_session,
    evaluate_timeline_invocation_cli_actions_with_session, extract_timeline_invocation_input,
    html_escape, latex_escape, render_simplify_timeline_cli_output, render_simplify_timeline_html,
    render_solve_timeline_cli_output, render_solve_timeline_html,
    render_timeline_command_cli_output, timeline_cli_actions_from_render,
    timeline_command_output_from_solver, SolveTimelineHtml, TimelineCliAction, TimelineCliRender,
    TimelineCommandOutput, TimelineHtml, TimelineSimplifyCommandOutput, TimelineSolveCommandOutput,
    VerbosityLevel, TIMELINE_HTML_FILE,
};
