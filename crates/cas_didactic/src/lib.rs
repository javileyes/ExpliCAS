//! Didactic/timeline facade crate.
//!
//! During migration this crate centralizes didactic and timeline APIs so
//! frontends can consume them without importing `cas_engine` root exports.

pub mod didactic;
#[cfg(test)]
mod events_tests;
mod runtime_bridge;
mod step_payload_render;
mod step_payloads;
pub mod timeline;

pub(crate) use runtime_bridge as runtime;

pub use cas_engine::to_display_steps;
pub use cas_engine::{DisplayEvalSteps, ImportanceLevel, PathStep, Step};
pub use cas_solver_core::engine_event_collector::EngineEventCollector;
pub use cas_solver_core::eval_option_axes::Language;
pub use cas_solver_core::step_types::pathsteps_to_expr_path;
pub use didactic::{
    enrich_steps, format_cli_simplification_steps_with_simplifier, CliSubstepsRenderPlan,
    EnrichedStep, StepDisplayMode, SubStep, TimelineSubstepsRenderPlan,
};
pub use step_payloads::{
    collect_step_payloads, collect_step_payloads_with_events,
    collect_step_payloads_with_events_localized,
};
pub use timeline::{
    evaluate_timeline_invocation_cli_actions_with_session, html_escape, latex_escape,
    render_simplify_timeline_cli_output, render_simplify_timeline_html,
    render_solve_timeline_cli_output, render_solve_timeline_html, SolveTimelineHtml,
    TimelineCliAction, TimelineCliRender, TimelineCommandOutput, TimelineHtml,
    TimelineSimplifyCommandOutput, TimelineSolveCommandOutput, VerbosityLevel, TIMELINE_HTML_FILE,
};
