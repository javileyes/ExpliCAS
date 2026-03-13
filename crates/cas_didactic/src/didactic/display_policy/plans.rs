mod cli;
mod shared;
mod timeline;

use super::classification::classify_sub_steps;

/// Rendering hints for CLI sub-step blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CliSubstepsRenderPlan {
    /// Optional category header to display before sub-steps.
    pub header: Option<&'static str>,
    /// If true, this block should be shown only once (deduplicated across steps).
    pub dedupe_once: bool,
}

/// Rendering hints for timeline sub-step blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineSubstepsRenderPlan {
    /// Category header to display before sub-steps.
    pub header: &'static str,
    /// If true, this block should be shown only once globally.
    pub dedupe_once: bool,
}

/// Build a CLI rendering plan for enriched sub-steps.
pub fn build_cli_substeps_render_plan(
    sub_steps: &[crate::didactic::SubStep],
) -> CliSubstepsRenderPlan {
    cli::build_cli_substeps_render_plan(classify_sub_steps(sub_steps), shared::should_dedupe_once)
}

/// Build a timeline rendering plan for enriched sub-steps.
pub fn build_timeline_substeps_render_plan(
    sub_steps: &[crate::didactic::SubStep],
) -> TimelineSubstepsRenderPlan {
    timeline::build_timeline_substeps_render_plan(
        classify_sub_steps(sub_steps),
        shared::should_dedupe_once,
    )
}
