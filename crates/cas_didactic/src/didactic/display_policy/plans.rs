mod cli;
mod shared;
mod timeline;
mod types;

use super::classification::classify_sub_steps;
pub use types::{CliSubstepsRenderPlan, TimelineSubstepsRenderPlan};

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
