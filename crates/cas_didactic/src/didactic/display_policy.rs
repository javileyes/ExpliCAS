mod classification;
mod plans;
mod render_lines;

/// Display verbosity mode for didactic simplification rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepDisplayMode {
    None,
    Succinct,
    Normal,
    Verbose,
}

pub use self::classification::{classify_sub_steps, SubStepClassification};
pub use self::plans::{
    build_cli_substeps_render_plan, build_timeline_substeps_render_plan, CliSubstepsRenderPlan,
    TimelineSubstepsRenderPlan,
};
pub(crate) use self::render_lines::{render_cli_enriched_substeps_lines, CliSubstepsRenderState};

pub(crate) use super::latex_plain_text::latex_to_plain_text;
pub(crate) use super::{EnrichedStep, SubStep};
