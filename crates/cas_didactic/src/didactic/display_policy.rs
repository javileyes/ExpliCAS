mod classification;
mod plans;
mod render_lines;
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use plans::build_cli_substeps_render_plan;

/// Display verbosity mode for didactic simplification rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepDisplayMode {
    None,
    Succinct,
    Normal,
    Verbose,
}

pub(crate) use self::classification::classify_sub_steps;
pub use self::classification::SubStepClassification;
pub(crate) use self::plans::build_timeline_substeps_render_plan;
pub use self::plans::{CliSubstepsRenderPlan, TimelineSubstepsRenderPlan};
pub(crate) use self::render_lines::{render_cli_enriched_substeps_lines, CliSubstepsRenderState};

pub(crate) use super::latex_plain_text::latex_to_plain_text;
pub(crate) use super::{EnrichedStep, SubStep};
