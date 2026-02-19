//! Didactic/timeline facade crate.
//!
//! During migration this crate re-exports didactic rendering APIs from `cas_engine`.

pub use cas_engine::step::{
    pathsteps_to_expr_path, DisplayEvalSteps, ImportanceLevel, PathStep, Step,
};
pub use cas_engine::to_display_steps;
pub use cas_engine::{enrich_steps, get_standalone_substeps, EnrichedStep, SubStep};
pub use cas_engine::{html_escape, latex_escape, SolveTimelineHtml, TimelineHtml, VerbosityLevel};
