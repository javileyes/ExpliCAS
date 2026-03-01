//! Didactic/timeline facade crate.
//!
//! During migration this crate centralizes didactic and timeline APIs so
//! frontends can consume them without importing `cas_engine` root exports.

pub mod didactic;
pub mod timeline;

pub use cas_solver::to_display_steps;
pub use cas_solver::{pathsteps_to_expr_path, DisplayEvalSteps, ImportanceLevel, PathStep, Step};
pub use didactic::{enrich_steps, get_standalone_substeps, EnrichedStep, SubStep};
pub use timeline::{html_escape, latex_escape, SolveTimelineHtml, TimelineHtml, VerbosityLevel};
