//! Backward-compatible step facade.
//!
//! Canonical step data types now live in `cas_solver_core`.

pub use cas_solver_core::step_model::Step;
pub use cas_solver_core::step_types::{
    pathsteps_to_expr_path, ImportanceLevel, PathStep, StepCategory, SubStep,
};

/// Display-ready steps after cleanup/enrichment.
pub type DisplayEvalSteps = cas_solver_core::display_steps::DisplaySteps<Step>;
