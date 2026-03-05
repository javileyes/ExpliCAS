//! Local aliases for step-trace runtime types.
//!
//! Keeping step-related aliases local avoids routing these through
//! `engine_exports` during migration.

pub type ImportanceLevel = cas_solver_core::step_types::ImportanceLevel;
pub type PathStep = cas_solver_core::step_types::PathStep;
pub type Step = cas_solver_core::step_model::Step;
pub type StepCategory = cas_solver_core::step_types::StepCategory;
