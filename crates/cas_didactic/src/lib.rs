//! Didactic/timeline facade crate.
//!
//! During migration this crate re-exports didactic rendering APIs from `cas_engine`.

pub use cas_engine::didactic::*;
pub use cas_engine::eval_step_pipeline::to_display_steps;
pub use cas_engine::step::DisplayEvalSteps;
pub use cas_engine::timeline::*;
