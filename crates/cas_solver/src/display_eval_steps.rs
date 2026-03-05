//! Local alias for display-ready eval steps.
//!
//! Keeping this alias inside `cas_solver` lets us control public re-exports
//! without routing through `engine_exports`.

pub type DisplayEvalSteps = cas_solver_core::display_steps::DisplaySteps<crate::Step>;

/// Construct display-ready step wrapper from cleaned engine steps.
pub fn build_display_eval_steps(steps: Vec<crate::Step>) -> DisplayEvalSteps {
    cas_solver_core::display_steps::DisplaySteps(steps)
}
