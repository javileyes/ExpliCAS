use crate::{DisplayEvalSteps, Step};

/// Convert raw eval steps to display-ready, cleaned steps.
pub fn to_display_steps(raw_steps: Vec<Step>) -> DisplayEvalSteps {
    cas_solver_core::eval_step_pipeline::to_display_eval_steps(raw_steps)
}
