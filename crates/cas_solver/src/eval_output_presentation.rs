//! Presentation helpers for session-backed eval output models.

pub(crate) use crate::eval_output_presentation_conditions::{
    collect_output_required_conditions, collect_output_required_display, collect_output_warnings,
};
pub(crate) use crate::eval_output_presentation_solution::{
    format_output_solution_set, solution_set_to_output_latex,
};
pub(crate) use crate::eval_output_presentation_solve::{
    collect_output_solve_steps, format_output_input_latex,
};
