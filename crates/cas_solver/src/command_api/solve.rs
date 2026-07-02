//! Solve command entrypoints exposed for CLI/frontends.

pub use crate::solution_display::{
    display_interval, display_solution_set, is_pure_residual_otherwise,
};
pub use crate::solve_command_eval_core::{
    evaluate_solve_command_with_session, SolveCommandEvalError, SolveCommandEvalOutput,
};
pub use crate::solve_command_session_eval::evaluate_solve_command_lines_with_session;
pub use crate::solve_input_parse_parse::parse_solve_command_input;
pub use crate::solve_input_parse_prepare::prepare_timeline_solve_equation;
pub use crate::solve_render_config::{SolveCommandRenderConfig, SolveDisplayMode};
pub use cas_api_models::{SolveCommandInput, SolvePrepareError, TimelineCommandInput};
