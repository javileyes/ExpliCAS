//! Solve and simplify command APIs re-exported for session clients.

pub use crate::repl_solve_runtime::{
    evaluate_full_simplify_command_lines_on_runtime as evaluate_full_simplify_command_lines_on_repl_core,
    evaluate_solve_command_message_on_runtime as evaluate_solve_command_message_on_repl_core,
    ReplSolveRuntimeContext,
};
pub use crate::solve_command_errors::{
    format_solve_command_error_message, format_solve_prepare_error_message,
};
pub use crate::solve_command_eval_core::SolveCommandEvalOutput;
pub use crate::solve_display_lines::format_solve_command_eval_lines;
pub use crate::solve_verify_display::format_verify_summary_lines;
pub use cas_solver_core::solve_command_types::{
    SolveCommandEvalError, SolveCommandInput, SolvePrepareError,
};
