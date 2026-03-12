//! Solve-budget-facing API for session/repl consumers.

pub use crate::options_budget_eval::{
    apply_solve_budget_command, evaluate_solve_budget_command_message,
};
pub use crate::options_budget_format::format_solve_budget_command_message;
pub use crate::repl_session_runtime::evaluate_solve_budget_command_message_on_runtime as evaluate_solve_budget_command_message_on_repl_core;
pub use cas_solver_core::session_runtime::SolveBudgetCommandResult;
