use crate::bindings_command::BindingsContext;
use crate::history_delete::{evaluate_delete_history_command_message, HistoryDeleteContext};
use crate::options_budget_eval::{evaluate_solve_budget_command_message, SolveBudgetContext};
use crate::session_api::bindings::evaluate_clear_command_lines;

use super::ReplSessionStateMutRuntimeContext;

/// Evaluate `clear` command lines against runtime state.
pub fn evaluate_clear_command_lines_on_runtime<C>(context: &mut C, line: &str) -> Vec<String>
where
    C: ReplSessionStateMutRuntimeContext,
    C::State: BindingsContext,
{
    evaluate_clear_command_lines(context.state_mut(), line)
}

/// Evaluate `del` command message against runtime state.
pub fn evaluate_delete_history_command_message_on_runtime<C>(context: &mut C, line: &str) -> String
where
    C: ReplSessionStateMutRuntimeContext,
    C::State: HistoryDeleteContext,
{
    evaluate_delete_history_command_message(context.state_mut(), line)
}

/// Evaluate `budget ...` command message against runtime session state.
pub fn evaluate_solve_budget_command_message_on_runtime<C>(context: &mut C, line: &str) -> String
where
    C: ReplSessionStateMutRuntimeContext,
    C::State: SolveBudgetContext,
{
    evaluate_solve_budget_command_message(context.state_mut(), line)
}
