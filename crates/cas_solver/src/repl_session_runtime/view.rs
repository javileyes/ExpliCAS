use crate::bindings_command::BindingsContext;
use crate::history_overview::HistoryOverviewContext;
use crate::session_api::bindings::evaluate_vars_command_lines_with_context;
use crate::session_api::history::evaluate_history_command_lines_with_context;

use super::ReplSessionViewRuntimeContext;

/// Render `vars` command output using runtime state/context.
pub fn evaluate_vars_command_message_on_runtime<C>(context: &C) -> String
where
    C: ReplSessionViewRuntimeContext,
    C::State: BindingsContext,
{
    evaluate_vars_command_lines_with_context(context.state(), context.simplifier_context())
        .join("\n")
}

/// Render `history` command output using runtime state/context.
pub fn evaluate_history_command_message_on_runtime<C>(context: &C) -> String
where
    C: ReplSessionViewRuntimeContext,
    C::State: HistoryOverviewContext,
{
    evaluate_history_command_lines_with_context(context.state(), context.simplifier_context())
        .join("\n")
}
