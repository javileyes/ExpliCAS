use crate::{
    evaluate_history_command_lines_from_history_with_context,
    evaluate_vars_command_lines_from_bindings_with_context, BindingsContext,
    HistoryOverviewContext,
};

use super::ReplSessionViewRuntimeContext;

/// Render `vars` command output using runtime state/context.
pub fn evaluate_vars_command_message_on_runtime<C>(context: &C) -> String
where
    C: ReplSessionViewRuntimeContext,
    C::State: BindingsContext,
{
    evaluate_vars_command_lines_from_bindings_with_context(
        context.state(),
        context.simplifier_context(),
    )
    .join("\n")
}

/// Render `history` command output using runtime state/context.
pub fn evaluate_history_command_message_on_runtime<C>(context: &C) -> String
where
    C: ReplSessionViewRuntimeContext,
    C::State: HistoryOverviewContext,
{
    evaluate_history_command_lines_from_history_with_context(
        context.state(),
        context.simplifier_context(),
    )
    .join("\n")
}
