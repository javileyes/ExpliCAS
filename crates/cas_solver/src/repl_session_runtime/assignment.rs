use crate::{
    evaluate_assignment_command_message_with_context,
    evaluate_let_assignment_command_message_with_context, AssignmentApplyContext,
};

use super::ReplSessionSimplifierRuntimeContext;

/// Evaluate `let ...` command against runtime and return user-facing message.
pub fn evaluate_let_assignment_command_message_on_runtime<C>(
    context: &mut C,
    input: &str,
) -> Result<String, String>
where
    C: ReplSessionSimplifierRuntimeContext,
    C::State: AssignmentApplyContext,
{
    context.with_state_and_simplifier_mut(|state, simplifier| {
        evaluate_let_assignment_command_message_with_context(state, simplifier, input)
    })
}

/// Evaluate assignment command against runtime and return user-facing message.
pub fn evaluate_assignment_command_message_on_runtime<C>(
    context: &mut C,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<String, String>
where
    C: ReplSessionSimplifierRuntimeContext,
    C::State: AssignmentApplyContext,
{
    context.with_state_and_simplifier_mut(|state, simplifier| {
        evaluate_assignment_command_message_with_context(state, simplifier, name, expr_str, lazy)
    })
}
