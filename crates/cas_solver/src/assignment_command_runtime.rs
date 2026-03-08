mod eval;
mod message;

use crate::{AssignmentApplyContext, AssignmentCommandOutput, Simplifier};

/// Evaluate assignment command pieces and return a typed output payload.
pub fn evaluate_assignment_command_with_context<C: AssignmentApplyContext>(
    context: &mut C,
    simplifier: &mut Simplifier,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<AssignmentCommandOutput, String> {
    eval::evaluate_assignment_command_with_context(context, simplifier, name, expr_str, lazy)
}

/// Evaluate `let ...` command tail and return assignment output payload.
pub fn evaluate_let_assignment_command_with_context<C: AssignmentApplyContext>(
    context: &mut C,
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<AssignmentCommandOutput, String> {
    eval::evaluate_let_assignment_command_with_context(context, simplifier, input)
}

/// Evaluate assignment command pieces and return formatted user-facing message.
pub fn evaluate_assignment_command_message_with_context<C: AssignmentApplyContext>(
    context: &mut C,
    simplifier: &mut Simplifier,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<String, String> {
    message::evaluate_assignment_command_message_with_context(
        context, simplifier, name, expr_str, lazy,
    )
}

/// Evaluate `let ...` command tail and return formatted user-facing message.
pub fn evaluate_let_assignment_command_message_with_context<C: AssignmentApplyContext>(
    context: &mut C,
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<String, String> {
    message::evaluate_let_assignment_command_message_with_context(context, simplifier, input)
}
