use crate::{
    apply_assignment_with_context, evaluate_assignment_command_with,
    evaluate_let_assignment_command_with, format_assignment_command_output_message,
    AssignmentApplyContext, AssignmentCommandOutput, Simplifier,
};

/// Evaluate assignment command pieces and return a typed output payload.
pub fn evaluate_assignment_command_with_context<C: AssignmentApplyContext>(
    context: &mut C,
    simplifier: &mut Simplifier,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<AssignmentCommandOutput, String> {
    evaluate_assignment_command_with(name, expr_str, lazy, |name, expr_str, lazy| {
        apply_assignment_with_context(context, simplifier, name, expr_str, lazy)
    })
}

/// Evaluate `let ...` command tail and return assignment output payload.
pub fn evaluate_let_assignment_command_with_context<C: AssignmentApplyContext>(
    context: &mut C,
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<AssignmentCommandOutput, String> {
    evaluate_let_assignment_command_with(input, |name, expr_str, lazy| {
        apply_assignment_with_context(context, simplifier, name, expr_str, lazy)
    })
}

/// Evaluate assignment command pieces and return formatted user-facing message.
pub fn evaluate_assignment_command_message_with_context<C: AssignmentApplyContext>(
    context: &mut C,
    simplifier: &mut Simplifier,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<String, String> {
    let output =
        evaluate_assignment_command_with_context(context, simplifier, name, expr_str, lazy)?;
    let rendered = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: output.expr
        }
    );
    Ok(format_assignment_command_output_message(&output, &rendered))
}

/// Evaluate `let ...` command tail and return formatted user-facing message.
pub fn evaluate_let_assignment_command_message_with_context<C: AssignmentApplyContext>(
    context: &mut C,
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<String, String> {
    let output = evaluate_let_assignment_command_with_context(context, simplifier, input)?;
    let rendered = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: output.expr
        }
    );
    Ok(format_assignment_command_output_message(&output, &rendered))
}
