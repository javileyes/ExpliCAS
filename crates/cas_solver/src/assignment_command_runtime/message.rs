use crate::{format_assignment_command_output_message, AssignmentApplyContext, Simplifier};

pub(super) fn evaluate_assignment_command_message_with_context<C: AssignmentApplyContext>(
    context: &mut C,
    simplifier: &mut Simplifier,
    name: &str,
    expr_str: &str,
    lazy: bool,
) -> Result<String, String> {
    let output = super::eval::evaluate_assignment_command_with_context(
        context, simplifier, name, expr_str, lazy,
    )?;
    let rendered = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: output.expr
        }
    );
    Ok(format_assignment_command_output_message(&output, &rendered))
}

pub(super) fn evaluate_let_assignment_command_message_with_context<C: AssignmentApplyContext>(
    context: &mut C,
    simplifier: &mut Simplifier,
    input: &str,
) -> Result<String, String> {
    let output =
        super::eval::evaluate_let_assignment_command_with_context(context, simplifier, input)?;
    let rendered = format!(
        "{}",
        cas_formatter::DisplayExpr {
            context: &simplifier.context,
            id: output.expr
        }
    );
    Ok(format_assignment_command_output_message(&output, &rendered))
}
